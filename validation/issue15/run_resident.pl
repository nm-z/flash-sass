#!/usr/bin/env perl
# Validate a leased resident core, retaining its CUDA contexts and IPC weights.
# The caller queues on the shared flock and passes that locked description.
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Cwd qw(abs_path);
use File::Path qw(make_path);
use File::Copy qw(copy);
use File::Spec;
use FindBin;
use lib $FindBin::Bin;
use EngineGuard qw(borrow_lock process_start verify_peer);
use ResidentConfig qw(parse_config validation_plan file_fnv);
use IO::Socket::UNIX;
use IO::Select;
use IO::Handle;
use Socket qw(SOCK_STREAM);
use POSIX qw(WNOHANG _exit strftime setpgid sigprocmask SIG_BLOCK SIG_SETMASK SIGINT SIGTERM SIGHUP);
use Time::HiRes qw(time sleep);
use JSON::PP;
use Digest::SHA;

my ($source, $out, $revision, $pid, $start_tick, $lease_fd, $baseline_sha);
my $socket = '/tmp/flash.sock';
GetOptions('source=s'=>\$source, 'out=s'=>\$out, 'source-sha=s'=>\$revision,
    'resident-pid=i'=>\$pid, 'resident-start=s'=>\$start_tick,
    'lease-fd=i'=>\$lease_fd, 'baseline-engine-sha256=s'=>\$baseline_sha,
    'socket=s'=>\$socket) or die "Invalid options\n";
defined($_) && length($_) or die
    "Required: --source --out --source-sha --resident-pid --resident-start --lease-fd --baseline-engine-sha256\n"
    for ($source, $out, $revision, $pid, $start_tick, $lease_fd, $baseline_sha);
die "Unexpected arguments\n" if @ARGV;
die "Full source SHA and baseline artifact SHA required\n"
    unless $revision =~ /\A[0-9a-f]{40}\z/ && $baseline_sha =~ /\A[0-9a-f]{64}\z/;
die "Invalid resident identity\n" unless $pid > 1 && $start_tick =~ /\A[0-9]+\z/;
$source = abs_path($source) // die "Missing built source\n";
die "Evidence directory already exists\n" if -e $out;
make_path($out); $out = abs_path($out);
STDOUT->autoflush(1);
my $pretty = JSON::PP->new->canonical->pretty;
my $state = {status=>'RUNNING', source=>$source, source_sha=>$revision,
    shared_lock=>'/tmp/flash-sass-engine-validation.lock', socket=>$socket,
    resident_pid=>$pid, resident_start_tick=>$start_tick, commands=>[], protocol=>[],
    started_utc=>strftime('%Y-%m-%dT%H:%M:%SZ', gmtime)};
my ($lease, $baseline, $plan, $logical_library, $cubin_file, $mutated, $restored);
my ($trace_probe_required, $settings_changed);
my $restoring = 0;
my %children;
$SIG{PIPE} = 'IGNORE';
$SIG{INT} = $SIG{TERM} = $SIG{HUP} = sub { die "Interrupted\n" };

sub utc { strftime('%Y-%m-%dT%H:%M:%SZ', gmtime) }
sub check { die "$_[1]\n" unless $_[0] }
sub read_file {
    open my $f, '<:raw', $_[0] or die "Read $_[0]: $!";
    local $/; my $s = <$f>; close $f; return $s;
}
sub save {
    my ($name, $s) = @_;
    open my $f, '>:raw', "$out/$name" or die "Write $name: $!";
    print {$f} $s or die "Write $name: $!";
    close $f or die "Close $name: $!";
}
sub persist {
    my $ok = eval {
        save('results.json.tmp', $pretty->encode($state));
        rename "$out/results.json.tmp", "$out/results.json" or die "Rename journal: $!";
        1;
    };
    if (!$ok) {
        my $error = $@;
        die $error unless $restoring;
        warn "Restoration continues despite journal failure: $error";
    }
}
sub sha {
    open my $f, '<:raw', $_[0] or die "Hash $_[0]: $!";
    my $h = Digest::SHA->new(256); $h->addfile($f); close $f; return $h->hexdigest;
}
sub safe_token {
    check(defined($_[0]) && length($_[0]) && $_[0] !~ /[\s\x00-\x1f\x7f=]/,
          'Unsafe line-protocol token');
    return $_[0];
}
sub absolute_from_core {
    my ($path) = @_;
    safe_token($path);
    return abs_path(File::Spec->file_name_is_absolute($path) ? $path :
                    "$state->{core_cwd}/$path") // die "Missing baseline artifact: $path\n";
}
sub live_image {
    my %images;
    for (split /\n/, read_file("/proc/$pid/maps")) {
        my @fields = split /\s+/;
        $images{$fields[5]} = 1 if defined($fields[5])
            && $fields[5] =~ m{\A/tmp/flashcore-\Q$pid\E-gen[0-9]+\.so\z};
    }
    check(keys(%images) == 1, 'Resident engine image is missing or ambiguous');
    return (keys %images)[0];
}
sub exchange {
    my ($command) = @_;
    check($command !~ /[\r\n\x00]/, 'Invalid command boundary');
    my $s = IO::Socket::UNIX->new(Type=>SOCK_STREAM, Peer=>$socket)
        or die "Connect resident core: $!";
    my ($ok, $error) = verify_peer($s, $pid, $start_tick);
    check($ok, $error // 'Resident peer verification failed');
    $s->autoflush(1);
    push @{$state->{protocol}}, {utc=>utc(), direction=>'send', line=>$command};
    persist();
    print {$s} "$command\n" or die "Write resident command: $!";
    my $select = IO::Select->new($s);
    my $buffer = '';
    while (index($buffer, "\n") < 0) {
        check($select->can_read(120), 'Resident response timeout');
        my $n = sysread($s, my $bytes, 16384);
        check(defined($n) && $n > 0, 'Resident response closed early');
        $buffer .= $bytes;
        check(length($buffer) <= 16384, 'Oversized resident response');
    }
    close $s;
    check($buffer =~ /\A[^\r\n]+\n\z/, 'Unexpected resident response framing');
    $buffer =~ s/\n\z//;
    push @{$state->{protocol}}, {utc=>utc(), direction=>'receive', line=>$buffer};
    persist();
    return $buffer;
}
sub reload_command {
    my ($library, $settings) = @_;
    return join(' ', 'reload', safe_token($library),
        map { safe_token($_) . '=' . safe_token("$settings->{$_}") } sort keys %$settings);
}
sub stop_child {
    my ($child) = @_;
    return unless defined($child) && $children{$child};
    my $entry = $children{$child};
    $entry->{reaped} = 1 if !$entry->{reaped} && waitpid($child, WNOHANG) != 0;
    # The private frontend may outlive a failed checker. Retain ownership of
    # its process group even after the direct child has been reaped.
    if (kill 0, -$child) {
        kill 'TERM', -$child;
        my $until = time() + 12;
        while (time() < $until && kill(0, -$child)) {
            $entry->{reaped} = 1 if !$entry->{reaped} && waitpid($child, WNOHANG) != 0;
            sleep .05;
        }
        kill 'KILL', -$child if kill(0, -$child);
    }
    waitpid($child, 0) unless $entry->{reaped};
    delete $children{$child};
}
sub command {
    my ($name, $timeout, @argv) = @_;
    my $began = time();
    my $blocked = POSIX::SigSet->new(SIGINT, SIGTERM, SIGHUP);
    my $previous = POSIX::SigSet->new;
    sigprocmask(SIG_BLOCK, $blocked, $previous) or die "Block spawn signals: $!";
    my $child = fork();
    if (!defined($child)) {
        my $error = "$!"; sigprocmask(SIG_SETMASK, $previous);
        die "Fork: $error";
    }
    if (!$child) {
        $SIG{INT} = $SIG{TERM} = $SIG{HUP} = 'DEFAULT';
        setpgid(0, 0) == 0 or _exit(125);
        open STDIN, '<', '/dev/null' or _exit(125);
        open STDOUT, '>:raw', "$out/$name.log" or _exit(126);
        open STDERR, '>&', \*STDOUT or _exit(126);
        chdir "$source/kern" or _exit(126);
        sigprocmask(SIG_SETMASK, $previous) or _exit(125);
        exec {$argv[0]} @argv or _exit(127);
    }
    setpgid($child, $child);
    $children{$child} = {reaped=>0};
    sigprocmask(SIG_SETMASK, $previous) or die "Restore spawn signal mask: $!";
    my ($rc, $timed_out);
    while (1) {
        if (waitpid($child, WNOHANG) == $child) {
            $rc = ($? & 127) ? 128 + ($? & 127) : $? >> 8;
            $children{$child}{reaped} = 1;
            stop_child($child);
            last;
        }
        if (time() - $began > $timeout) {
            $timed_out = 1; stop_child($child); $rc = 124; last;
        }
        sleep .05;
    }
    push @{$state->{commands}}, {name=>$name, argv=>\@argv, exit_code=>$rc,
        timed_out=>$timed_out ? JSON::PP::true : JSON::PP::false, seconds=>time()-$began};
    persist();
    print utc(), " $name exit=$rc\n";
    check(!$rc, "$name failed (exit $rc); see $out/$name.log");
}
sub check_original_artifacts {
    check(sha($logical_library) eq $baseline_sha, 'Baseline logical library changed');
    check(sha($cubin_file) eq $state->{baseline}{cubin_sha256}, 'Baseline cubin changed');
}
sub verify_candidate {
    my ($pair) = @_;
    my $config = parse_config(exchange('config'));
    check($config->{rev} eq $revision, 'Candidate source revision differs');
    for my $field (qw(kv_host ple_disk layers nblk dies attn_dies split attn_map prefill_chunk)) {
        check($config->{$field} eq $baseline->{$field}, "Candidate changed $field");
    }
    check($config->{maxt} == 512 && $config->{img_rows} == 1
        && $config->{two_position} == $pair && $config->{trace_tokens} == 4
        && $config->{cubin} eq "$source/kern/die_trace.cubin", 'Candidate runtime configuration differs');
    check(sha(live_image()) eq $state->{artifacts}{'libengine.so'}, 'Loaded candidate library differs');
    check($config->{cubin_fnv} eq $state->{candidate_cubin_fnv}, 'Loaded candidate cubin differs');
    return $config;
}
sub restore {
    return if !$mutated || $restored;
    $state->{status} = 'RESTORING'; persist();
    check_original_artifacts();
    check(exchange(reload_command($logical_library, $settings_changed ? $plan->{restore} : {})) eq 'ok: engine reloaded',
          'Baseline engine reload failed');
    my $config = parse_config(exchange('config'));
    for my $field (keys %$baseline) {
        check(defined($config->{$field}) && $config->{$field} eq $baseline->{$field},
              "Restored configuration differs: $field");
    }
    check(sha(live_image()) eq $baseline_sha, 'Restored resident image differs');
    my ($current_start, $error) = process_start($pid);
    check(defined($current_start) && $current_start eq $start_tick,
          $error // 'Resident process changed during restoration');
    $restored = 1;
    $state->{restoration} = {status=>'PASS', config=>$config,
        engine_sha256=>$baseline_sha, verified_utc=>utc()};
    persist();
    print utc(), " baseline library and effective configuration restored\n";
}

persist();
my $ok = eval {
    my $error; ($lease, $error) = borrow_lock($lease_fd, $state->{shared_lock});
    check($lease, $error // 'Missing inherited lease');
    $state->{lock_acquired_utc} = utc(); persist();
    open my $h, '-|', 'hostname' or die "Hostname: $!";
    my $hostname = <$h>; close $h; chomp $hostname;
    check($hostname eq 'archy', 'This recorded resident layout requires archy');
    $state->{hostname} = $hostname;
    $state->{core_cwd} = readlink("/proc/$pid/cwd") // die "Read core cwd: $!";
    $state->{runner_sha256} = sha(abs_path($0));
    $state->{artifacts} = {map { $_=>sha("$source/kern/$_") }
        qw(libengine.so flashd die.cubin die_trace.cubin)};
    $state->{candidate_cubin_fnv} = file_fnv("$source/kern/die_trace.cubin");
    $baseline = parse_config(exchange('config'));
    my $image = live_image();
    check(sha($image) eq $baseline_sha, 'Current resident library differs from approved baseline artifact');
    $trace_probe_required = !exists $baseline->{trace_tokens};
    # Validate the baseline shape before any reload. A missing trace setting
    # is read from the candidate under the unchanged environment below; zero
    # here is only a shape-check placeholder, never an assumed restore value.
    $plan = validation_plan({%$baseline, $trace_probe_required ? (trace_tokens=>'0') : ()},
        $baseline_sha, "$source/kern/die_trace.cubin");

    my $stderr = readlink("/proc/$pid/fd/2") // die "Read core stderr path: $!";
    check(-f $stderr, 'Cannot establish current logical library from core log');
    open my $log, '<:raw', $stderr or die "Read core log: $!";
    my $length = -s $log; seek($log, $length > 262144 ? $length - 262144 : 0, 0) or die "Seek core log: $!";
    my $tail = do { local $/; <$log> }; close $log;
    my @ready = $tail =~ /^flashcore: engine (.+) ready in [^\r\n]+$/mg;
    check(@ready, 'No current engine-ready record');
    $logical_library = absolute_from_core($ready[-1]);
    $cubin_file = absolute_from_core($baseline->{cubin});
    check(file_fnv($cubin_file) eq $baseline->{cubin_fnv},
          'Baseline cubin file differs from the image fingerprint reported by the resident engine');
    $state->{baseline} = {config=>$baseline, private_image=>$image,
        engine_sha256=>$baseline_sha, logical_library=>$logical_library,
        cubin_file=>$cubin_file, cubin_sha256=>sha($cubin_file),
        trace_restore_basis=>$trace_probe_required ? 'pending unchanged-environment readback' : $plan->{trace_restore_basis}};
    check_original_artifacts();
    copy($image, "$out/baseline-libengine.so") or die "Save exact baseline image: $!";
    copy($cubin_file, "$out/baseline-die.cubin") or die "Save baseline cubin: $!";
    check(sha("$out/baseline-libengine.so") eq $baseline_sha, 'Baseline image copy differs');
    check(sha("$out/baseline-die.cubin") eq $state->{baseline}{cubin_sha256},
          'Baseline cubin copy differs');
    save('restore-command.txt', reload_command($logical_library, {}) . "\n");
    $state->{plan} = $plan;
    $state->{restoration_scope} = 'Exact library bytes and effective configuration; reload clears engine snapshots and caches. Startup environment is not used as effective readback.';
    persist();
    command('gpu-before', 10, 'nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm', '--format=csv');
    command('session-fixtures', 30, "$source/kern/trace_session_test");
    my $gguf = '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf';
    command('output-token-regression', 30, "$source/kern/output_tokens_test", $gguf);
    if ($trace_probe_required) {
        $state->{status} = 'READING_UNCHANGED_ENVIRONMENT';
        $mutated = 1; $state->{mutation_started_utc} //= utc(); persist();
        check(exchange(reload_command("$source/kern/libengine.so", {})) eq 'ok: engine reloaded',
              'Unchanged-environment candidate reload failed');
        my $config = parse_config(exchange('config'));
        check($config->{rev} eq $revision && sha(live_image()) eq $state->{artifacts}{'libengine.so'},
              'Trace readback came from a different candidate library');
        for my $field (grep {$_ ne 'rev'} keys %$baseline) {
            check(defined($config->{$field}) && $config->{$field} eq $baseline->{$field},
                  "Unchanged-environment reload changed $field");
        }
        check(exists($config->{trace_tokens}), 'Candidate does not expose the effective trace setting');
        $plan = validation_plan({%$baseline, trace_tokens=>$config->{trace_tokens}},
            $baseline_sha, "$source/kern/die_trace.cubin");
        $state->{unchanged_environment_config} = $config;
        $state->{baseline}{trace_restore_basis} = 'candidate config readback before any environment assignment';
        $state->{plan} = $plan; persist();
    }
    save('restore-command.txt', reload_command($logical_library, $plan->{restore}) . "\n");
    for my $pair (0, 1) {
        check_original_artifacts();
        $state->{status} = "LOADING_PAIR_$pair";
        $mutated = 1; $settings_changed = 1; $state->{mutation_started_utc} //= utc(); persist();
        check(exchange(reload_command("$source/kern/libengine.so", $plan->{candidates}[$pair]))
            eq 'ok: engine reloaded', 'Candidate reload failed');
        $state->{candidate_configs}[$pair] = verify_candidate($pair);
        $state->{status} = "MEASURING_PAIR_$pair"; persist();
        my @resident = ('--socket', $socket, '--source-sha', $revision,
            '--resident-pid', $pid, '--resident-start', $start_tick, '--lease-fd', fileno($lease));
        command("cli-pair$pair", 600, 'perl', "$source/kern/timings_core_check.pl",
            @resident, '--out', "$out/cli-pair$pair");
        command("http-pair$pair", 600, 'perl', "$source/kern/timings_http_real.pl",
            @resident, '--flashd', "$source/kern/flashd", '--gguf', $gguf,
            '--out', "$out/http-pair$pair");
        verify_candidate($pair);
    }
    1;
};
my $failure = $@;
# Signals during the first cleanup must not skip the restore attempt.
local $SIG{INT} = local $SIG{TERM} = local $SIG{HUP} = 'IGNORE';
stop_child($_) for keys %children;
if ($mutated) {
    $restoring = 1;
    while (!eval { restore(); 1 }) {
        my $restore_error = $@;
        $state->{status} = 'RESTORE_FAILED';
        $state->{restore_error} = "$restore_error";
        eval { persist() };
        warn utc(), " RESTORE_FAILED; retaining shared lease: $restore_error";
        # Keep the lease while the operator repairs the named artifact/path.
        # Never admit another validator with unverified resident state.
        sleep 30;
    }
    $restoring = 0;
}
if ($lease) {
    eval { command('gpu-after', 10, 'nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm', '--format=csv') };
    if ($@ && $ok) { $ok = 0; $failure = $@; }
}
$state->{status} = $ok ? 'PASS' : ($mutated ? 'FAIL' : 'BLOCKED');
$state->{error} = "$failure" unless $ok;
$state->{finished_utc} = utc(); persist();
close $lease if $lease; # Never LOCK_UN: the caller also owns this description.
print utc(), " $state->{status} resident CLI/HTTP timing validation\n";
warn $failure unless $ok;
exit($ok ? 0 : $mutated ? 1 : 75);
