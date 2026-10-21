#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use Cwd qw(abs_path);
use File::Spec;
use Fcntl qw(:flock F_GETFD F_SETFD FD_CLOEXEC);
use Getopt::Long;
use JSON::PP;
use POSIX qw(WNOHANG _exit);
use Time::HiRes qw(time sleep);
use Digest::SHA qw(sha256_hex);

require "$FindBin::Bin/run_qsa_model_matrix.pl";

my $J = JSON::PP->new->canonical->pretty;

# Die numbers in statehash are logical ordinals in manifest order, not the
# physical indices in CUDA_VISIBLE_DEVICES. Attention state lives only on its
# owner; recurrent state, PLE and a vocabulary shard live on every die.
sub owned_layers {
    my ($o, $die) = @_;
    return grep { $_ % 4 != 3 || $o->{attn_owners}[int($_ / 4)] == $die }
        0 .. $o->{layers} - 1;
}

sub request_scope {
    my ($o) = @_;
    $o->{visible_list} = numbers($o->{visible}, 0, 255, 'numeric CUDA visibility');
    $o->{dies} = scalar @{$o->{visible_list}};
    die "require one manifest and holder PID per visible die (at most six)\n"
        unless $o->{dies} <= 6 && @{$o->{manifests}} == $o->{dies} &&
               @{$o->{holder_pids}} == $o->{dies};
    my (%paths, %pids);
    for (@{$o->{manifests}}) {
        die "empty manifest path\n" unless length;
        $_ = File::Spec->rel2abs($_);
        die "duplicate manifest path\n" if $paths{$_}++;
    }
    for (@{$o->{holder_pids}}) {
        die "invalid or duplicate holder PID\n" unless /^\d+$/ && $_ > 0 && !$pids{0 + $_}++;
    }
    # A multi-die test must state its topology; never silently reinterpret
    # weighted production arenas as an equal split or move their attention.
    $o->{split_shares} //= $o->{dies} == 1 ? '1' : $ENV{SPLIT_SHARES};
    $o->{attn_map} //= $o->{dies} == 1 ? join(',', (0) x 12) : $ENV{ATTN_MAP};
    die "multi-die requests require --split-shares and --attn-map (or their environment settings)\n"
        unless defined $o->{split_shares} && defined $o->{attn_map};
    my @shares = split /,/, $o->{split_shares}, -1;
    my @owners = split /,/, $o->{attn_map}, -1;
    die "invalid split shares\n" unless @shares == $o->{dies} &&
        !grep { !/^\d+$/ || $_ < 1 || $_ > 1000000 } @shares;
    die "invalid attention map\n" unless @owners == 12 &&
        !grep { !/^\d+$/ || $_ >= $o->{dies} } @owners;
    $o->{visible} = join(',', @{$o->{visible_list}});
    $o->{split_shares} = join(',', map { 0 + $_ } @shares);
    $o->{attn_map} = join(',', map { 0 + $_ } @owners);
    $o->{attn_owners} = [map { 0 + $_ } @owners];
    my %active = map { $owners[$_] + 0 => 1 } 0 .. int($o->{layers} / 4) - 1;
    $o->{attn_dies} = join(',', sort { $a <=> $b } keys %active);
    die "--tokens must be 6..262142\n"
        unless $o->{tokens} >= 6 && $o->{tokens} <= 262142;
    # Leave two prompt-parser slots and one generated position; round only
    # capacity, never the requested visible length.
    $o->{maxt} = 128 * int(($o->{tokens} + 2 + 127) / 128);
    $o->{snapshot_list} = numbers($o->{snapshots}, 1, $o->{tokens} - 1, 'snapshot position');
}

sub request_plan {
    my ($snapshots) = @_;
    $snapshots //= [1, 2, 3, 5];
    my @p = (
        { name => 'fresh-A', prompt => 'A', resume => 0, snap => 0 },
        { name => 'fresh-B', prompt => 'B', resume => 0, snap => 0 },
    );
    for my $s (@$snapshots) {
        push @p,
            { name => "save-$s-A", prompt => 'A', resume => 0, snap => $s },
            { name => "resume-$s-A", prompt => 'A', resume => $s, snap => 0 },
            { name => "reset-$s-B", prompt => 'B', resume => 0, snap => 0 };
    }
    push @p, { name => 'final-A', prompt => 'A', resume => 0, snap => 0 };
    push @p, { name => 'generation-A', prompt => 'G', resume => 0, snap => 0, n_new => 1 };
    return \@p;
}

sub commands_for {
    my ($plan, $tokens) = @_;
    $tokens //= 9;
    my @cmd = ('config');
    for my $r (@$plan) {
        push @cmd, "snap $r->{snap}" if $r->{snap};
        my $first = $r->{prompt} eq 'B' ? 30 : 10;
        my @ids = map { ($first + $_) % 248320 } 0 .. $tokens - 1;
        push @cmd, 'gen ' . ($r->{n_new} // 0) . ' ' . join(' ', @ids), 'statehash';
    }
    push @cmd, 'quit';
    return \@cmd;
}

sub expected_pairs {
    my ($request, $mode, $tokens) = @_;
    $tokens //= 9;
    return 0 if $mode eq 'scalar';
    my $prefix = $tokens - 1;
    # gen 1 uses headless chunks of three, each with a paired prefix and an
    # optional scalar tail. gen 0 retains the head at every position.
    return 2 * (int($prefix / 3) + int(($prefix % 3) / 2)) if $request->{n_new};
    my $r = $request->{resume};
    my $s = $request->{snap};
    return $s ? 2 * int(($s - $r) / 2) + 2 * int(($prefix - $s) / 2)
              : 2 * int(($prefix - $r) / 2);
}

# Only this just-forked PID may be signalled. Input/output are regular files.
sub run_private_core {
    my ($o, $argv, $env, $input, $prefix, $deadline, $lock) = @_;
    my $start = time;
    my $pid = fork();
    die "fork: $!\n" unless defined $pid;
    if (!$pid) {
        close $lock;
        POSIX::close($o->{lock_fd});
        chdir "$o->{repo}/kern" or _exit(126);
        open STDIN, '<', $input or _exit(126);
        open STDOUT, '>', "$prefix.stdout" or _exit(126);
        open STDERR, '>', "$prefix.stderr" or _exit(126);
        delete @ENV{grep { /^(?:FLASH_|MODEL_)/ } keys %ENV};
        @ENV{keys %$env} = values %$env;
        exec {$argv->[0]} @$argv or _exit(127);
    }
    my ($cancel, $timeout, $status) = (0, 0, 0);
    local $SIG{INT} = sub { $cancel = 1 };
    local $SIG{TERM} = sub { $cancel = 1 };
    while (1) {
        my $got = waitpid($pid, WNOHANG);
        if ($got == $pid) { $status = $?; last; }
        die "waitpid: $!\n" if $got < 0;
        if ($cancel || time >= $deadline) {
            $timeout = !$cancel;
            kill 'TERM', $pid;
            my $until = time + 2;
            while (time < $until) {
                $got = waitpid($pid, WNOHANG);
                last if $got == $pid;
                sleep 0.05;
            }
            if ($got != $pid) {
                kill 'KILL', $pid;
                waitpid($pid, 0);
            }
            $status = $?;
            last;
        }
        sleep 0.05;
    }
    return {
        argv => $argv, environment => $env, stdin => $input,
        stdout => "$prefix.stdout", stderr => "$prefix.stderr",
        pid => $pid, exit_code => $status >> 8, signal => $status & 127,
        timed_out => $timeout ? JSON::PP::true : JSON::PP::false,
        interrupted => $cancel ? JSON::PP::true : JSON::PP::false,
        elapsed_seconds => time - $start,
    };
}

sub parse_capture {
    my ($text, $o, $mode, $kv, $plan, $references) = @_;
    my (@records, $config, $tokens, $current, $header);
    for my $line (split /\r?\n/, $text) {
        die "engine reported an error: $line\n" if $line =~ /^error:|\bMISMATCH\b/;
        if ($line =~ /^config /) {
            die "duplicate config response\n" if defined $config;
            my %c;
            while ($line =~ /(?:^|\s)([a-z_]+)=(\S+)/g) {
                die "duplicate config field $1\n" if exists $c{$1};
                $c{$1} = $2;
            }
            my %expected = (
                rev => $o->{revision}, kv_host => "$kv", ple_disk => '1',
                maxt => "$o->{maxt}", layers => "$o->{layers}/48",
                nblk => "$o->{nblk}", dies => "$o->{dies}",
                split => $o->{split_shares}, attn_map => $o->{attn_map},
                attn_dies => $o->{attn_dies},
                two_position => $mode eq 'scalar' ? '0' : '1',
                prefill_chunk => $mode eq 'scalar' ? '0' : '3',
            );
            for my $key (sort keys %expected) {
                die "config $key mismatch: " . ($c{$key} // '<missing>') . "\n"
                    unless defined $c{$key} && $c{$key} eq $expected{$key};
            }
            $config = { raw => $line, fields => \%c,
                        sha256 => sha256_hex($line . "\n") };
        } elsif ($line =~ /^tokens:\s*(.*)$/) {
            die "unexpected duplicate tokens response\n" if defined $tokens;
            my @v = grep { length } split /\s+/, $1;
            my $r = $plan->[scalar @records] or die "extra token response\n";
            # gen N runs nprompt+N positions and emits from nprompt-1.
            my $expected_tokens = $r->{n_new} ? $r->{n_new} + 1 : $o->{tokens};
            die "wrong number of numeric token outputs\n"
                unless @v == $expected_tokens && !grep { !/^\d+$/ || $_ >= 248320 } @v;
            $tokens = [map { 0 + $_ } @v];
        } elsif ($line =~ /^time:/) {
            die "time response without tokens or before preceding statehash\n"
                unless defined $tokens && !defined $current;
            my $r = $plan->[scalar @records]
                or die "extra generation response\n";
            my ($ran, $resumed) =
                $line =~ /(\d+)\s+positions run\s+\((\d+)\s+from the snapshot\b/;
            my ($pairs) = $line =~ /(\d+)\s+paired positions\b/;
            my ($chunks) = $line =~ /(\d+)\s+prefill chunks\b/;
            die "incomplete generation statistics\n"
                unless defined $ran && defined $resumed &&
                       defined $pairs && defined $chunks;
            my $positions = $o->{tokens} + ($r->{n_new} // 0);
            die "$r->{name}: wrong resumed/executed counts\n"
                unless $resumed == $r->{resume} && $ran == $positions - $r->{resume};
            my $expected = expected_pairs($r, $mode, $o->{tokens});
            die "$r->{name}: paired positions $pairs, expected $expected\n"
                unless $pairs == $expected;
            my $expected_chunks = $r->{n_new} && $mode eq 'chunk3'
                ? int(($o->{tokens} - 1 + 2) / 3) : 0;
            die "$r->{name}: headless chunk count mismatch\n"
                unless $chunks == $expected_chunks;
            $current = {
                %$r, tokens => $tokens, time_line => $line,
                positions_run => 0 + $ran, paired_positions => 0 + $pairs,
                prefill_chunks => 0 + $chunks, layer_states => {},
                ple => {}, logits => {},
            };
            undef $tokens;
        } elsif ($line =~
                 /^statehash positions=(\d+)\s+paired_positions=(\d+)\s+prefill_chunks=(\d+)/) {
            die "unexpected/duplicate statehash header\n"
                unless defined $current && !defined $header;
            $header = { positions => 0 + $1, paired => 0 + $2, chunks => 0 + $3 };
        } elsif ($line =~
                 /^statehash die=(\d+)\s+layer=(\d+)\s+state=([0-9a-fA-F]{16})\s*$/) {
            die "state outside current statehash\n" unless $current && $header;
            my ($die, $layer, $hash) = (0 + $1, 0 + $2, lc $3);
            die "unexpected or duplicate layer state\n"
                unless $die < $o->{dies} && $layer < $o->{layers} &&
                       ($layer % 4 != 3 || $o->{attn_owners}[int($layer / 4)] == $die) &&
                       !exists $current->{layer_states}{$die}{$layer};
            $current->{layer_states}{$die}{$layer} = $hash;
        } elsif ($line =~
                 /^statehash die=(\d+)\s+ple=([0-9a-fA-F]{16})\s+logits=([0-9a-fA-F]{16})\s*$/) {
            my ($die, $ple, $logits) = (0 + $1, lc $2, lc $3);
            die "unexpected or duplicate PLE/logits state\n"
                unless $current && $header && $die < $o->{dies} &&
                       !exists $current->{ple}{$die};
            ($current->{ple}{$die}, $current->{logits}{$die}) = ($ple, $logits);
        } elsif ($line =~ /^statehash done\s*$/) {
            die "incomplete statehash\n"
                unless $current && $header;
            for my $die (0 .. $o->{dies} - 1) {
                my @layers = owned_layers($o, $die);
                die "incomplete statehash for die $die\n" unless
                    exists $current->{ple}{$die} && exists $current->{logits}{$die} &&
                    exists $current->{layer_states}{$die} &&
                    scalar(keys %{$current->{layer_states}{$die}}) == @layers;
            }
            die "statehash position/execution counts disagree\n"
                unless $header->{positions} == $o->{tokens} + ($current->{n_new} // 0) &&
                       $header->{paired} == $current->{paired_positions} &&
                       $header->{chunks} == $current->{prefill_chunks};
            my $fingerprint = {
                layer_states => $current->{layer_states},
                ple => $current->{ple}, logits => $current->{logits},
            };
            my $key = $current->{prompt};
            if (!exists $references->{$key}) {
                die "first reference must be a fresh request\n"
                    if $current->{resume} || $current->{snap};
                $references->{$key} = {
                    fingerprint => $fingerprint, tokens => [@{$current->{tokens}}],
                    kv_host => $kv, mode => $mode,
                };
            }
            my $ref = $references->{$key};
            die "$current->{name}: final state/logits fingerprint mismatch\n"
                unless $J->encode($fingerprint) eq $J->encode($ref->{fingerprint});
            if ($current->{n_new}) {
                die "$current->{name}: generated token mismatch\n"
                    unless $J->encode($current->{tokens}) eq $J->encode($ref->{tokens});
            } else {
                for my $i ($current->{resume} .. $o->{tokens} - 1) {
                    die "$current->{name}: token mismatch at position $i\n"
                        unless $current->{tokens}[$i] == $ref->{tokens}[$i];
                }
            }
            $current->{statehash_header} = $header;
            $current->{passed} = JSON::PP::true;
            push @records, $current;
            undef $current;
            undef $header;
        } elsif ($line =~ /^statehash /) {
            die "unrecognized statehash response: $line\n";
        }
    }
    die "missing config or incomplete response sequence\n"
        unless $config && !defined $tokens && !defined $current &&
               @records == @$plan;
    return { config => $config, requests => \@records };
}

sub snapshot_self_test {
    my $status = system($^X, "$FindBin::Bin/test_qsa_engine_requests.pl");
    return $status == -1 ? 127 : ($status & 127) ? 1 : $status >> 8;
}

sub snapshot_main {
    my %o = (
        repo => File::Spec->rel2abs("$FindBin::Bin/.."),
        layers => 8, nblk => 1, modes => 'scalar,chunk3', kv => '0,1',
        tokens => 9, snapshots => '1,2,3,5',
        visible => $ENV{CUDA_VISIBLE_DEVICES} // '7',
        lock_fd => 9, timeout => 600, min_free_mib => 1024,
        min_host_mib => 1024, ple_off => $ENV{FLASH_PLE_OFF} // 192,
    );
    my (@manifests, @holders, @weight_manifests);
    GetOptions(
        'repo=s' => \$o{repo}, 'out=s' => \$o{out},
        'ple=s' => \$o{ple}, 'manifest=s@' => \@manifests,
        'holder-pid=i@' => \@holders, 'revision=s' => \$o{revision},
        'weight-manifest=s@' => \@weight_manifests, 'build-log=s' => \$o{build_log},
        'layers=i' => \$o{layers}, 'nblk=i' => \$o{nblk},
        'tokens=i' => \$o{tokens}, 'snapshots=s' => \$o{snapshots},
        'split-shares=s' => \$o{split_shares}, 'attn-map=s' => \$o{attn_map},
        'modes=s' => \$o{modes}, 'kv=s' => \$o{kv},
        'cuda-visible=s' => \$o{visible}, 'lock-fd=i' => \$o{lock_fd},
        'timeout=i' => \$o{timeout}, 'ple-off=i' => \$o{ple_off},
        'min-free-mib=i' => \$o{min_free_mib},
        'min-host-mib=i' => \$o{min_host_mib}, 'plan' => \$o{plan},
        'self-test' => \$o{self_test},
    ) or die "invalid options\n";
    die "unexpected arguments\n" if @ARGV;
    return snapshot_self_test() if $o{self_test};
    @manifests = ($ENV{QSA_MODEL_MANIFEST})
        if !@manifests && defined $ENV{QSA_MODEL_MANIFEST};
    @holders = ($ENV{QSA_MODEL_HOLDER_PID})
        if !@holders && defined $ENV{QSA_MODEL_HOLDER_PID};
    die "require --ple, --manifest, --holder-pid and --revision\n"
        unless $o{ple} && @manifests && @holders && $o{revision};
    die "invalid scope/options\n"
        unless $o{layers} >= 4 && $o{layers} <= 48 &&
               $o{nblk} >= 1 && $o{nblk} <= 4096 &&
               $o{lock_fd} >= 3 && $o{timeout} >= 1 &&
               $o{timeout} <= 86400 && $o{ple_off} >= 0 &&
               $o{min_free_mib} > 0 && $o{min_host_mib} > 0 &&
               $o{revision} =~ /^[0-9a-f]{7,40}(?:-dirty)?$/;
    my @modes = split /,/, $o{modes};
    die "--modes must be scalar or scalar,chunk3\n"
        unless $o{modes} eq 'scalar' || $o{modes} eq 'scalar,chunk3';
    my $kv = numbers($o{kv}, 0, 1, 'KV placement');
    for my $key (qw(repo ple)) {
        $o{$key} = File::Spec->rel2abs($o{$key});
    }
    $o{manifests} = \@manifests;
    $o{holder_pids} = \@holders;
    request_scope(\%o);
    my $plan = request_plan($o{snapshot_list});
    my $commands = commands_for($plan, $o{tokens});
    my $report = {
        schema => 'flash-sass.private-qsa-request-check.v2',
        status => 'PLAN_ONLY', acceptance_complete => JSON::PP::false,
        scope => ($o{layers} == 48 ? 'Full-model' : 'Partial-model') .
                 ' GPU request/snapshot and paired-launch equivalence.',
        layers => $o{layers}, block_budget => $o{nblk},
        tokens => $o{tokens}, capacity => $o{maxt}, dies => $o{dies},
        cuda_visible_devices => $o{visible}, split_shares => $o{split_shares},
        attn_map => $o{attn_map}, manifests => $o{manifests},
        holder_pids => $o{holder_pids}, snapshots => $o{snapshot_list},
        requested_selection_exclusion => $o{tokens} >= 4 * ($o{nblk} + 1)
            ? JSON::PP::true : JSON::PP::false,
        expected_revision => $o{revision}, commands => $commands,
        request_plan => $plan, cores => [], references => {},
        limitations => [
            'No scalar model or independent checkpoint comparison.',
            'State comparisons use engine FNV fingerprints.',
            'The explicit die layout must match the supplied holder arenas; a planned layout is not measured production-placement coverage.',
            'Manifest hashes alone do not establish checkpoint conversion lineage; attach weight identities with --weight-manifest.',
            'gen 0 is teacher-forced. The separate gen 1 response is compared with the full-head scalar-launch engine mode, not with a CPU reference for generated IDs.',
            'Resumed token comparisons exclude the cleared output prefix.',
        ],
    };
    if ($o{plan}) { print $J->encode($report); return 0; }
    die "--out NEW_DIRECTORY required\n" unless $o{out};
    $o{out} = File::Spec->rel2abs($o{out});
    mkdir $o{out} or die "create fresh output directory: $!\n";
    my $deadline = time + $o{timeout};
    my $result_file = "$o{out}/results.json";
    my $ok = eval {
        # The wrapper retains its reservation; no child inherits this descriptor.
        open my $inherited, '<&=', $o{lock_fd} or die "open inherited lock fd: $!\n";
        my @a = stat $inherited;
        my @b = stat '/tmp/flash-sass-engine-validation.lock';
        die "wrong inherited lock file\n"
            unless @a && @b && $a[0] == $b[0] && $a[1] == $b[1] &&
                   flock($inherited, LOCK_EX | LOCK_NB);
        my $fdflags = fcntl($inherited, F_GETFD, 0);
        die "read lock descriptor flags: $!\n" unless defined $fdflags;
        fcntl($inherited, F_SETFD, $fdflags | FD_CLOEXEC)
            or die "set lock descriptor close-on-exec: $!\n";
        my @files = (
            "$o{repo}/kern/flashcore", "$o{repo}/kern/libengine.so",
            "$o{repo}/kern/die.cubin", @{$o{manifests}}, @weight_manifests,
            "$o{repo}/kern/engine.c", "$o{repo}/kern/flashcore.c",
            "$o{repo}/kern/prog.c", abs_path(__FILE__),
            "$FindBin::Bin/run_qsa_model_matrix.pl",
        );
        push @files, $o{build_log} if defined $o{build_log};
        $report->{file_identities} = [map { identity($_) } @files];
        $report->{git_head} = checked_capture(\%o,
            ['git', 'rev-parse', 'HEAD'], "$o{out}/source-head");
        $report->{git_status} = checked_capture(\%o,
            ['git', 'status', '--porcelain'], "$o{out}/source-status");
        checked_capture(\%o, ['git', 'diff', '--binary', 'HEAD'], "$o{out}/source-diff");
        $report->{source_diff} = identity("$o{out}/source-diff.stdout");
        write_text("$o{out}/commands.txt", join("\n", @$commands) . "\n");
        $report->{commands_identity} = identity("$o{out}/commands.txt");
        $report->{status} = 'RUNNING';
        write_text($result_file, $J->encode($report));
        for my $placement (@$kv) {
            for my $mode (@modes) {
                die "whole-run deadline reached\n" if time >= $deadline;
                my $name = "kv$placement-$mode";
                my %guard = (
                    %o, wait_seconds => 30,
                );
                my $lock = reserve(\%guard, $name);
                die "whole-run deadline reached before core launch\n"
                    if time >= $deadline;
                my %env = (
                    CUDA_VISIBLE_DEVICES => $o{visible},
                    ATTN_MAP => $o{attn_map}, SPLIT_SHARES => $o{split_shares},
                    FLASH_MAXT => "$o{maxt}", FLASH_ENGINE_MAXT => "$o{maxt}",
                    FLASH_LAYERS => "$o{layers}", FLASH_NBLK => "$o{nblk}",
                    FLASH_MAXIMG => '0', FLASH_PLE_DISK => '1',
                    FLASH_PLE_OFF => "$o{ple_off}", FLASH_KV_HOST => "$placement",
                    FLASH_TWO_POSITION => $mode eq 'scalar' ? '0' : '1',
                    FLASH_PREFILL_CHUNK => $mode eq 'scalar' ? '0' : '3',
                );
                for my $key (qw(CUDA_DEVICE_ORDER LD_LIBRARY_PATH)) {
                    $env{$key} = $ENV{$key} if defined $ENV{$key};
                }
                my $argv = [
                    "$o{repo}/kern/flashcore", $o{ple},
                    "$o{repo}/kern/libengine.so", @{$o{manifests}},
                ];
                my $r = run_private_core(
                    \%o, $argv, \%env, "$o{out}/commands.txt",
                    "$o{out}/$name", $deadline, $lock);
                close $lock;
                $r->{name} = $name;
                $r->{stdout_identity} = identity($r->{stdout});
                $r->{stderr_identity} = identity($r->{stderr});
                push @{$report->{cores}}, $r;
                write_text($result_file, $J->encode($report));
                die "$name: private core did not exit successfully\n"
                    if $r->{exit_code} || $r->{signal} ||
                       $r->{timed_out} || $r->{interrupted};
                my $checked = parse_capture(
                    slurp($r->{stdout}), \%o, $mode, $placement,
                    $plan, $report->{references});
                @{$r}{keys %$checked} = values %$checked;
                $r->{passed} = JSON::PP::true;
                write_text($result_file, $J->encode($report));
            }
        }
        $report->{status} = 'SELECTED_REQUEST_CHECKS_PASSED';
        1;
    };
    if (!$ok) {
        $report->{status} = $@ =~ /^BLOCKED:/ ? 'BLOCKED' : 'FAILED';
        $report->{error} = "$@";
    }
    $report->{finished_epoch} = time;
    write_text($result_file, $J->encode($report));
    print "$report->{status}: $result_file\n";
    print STDERR $report->{error} if $report->{error};
    return $ok ? 0 : 1;
}

unless (caller) { exit snapshot_main(); }
1;
