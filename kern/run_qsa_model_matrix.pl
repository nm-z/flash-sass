#!/usr/bin/env perl
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA;
use Fcntl qw(:flock F_GETFD F_SETFD FD_CLOEXEC);
use File::Basename qw(dirname basename);
use File::Spec;
use File::Temp qw(tempdir);
use Getopt::Long qw(GetOptionsFromArray);
use JSON::PP;
use POSIX qw(WNOHANG _exit);
use Time::HiRes qw(time sleep);

my $LOCK = '/tmp/flash-sass-engine-validation.lock';
my $JSON = JSON::PP->new->canonical->pretty;
my $interrupted = 0;
my $inherited_lock_fd;

sub usage {
    return <<'USAGE';
Usage:
  perl kern/run_qsa_model_matrix.pl --repo ROOT --out NEW_DIRECTORY
    --cache TENSOR_CACHE --ids IDS --tokens N --blocks FIRST:END
    --flags eh --manifest MANIFEST [--manifest MANIFEST ...]
    --cuda-visible 0,1,2,3,4,5
    [--budgets 1,512] [--kv 0,1] [--modes token,loop]
    [--controls all|none|layer,logits,dn,conv,ple,selection]
    [--holder-pid PID ...] [--lock-fd 9]
    [--weight-manifest FILE ...] [--build-log FILE]
    [--min-free-mib 1024] [--min-host-mib 1024]
    [--wait-seconds 300] [--timeout-seconds 600]
    [--plan]

  perl kern/run_qsa_model_matrix.pl --self-test

Flags contain e/h, or '-' for neither. Blocks use an exclusive end.
The first N IDs are used; extra IDs are permitted and ignored.
Inherited ATTN_MAP, ATTN_DIE, SPLIT_SHARES, MODEL_PLE and MODEL_PLE_OFF are preserved.
QSA_MODEL_MANIFEST and QSA_MODEL_HOLDER_PID supply defaults for one holder.

--plan prints the requested cases without CUDA or nvidia-smi access.
An inherited --lock-fd must refer to the fixed lock file and admit LOCK_EX.
The runner never starts, stops or reconfigures weight holders or an engine.
Minimum-memory options are admission thresholds, not allocation guarantees.
Successful results mean only that every selected case passed.
USAGE
}

sub slurp {
    my ($path) = @_;
    open my $fh, '<', $path or die "open $path: $!\n";
    local $/;
    my $text = <$fh>;
    close $fh or die "close $path: $!\n";
    return defined $text ? $text : '';
}

sub write_text {
    my ($path, $text) = @_;
    open my $fh, '>', $path or die "write $path: $!\n";
    print {$fh} $text or die "write $path: $!\n";
    close $fh or die "close $path: $!\n";
}

sub identity {
    my ($path) = @_;
    open my $fh, '<', $path or die "hash $path: $!\n";
    binmode $fh;
    my $sha = Digest::SHA->new(256)->addfile($fh)->hexdigest;
    close $fh;
    my @st = stat $path;
    return { path => File::Spec->rel2abs($path), bytes => $st[7],
             mtime => $st[9], sha256 => $sha };
}

sub numbers {
    my ($text, $low, $high, $label) = @_;
    my @v = split /,/, $text, -1;
    die "invalid $label\n" unless @v;
    my %seen;
    for (@v) {
        die "invalid $label: $_\n"
            unless /\A[0-9]+\z/ && $_ >= $low && $_ <= $high;
        die "duplicate $label: $_\n" if $seen{0 + $_}++;
        $_ = 0 + $_;
    }
    return \@v;
}

sub make_cases {
    my ($o) = @_;
    my @injections = ('layer');
    push @injections, 'logits' if $o->{flags} =~ /h/;
    my $has_dn = grep { $_ % 4 != 3 } $o->{first} .. $o->{end} - 1;
    push @injections, 'dn', 'conv' if $has_dn;
    push @injections, 'ple' if $o->{first} <= 1 && $o->{end} > 1;
    my @cases;
    for my $budget (@{$o->{budget_list}}) {
        my @controls = @injections;
        push @controls, 'selection' if $o->{tokens} >= 4 * ($budget + 1);
        my %reachable = map { $_ => 1 } @controls;
        if ($o->{controls} eq 'none') {
            @controls = ();
        } elsif ($o->{controls} ne 'all') {
            @controls = split /,/, $o->{controls}, -1;
            die "unreachable control $_ at budget $budget\n"
                for grep { !$reachable{$_} } @controls;
        }
        for my $kv (@{$o->{kv_list}}) {
            for my $mode (@{$o->{mode_list}}) {
                for my $injection ('', @controls) {
                    my $flags = $o->{flags};
                    $flags = '' if $flags eq '-';
                    $flags .= 'l' if $mode eq 'loop';
                    $flags = '-' unless length $flags;
                    my %env = (
                        %{$o->{base_env}},
                        MODEL_ATTN => 'qsa',
                        MODEL_IDS => $o->{ids},
                        FLASH_NBLK => "$budget",
                        FLASH_KV_HOST => "$kv",
                        CUDA_VISIBLE_DEVICES => $o->{visible},
                    );
                    $env{MODEL_INJECT} = $injection if length $injection;
                    push @cases, {
                        name => join('-', "b$budget", "kv$kv", $mode,
                                     $injection || 'positive'),
                        budget => $budget, kv_host => $kv, mode => $mode,
                        attention_layers => scalar(grep { $_ % 4 == 3 }
                            $o->{first} .. $o->{end} - 1),
                        dies => scalar(@{$o->{manifests}}),
                        injection => $injection,
                        expected_exit => length($injection) ? 3 : 0,
                        argv => [
                            "$o->{repo}/kern/model_test", $o->{cache},
                            "$o->{first}", "$o->{end}", "$o->{tokens}",
                            $flags, @{$o->{manifests}},
                        ],
                        environment => \%env,
                        cwd => "$o->{repo}/kern",
                    };
                }
            }
        }
    }
    return \@cases;
}

sub check_output {
    my ($case, $result, $output, $tokens) = @_;
    return (0, 'child timed out') if $result->{timed_out};
    return (0, 'child terminated by signal') if $result->{signal};
    return (0, "exit $result->{exit_code}; expected $case->{expected_exit}")
        if $result->{exit_code} != $case->{expected_exit};
    return (0, 'production QSA mode banner is absent')
        unless $output =~ /^attention mode: qsa \(production INT8 QSA;/m;
    return (0, 'effective budget or KV placement differs from the case')
        unless $output =~ /^FLASH_NBLK=\Q$case->{budget}\E FLASH_KV_HOST=\Q$case->{kv_host}\E;/m;
    my @assertions = $output =~ /^\s+die \d+ asserted QSA opcodes and FLASH_NBLK=\Q$case->{budget}\E for (\d+) layer\(s\)$/mg;
    my $layers = 0;
    $layers += $_ for @assertions;
    return (0, 'QSA opcode assertions do not cover the requested attention layers/dies')
        unless @assertions == $case->{dies} && $layers == $case->{attention_layers};
    if (length $case->{injection}) {
        my $kind = quotemeta($case->{injection});
        return (0, 'injection was not explicitly reached')
            unless $output =~ /MODEL_INJECT=$kind[^\n]*\breached\b/;
        return (0, 'expected comparison failure is absent')
            unless $output =~ /\bMISMATCH\b/ && $output =~ /^FAIL\b/m;
        return (1, 'injection reached and mismatch rejected');
    }
    return (0, 'positive run contains a mismatch')
        if $output =~ /\bMISMATCH\b|^FAIL\b/m;
    if ($case->{mode} eq 'token') {
        return (0, 'complete per-token comparison summary is absent')
            unless $output =~
                /^OK:\s+\Q$tokens\E of \Q$tokens\E tokens exact\b/m;
    } else {
        return (0, 'loop comparison summary is absent')
            unless $output =~ /^OK:\s+one launch\b/m;
    }
    return (1, 'requested positive comparison completed');
}

# The only process this helper may signal is the child it just forked.
sub child {
    my ($argv, $cwd, $env, $prefix, $limit) = @_;
    my $started = time;
    my $pid = fork();
    die "fork: $!\n" unless defined $pid;
    if (!$pid) {
        POSIX::close($inherited_lock_fd) if defined $inherited_lock_fd;
        chdir $cwd or _exit(126);
        open STDIN, '<', '/dev/null' or _exit(126);
        open STDOUT, '>', "$prefix.stdout" or _exit(126);
        open STDERR, '>', "$prefix.stderr" or _exit(126);
        if (defined $env) {
            delete @ENV{grep {
                /^(?:MODEL_|FLASH_)/ ||
                /\A(?:ATTN_MAP|ATTN_DIE|SPLIT_SHARES|CUDA_VISIBLE_DEVICES)\z/
            } keys %ENV};
            @ENV{keys %$env} = values %$env;
        }
        exec {$argv->[0]} @$argv or do {
            print STDERR "exec $argv->[0]: $!\n";
            _exit(127);
        };
    }
    my ($status, $timed_out) = (0, 0);
    while (1) {
        my $got = waitpid($pid, WNOHANG);
        if ($got == $pid) { $status = $?; last; }
        die "waitpid: $!\n" if $got < 0;
        if ($interrupted || time - $started >= $limit) {
            $timed_out = $interrupted ? 0 : 1;
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
        argv => $argv, cwd => $cwd,
        stdout => "$prefix.stdout", stderr => "$prefix.stderr",
        exit_code => $status >> 8, signal => $status & 127,
        timed_out => $timed_out ? JSON::PP::true : JSON::PP::false,
        elapsed_seconds => time - $started,
    };
}

sub checked_capture {
    my ($o, $argv, $prefix) = @_;
    my $r = child($argv, $o->{repo}, undef, $prefix, 20);
    die "inventory command failed; see $prefix.stderr\n"
        if $r->{timed_out} || $r->{signal} || $r->{exit_code};
    return slurp("$prefix.stdout");
}

sub holder_allowed {
    my ($pid, $o) = @_;
    return 0 unless grep { $_ == $pid } @{$o->{holder_pids}};
    my $exe = readlink("/proc/$pid/exe");
    return 0 unless defined $exe && basename($exe) eq 'hold';
    my @args = split /\0/, slurp("/proc/$pid/cmdline");
    my %arg = map { $_ => 1 } @args;
    return scalar grep { $arg{$_} } @{$o->{manifests}};
}

sub preflight {
    my ($o, $prefix) = @_;
    checked_capture($o, ['ps', '-eo', 'pid=,ppid=,etimes=,stat=,comm='], "$prefix.ps");
    my $gpu = checked_capture($o, [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], "$prefix.gpu");
    my $apps = checked_capture($o, [
        'nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ], "$prefix.apps");
    my %wanted = map { $_ => 1 } @{$o->{visible_list}};
    my (%uuid, %found);
    for my $line (split /\n/, $gpu) {
        my @v = map { s/^\s+|\s+$//gr } split /,/, $line;
        next unless @v == 6 && $wanted{$v[0]};
        $found{$v[0]} = 1;
        $uuid{$v[1]} = 1;
        return "GPU $v[0] memory/utilization unavailable"
            unless $v[4] =~ /^\d+$/ && $v[5] =~ /^\d+$/;
        return "GPU $v[0] has $v[4] MiB free; need $o->{min_free_mib}"
            if $v[4] < $o->{min_free_mib};
        return "GPU $v[0] utilization is $v[5]%" if $v[5] != 0;
    }
    for my $index (keys %wanted) {
        return "GPU index $index was not reported" unless $found{$index};
    }
    for my $line (split /\n/, $apps) {
        my @v = map { s/^\s+|\s+$//gr } split /,/, $line, 4;
        next unless @v == 4 && $uuid{$v[0]};
        return "unrecognized GPU process row: $line" unless $v[1] =~ /^\d+$/;
        return "GPU process $v[1] is not an allowed holder"
            unless holder_allowed(0 + $v[1], $o);
    }
    my $mem = slurp('/proc/meminfo');
    my ($available) = $mem =~ /^MemAvailable:\s+(\d+)\s+kB$/m;
    return 'host MemAvailable unavailable' unless defined $available;
    return "host memory below $o->{min_host_mib} MiB"
        if $available < 1024 * $o->{min_host_mib};
    return '';
}

sub reserve {
    my ($o, $name) = @_;
    my $deadline = time + $o->{wait_seconds};
    my $attempt = 0;
    while (1) {
        die "interrupted\n" if $interrupted;
        my $fh;
        if (defined $o->{lock_fd}) {
            open $fh, '<&', $o->{lock_fd}
                or die "duplicate inherited lock fd: $!\n";
            my @a = stat $fh;
            my @b = stat $LOCK;
            die "inherited fd does not name $LOCK\n"
                unless @a && @b && $a[0] == $b[0] && $a[1] == $b[1];
        } else {
            open $fh, '>>', $LOCK or die "open $LOCK: $!\n";
        }
        my $flags = fcntl($fh, F_GETFD, 0);
        die "read reservation descriptor flags: $!\n" unless defined $flags;
        fcntl($fh, F_SETFD, $flags | FD_CLOEXEC)
            or die "set reservation descriptor flags: $!\n";
        my $reason = 'shared validation lock is held';
        if (flock($fh, LOCK_EX | LOCK_NB)) {
            $reason = preflight($o, "$o->{out}/$name.preflight." . $attempt++);
            return $fh unless length $reason;
        }
        close $fh;
        # Closing a duplicated inherited descriptor does not unlock its owner.
        die "BLOCKED: $reason\n" if time >= $deadline;
        print STDERR "waiting: $reason\n";
        sleep 1;
    }
}

sub self_test {
    my $dir = tempdir('qsa-model-runner-test-XXXXXX', TMPDIR => 1, CLEANUP => 1);
    my %scope = (budget => 1, kv_host => 1, attention_layers => 2, dies => 1);
    my $positive = { %scope, mode => 'token', injection => '', expected_exit => 0 };
    my $negative = { %scope, mode => 'token', injection => 'layer', expected_exit => 3 };
    my $banner = 'print "attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=1 FLASH_KV_HOST=1; cache positions=128 compact positions=2080\n  die 0 asserted QSA opcodes and FLASH_NBLK=1 for 2 layer(s)\n";';
    my @tests = (
        [$positive, 'print "OK: 9 of 9 tokens exact\n"; exit 0;', 1],
        [$positive, 'print "OK: 8 of 9 tokens exact\n"; exit 0;', 0],
        [$positive, 'print "OK: 9 of 9 tokens run\n"; exit 0;', 0],
        [$positive, 'print "OK: 9 of 9 tokens exact\n"; exit 3;', 0],
        [$negative, 'print "MODEL_INJECT=layer reached\nMISMATCH\nFAIL\n"; exit 3;', 1],
        [$negative, 'print "MISMATCH\nFAIL\n"; exit 3;', 0],
        [$negative, 'print "MODEL_INJECT=layer reached\nMISMATCH\nFAIL\n"; exit 0;', 0],
        [$negative, 'print "MODEL_INJECT=selection reached\nMISMATCH\nFAIL\n"; exit 3;', 0],
    );
    my $failed = 0;
    for my $i (0 .. $#tests) {
        my ($case, $code, $expected) = @{$tests[$i]};
        my $r = child([$^X, '-e', $banner . $code], $dir, undef, "$dir/case$i", 5);
        my ($passed) = check_output(
            $case, $r, slurp($r->{stdout}) . slurp($r->{stderr}), 9);
        if (!!$passed != !!$expected) {
            print STDERR "self-test $i failed\n";
            $failed++;
        }
    }
    for my $stale (
        "attention mode: dense\nOK: 9 of 9 tokens exact\n",
        "attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=512 FLASH_KV_HOST=1;\n  die 0 asserted QSA opcodes and FLASH_NBLK=512 for 2 layer(s)\nOK: 9 of 9 tokens exact\n",
        "attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=1 FLASH_KV_HOST=1;\n  die 0 asserted QSA opcodes and FLASH_NBLK=1 for 1 layer(s)\nOK: 9 of 9 tokens exact\n"
    ) {
        my ($passed) = check_output($positive,
            {exit_code => 0, signal => 0, timed_out => 0}, $stale, 9);
        $failed++ if $passed;
    }
    my $o = {
        first => 0, end => 8, flags => 'eh', tokens => 9, controls => 'all',
        budget_list => [1, 512], kv_list => [0, 1],
        mode_list => ['token', 'loop'], base_env => {},
        ids => 'ids', visible => '7', repo => '/repo',
        cache => '/cache', manifests => ['/manifest'],
    };
    my $cases = make_cases($o);
    my @selection = grep { $_->{injection} eq 'selection' } @$cases;
    $failed++ unless @selection == 4 &&
        !grep { $_->{budget} != 1 } @selection;
    $o->{controls} = 'none';
    $failed++ unless @{make_cases($o)} == 8;
    $o->{controls} = 'selection';
    my $reachable = eval { make_cases($o); 1 };
    $failed++ if $reachable;
    print $failed ? "FAIL: $failed CPU runner checks\n"
                  : "OK: 8 child-result checks, 3 stale/scope checks and selection/control planning\n";
    return $failed ? 1 : 0;
}

sub main {
    my @args = @_;
    my %o = (
        repo => File::Spec->rel2abs(dirname(__FILE__) . '/..'),
        tokens => 9, blocks => '0:48', flags => 'eh', controls => 'all',
        budgets => '1,512', kv => '0,1', modes => 'token,loop',
        visible => $ENV{CUDA_VISIBLE_DEVICES} // '',
        min_free_mib => 1024, min_host_mib => 1024,
        wait_seconds => 300, timeout_seconds => 600,
    );
    my (@manifests, @holders, @weight_manifests);
    GetOptionsFromArray(\@args,
        'repo=s' => \$o{repo}, 'out=s' => \$o{out},
        'cache=s' => \$o{cache}, 'ids=s' => \$o{ids},
        'tokens=i' => \$o{tokens}, 'blocks=s' => \$o{blocks},
        'flags=s' => \$o{flags}, 'budgets=s' => \$o{budgets},
        'kv=s' => \$o{kv}, 'modes=s' => \$o{modes},
        'controls=s' => \$o{controls},
        'manifest=s@' => \@manifests, 'holder-pid=i@' => \@holders,
        'weight-manifest=s@' => \@weight_manifests,
        'build-log=s' => \$o{build_log},
        'cuda-visible=s' => \$o{visible}, 'lock-fd=i' => \$o{lock_fd},
        'min-free-mib=i' => \$o{min_free_mib},
        'min-host-mib=i' => \$o{min_host_mib},
        'wait-seconds=i' => \$o{wait_seconds},
        'timeout-seconds=i' => \$o{timeout_seconds},
        'plan' => \$o{plan}, 'self-test' => \$o{self_test},
        'help' => \$o{help},
    ) or die usage();
    die "unexpected arguments: @args\n" if @args;
    if ($o{help}) { print usage(); return 0; }
    return self_test() if $o{self_test};
    @manifests = ($ENV{QSA_MODEL_MANIFEST})
        if !@manifests && defined $ENV{QSA_MODEL_MANIFEST};
    @holders = ($ENV{QSA_MODEL_HOLDER_PID})
        if !@holders && defined $ENV{QSA_MODEL_HOLDER_PID};
    die usage() unless defined $o{cache} && defined $o{ids} && @manifests;
    die "invalid block range\n"
        unless $o{blocks} =~ /\A(\d+):(\d+)\z/ &&
               $1 < $2 && $2 <= 48;
    ($o{first}, $o{end}) = ($1 + 0, $2 + 0);
    die "range contains no attention block\n"
        unless grep { $_ % 4 == 3 } $o{first} .. $o{end} - 1;
    die "invalid flags\n" unless $o{flags} =~ /\A(?:-|e|h|eh|he)\z/;
    die "invalid token count\n" unless $o{tokens} >= 1 && $o{tokens} <= 262144;
    die "invalid wait/timeout/memory threshold\n"
        unless $o{wait_seconds} >= 0 && $o{wait_seconds} <= 3600 &&
               $o{timeout_seconds} >= 1 && $o{timeout_seconds} <= 86400 &&
               $o{min_free_mib} >= 1 && $o{min_host_mib} >= 1;
    die "invalid holder PID\n" if grep { !/^\d+$/ || $_ < 1 } @holders;
    die "invalid inherited lock descriptor\n"
        if defined($o{lock_fd}) && $o{lock_fd} < 3;
    $inherited_lock_fd = $o{lock_fd};
    if ($o{controls} ne 'all' && $o{controls} ne 'none') {
        my %seen;
        die "invalid or duplicate control\n" if grep {
            !/\A(?:layer|logits|dn|conv|ple|selection)\z/ || $seen{$_}++
        } split /,/, $o{controls}, -1;
    }
    $o{budget_list} = numbers($o{budgets}, 1, 4096, 'budget');
    $o{kv_list} = numbers($o{kv}, 0, 1, 'KV placement');
    $o{visible_list} = numbers($o{visible}, 0, 255, 'numeric CUDA visibility');
    $o{visible} = join(',', @{$o{visible_list}});
    my @modes = split /,/, $o{modes}, -1;
    die "invalid launch mode\n"
        unless @modes && !grep { $_ ne 'token' && $_ ne 'loop' } @modes;
    die "duplicate launch mode\n" if @modes > 2 ||
        (@modes == 2 && $modes[0] eq $modes[1]);
    die "manifest/device counts differ or exceed six dies\n"
        unless @manifests == @{$o{visible_list}} && @manifests <= 6;
    $o{mode_list} = \@modes;
    $o{manifests} = [map { File::Spec->rel2abs($_) } @manifests];
    $o{holder_pids} = \@holders;
    $o{repo} = File::Spec->rel2abs($o{repo});
    $o{cache} = File::Spec->rel2abs($o{cache});
    $o{ids} = File::Spec->rel2abs($o{ids});
    my %base;
    for my $key (qw(ATTN_MAP ATTN_DIE SPLIT_SHARES MODEL_PLE MODEL_PLE_OFF
                    CUDA_DEVICE_ORDER LD_LIBRARY_PATH)) {
        $base{$key} = $ENV{$key} if defined $ENV{$key};
    }
    $o{base_env} = \%base;
    my @gaps = (
        'This runner does not exercise request reset or snapshot/resume.',
        'Huge-cache direct-position component validation is a separate matrix.',
        'Loop mode retains final-token layer buffers; earlier predictions have weaker coverage.',
        'Manifest fingerprints alone do not establish checkpoint conversion provenance.',
    );
    push @gaps, 'The selected block/embedding/head scope is not a complete model.'
        unless $o{first} == 0 && $o{end} == 48 &&
               $o{flags} =~ /e/ && $o{flags} =~ /h/;
    push @gaps, 'Production-budget exclusion is not reached by this token count.'
        unless $o{tokens} >= 2052 &&
               grep { $_ == 512 } @{$o{budget_list}};
    my $report = {
        schema => 'flash-sass.qsa-model-matrix.v1',
        status => 'PLAN_ONLY', acceptance_complete => JSON::PP::false,
        scope => {
            blocks => [$o{first}, $o{end}], tokens => $o{tokens},
            flags => $o{flags}, cuda_visible_devices => $o{visible},
            minimum_gpu_free_mib => $o{min_free_mib},
            minimum_host_available_mib => $o{min_host_mib},
        },
        remaining_scope => \@gaps,
        plan => make_cases(\%o), cases => [],
    };
    if ($o{plan}) { print $JSON->encode($report); return 0; }
    die "--out NEW_DIRECTORY is required\n" unless defined $o{out};
    $o{out} = File::Spec->rel2abs($o{out});
    mkdir $o{out} or die "create fresh output directory $o{out}: $!\n";
    my $result_path = "$o{out}/results.json";
    local $SIG{INT} = sub { $interrupted = 1 };
    local $SIG{TERM} = sub { $interrupted = 1 };
    my $ok = eval {
        my @ids = grep { length } split /\s+/, slurp($o{ids});
        die "need at least $o{tokens} token IDs\n" if @ids < $o{tokens};
        @ids = @ids[0 .. $o{tokens} - 1];
        die "invalid token ID\n"
            if grep { !/\A\d+\z/ || $_ >= 248320 } @ids;
        $report->{input_ids} = identity($o{ids});
        $o{ids} = "$o{out}/ids.txt";
        write_text($o{ids}, join("\n", @ids) . "\n");
        $report->{effective_ids} = identity($o{ids});
        $report->{plan} = make_cases(\%o);
        my @files = (
            "$o{repo}/kern/model_test", "$o{repo}/kern/die.cubin",
            @{$o{manifests}}, @weight_manifests,
        );
        push @files, "$o{repo}/kern/libengine.so" if -f "$o{repo}/kern/libengine.so";
        push @files, $o{build_log} if defined $o{build_log};
        push @files, abs_path(__FILE__) if defined abs_path(__FILE__);
        $report->{file_identities} = [map { identity($_) } @files];
        $report->{cache_directory} = $o{cache};
        $report->{git_head} = checked_capture(\%o,
            ['git', 'rev-parse', 'HEAD'], "$o{out}/source-head");
        $report->{git_status} = checked_capture(\%o,
            ['git', 'status', '--porcelain'], "$o{out}/source-status");
        checked_capture(\%o, ['git', 'diff', '--binary', 'HEAD'],
                        "$o{out}/source-diff");
        $report->{source_diff} = identity("$o{out}/source-diff.stdout");
        $report->{source_identity_note} =
            'Repository snapshot plus exact binary hashes; use the build log to link them. The model_test executable does not load libengine.so.';
        $report->{status} = 'RUNNING';
        write_text($result_path, $JSON->encode($report));
        for my $case (@{$report->{plan}}) {
            print STDERR "case $case->{name}\n";
            my $lock = reserve(\%o, $case->{name});
            my $r = child($case->{argv}, $case->{cwd}, $case->{environment},
                          "$o{out}/$case->{name}", $o{timeout_seconds});
            my $text = slurp($r->{stdout}) . slurp($r->{stderr});
            my ($passed, $reason) = check_output($case, $r, $text, $o{tokens});
            $r->{name} = $case->{name};
            $r->{environment} = $case->{environment};
            $r->{expected_exit} = $case->{expected_exit};
            $r->{passed} = $passed ? JSON::PP::true : JSON::PP::false;
            $r->{reason} = $reason;
            $r->{stdout_identity} = identity($r->{stdout});
            $r->{stderr_identity} = identity($r->{stderr});
            push @{$report->{cases}}, $r;
            close $lock;
            write_text($result_path, $JSON->encode($report));
            die "case $case->{name}: $reason\n" unless $passed;
            die "interrupted\n" if $interrupted;
        }
        $report->{status} = 'SELECTED_CASES_PASSED';
        1;
    };
    if (!$ok) {
        $report->{error} = "$@";
        $report->{status} = $@ =~ /^BLOCKED:/ ? 'BLOCKED' : 'FAILED';
    }
    $report->{finished_epoch} = time;
    write_text($result_path, $JSON->encode($report));
    print "$report->{status}: $result_path\n";
    print STDERR $report->{error} if $report->{error};
    return $ok ? 0 : 1;
}

exit main(@ARGV);
