#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use JSON::PP;
use IO::Socket::UNIX;
use IO::Select;
use IO::Handle;
use Socket qw(SOCK_STREAM);
use Math::BigInt;
use POSIX qw(isfinite);
use Cwd qw(abs_path);
use File::Path qw(make_path);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
use FindBin;
use lib "$FindBin::Bin/../validation/issue15";
use EngineGuard qw(borrow_lock verify_peer);

my ($socket_path, $out, $source_sha, $resident_pid, $resident_start, $lease_fd, $lease);
GetOptions('socket=s' => \$socket_path, 'out=s' => \$out,
           'source-sha=s' => \$source_sha, 'resident-pid=i' => \$resident_pid,
           'resident-start=s' => \$resident_start, 'lease-fd=i' => \$lease_fd)
    or die "Invalid options\n";
defined($_) && length($_) or die
    "Usage: $0 --socket OWNED_SOCKET --out NEW_DIRECTORY --source-sha COMMIT\n"
    for ($socket_path, $out, $source_sha);
my $resolved = abs_path($socket_path) // $socket_path;
if (defined($resident_pid) || defined($resident_start) || defined($lease_fd)) {
    die "Resident mode requires PID, start tick, and inherited lease FD\n"
        unless defined($resident_pid) && $resident_pid > 1 && defined($resident_start)
            && $resident_start =~ /\A[0-9]+\z/ && defined($lease_fd);
    my $error; ($lease, $error) = borrow_lock($lease_fd, '/tmp/flash-sass-engine-validation.lock');
    die "$error\n" unless $lease;
}
die "Use an owned shadow core\n"
    if !$lease && $resolved eq (abs_path('/tmp/flash.sock') // '/tmp/flash.sock');
die "Evidence directory already exists: $out\n" if -e $out;
make_path($out);
$out = abs_path($out);
my $json = JSON::PP->new->utf8->canonical->allow_nonref;
my $pretty = JSON::PP->new->utf8->canonical->pretty->allow_nonref;
open my $log, '>', "$out/protocol.jsonl" or die "$out/protocol.jsonl: $!";
$log->autoflush(1);
STDOUT->autoflush(1);
$SIG{PIPE} = 'IGNORE';
my $state = {status => 'RUNNING', source_sha => $source_sha,
             socket => $resolved, cases => []};
$state->{resident} = {pid => $resident_pid, start_tick => $resident_start,
                     shared_lock => '/tmp/flash-sass-engine-validation.lock'} if $lease;
my ($sock, $select);
my $buffer = '';
my $last_receive_ns;

sub check { die "$_[1]\n" unless $_[0] }
sub persist {
    open my $f, '>', "$out/results.json.tmp" or die "results: $!";
    print {$f} $pretty->encode($state) or die "write results: $!";
    close $f or die "close results: $!";
    rename "$out/results.json.tmp", "$out/results.json" or die "rename results: $!";
}
sub event {
    my $stamp = clock_gettime(CLOCK_MONOTONIC);
    my $ns = sprintf('%.0f', $stamp * 1e9);
    print {$log} $json->encode({monotonic_s => $stamp, monotonic_ns => $ns,
                                direction => $_[0], line => $_[1]}), "\n"
        or die "write protocol: $!";
    return $ns;
}
sub send_command {
    event('send', $_[0]);
    print {$sock} "$_[0]\n" or die "send: $!";
}
sub read_line {
    while (index($buffer, "\n") < 0) {
        $select->can_read(180) or die "Core response timed out\n";
        my $n = sysread($sock, my $chunk, 65536);
        defined($n) or die "read core: $!";
        $n > 0 or die "Core closed the connection\n";
        $buffer .= $chunk;
    }
    my $line = substr($buffer, 0, index($buffer, "\n") + 1, '');
    $line =~ s/\r?\n\z//;
    $last_receive_ns = event('receive', $line);
    return $line;
}
sub ask { send_command($_[0]); return read_line() }
sub control {
    my ($command) = @_;
    my ($verb) = split / /, $command;
    my $answer = ask($command);
    check(index($answer, "ok: $verb") == 0, "$command: $answer");
}
sub near {
    my ($actual, $expected) = @_;
    return check(!defined($actual), 'Expected a null timing') unless defined $expected;
    check(defined($actual) && isfinite($actual), 'Non-finite or absent timing');
    my $scale = abs($actual) > abs($expected) ? abs($actual) : abs($expected);
    my $limit = 2e-7 * $scale;
    $limit = 1e-6 if $limit < 1e-6;
    check(abs($actual - $expected) <= $limit, "Timing mismatch: $actual versus $expected");
}
sub count_check {
    my ($t, $p, $r, $g, $stopped) = @_;
    check(ref($t) eq 'HASH', 'Canonical timing object missing');
    check($t->{schema_version} == 1 && $t->{clock} eq 'CLOCK_MONOTONIC', 'Host schema/clock');
    check(JSON::PP::is_bool($t->{counts_known}) && $t->{counts_known}
          && JSON::PP::is_bool($t->{counts_final}) && $t->{counts_final}, 'Count validity');
    check($t->{counter_semantics} eq 'last_observed_completed_positions', 'Counter semantics');
    check($t->{error_code} == 0 && defined($t->{aborted}) && !$t->{aborted}, 'Unexpected error/abort');
    check(defined($t->{stop_token_observed})
          && !!$t->{stop_token_observed} == !!$stopped, 'Stop observation');
    my $c = $p - 1 + $g;
    my $processed = ($c < $p ? $c : $p) - $r;
    my %expected = (prompt_tokens => $p, resumed_positions => $r, positions_completed => $c,
        positions_executed => $c-$r, prompt_positions_processed => $processed,
        generated_tokens => $g, decode_intervals => $g > 1 ? $g-1 : 0);
    for my $k (keys %expected) {
        check(defined($t->{$k}) && $t->{$k} == $expected{$k}, "Count mismatch: $k");
    }
    my ($first, $run) = @{$t}{qw(first_observed_ms run_ms)};
    check(defined($first) && isfinite($first) && $first >= 0
          && defined($run) && isfinite($run) && $run >= $first, 'Host timing boundaries');
    my $post = $run - $first;
    near($t->{post_first_ms}, $post);
    near($t->{prompt_positions_per_second}, $first > 0 ? 1000*$processed/$first : undef);
    near($t->{decode_tokens_per_second}, $g >= 2 && $post > 0 ? 1000*($g-1)/$post : undef);
    my %aliases = (prompt_n => 'prompt_tokens', cache_n => 'resumed_positions',
        predicted_n => 'generated_tokens', prompt_ms => 'first_observed_ms',
        predicted_ms => 'post_first_ms', decode_ms => 'post_first_ms',
        prompt_per_second => 'prompt_positions_per_second',
        predicted_per_second => 'decode_tokens_per_second');
    for my $k (keys %aliases) {
        check(exists($t->{$k}) && $json->encode($t->{$k}) eq
              $json->encode($t->{$aliases{$k}}), "Canonical alias mismatch: $k");
    }
}
sub bigint {
    check(defined($_[0]) && $_[0] =~ /\A[0-9]+\z/, 'Invalid integer device timestamp');
    return Math::BigInt->new($_[0]);
}
sub trace_check {
    my ($t, $p, $g, $raw_json) = @_;
    send_command('trace 0');
    my $header = read_line();
    check($header =~ /^trace_begin /, "Unexpected trace header: $header");
    my %m = $header =~ /(\w+)=([^\s]+)/g;
    check(defined($t->{request_id}) && $m{request} eq $t->{request_id},
          'Canonical/device trace request identity');
    check($m{die} eq '0' && $m{status} eq 'completed'
          && $m{ctas} eq '1' && $m{aborted} eq '0', 'Trace state');
    check($m{first} == $p-1 && $m{count} == $g
          && $m{completed} == $p-1+$g && $m{requested} == $p+3, 'Trace counts');
    my @columns = split /,/, read_line(), -1;
    my %col; @col{@columns} = (0 .. $#columns);
    check(exists($col{$_}), "Missing trace column: $_")
        for qw(position phase start_ns publication_ns);
    my ($start, $first, $last, $rows);
    $rows = 0;
    my (%seen, %token_events);
    while (1) {
        my $line = read_line();
        last if $line eq 'trace_end';
        my @v = split /,/, $line, -1;
        check(@v == @columns, 'Malformed trace CSV row');
        my ($pos, $phase) = @v[@col{qw(position phase)}];
        check($pos =~ /^\d+$/ && $phase =~ /^\d+$/ && $pos >= $p-1
              && $pos < $p-1+$g && $phase < $m{nph}, 'Trace row outside requested coverage');
        check(!exists($seen{"$pos,$phase"}), 'Duplicate trace row');
        $seen{"$pos,$phase"} = 1; $rows++;
        $start = $v[$col{start_ns}] if $pos == $p-1 && $phase == 0;
        $first = $v[$col{publication_ns}] if $pos == $p-1 && $phase == $m{nph}-1;
        $last = $v[$col{publication_ns}] if $pos == $p+$g-2 && $phase == $m{nph}-1;
        $token_events{$pos}{start_ns} = $v[$col{start_ns}] if $phase == 0;
        $token_events{$pos}{publication_ns} = $v[$col{publication_ns}] if $phase == $m{nph}-1;
    }
    check($rows == $g*$m{nph}, 'Incomplete trace coverage');
    ($start, $first, $last) = map { bigint($_) } ($start, $first, $last);
    check($start->is_pos && $first->bcmp($start) >= 0 && $last->bcmp($first) >= 0,
          'Invalid device timestamp order');
    check($t->{device_clock} eq 'SR_GLOBALTIMER'
          && $t->{device_origin} eq 'first_generated_position_start'
          && $t->{device_traced_tokens} == $g, 'Device timing metadata');
    my %ns = (device_first_start_ns => $start, device_first_publish_ns => $first,
              device_last_traced_publish_ns => $last);
    for my $k (keys %ns) {
        my ($quoted) = $raw_json =~ /"\Q$k\E"\s*:\s*"([0-9]+)"/;
        check(defined($quoted), "$k must be a decimal JSON string");
        check(bigint($quoted)->bcmp($ns{$k}) == 0, "Raw timestamp mismatch: $k");
    }
    near($t->{device_first_publish_ms}, $first->copy->bsub($start)->numify * 1e-6);
    if ($g >= 2) {
        my $span = $last->copy->bsub($first);
        check($span->is_pos, 'Non-positive device decode interval');
        near($t->{device_decode_ms}, $span->numify * 1e-6);
        near($t->{device_decode_tokens_per_second}, ($g-1)*1e9/$span->numify);
    } else {
        check(!defined($t->{device_decode_ms}) || $t->{device_decode_ms} == 0,
              'One token has no device decode interval');
        near($t->{device_decode_tokens_per_second}, undef);
    }
    $m{token_events} = [map {{position => 0+$_, %{$token_events{$_}}}}
                           sort {$a <=> $b} keys %token_events];
    my $previous_publication;
    for my $token (@{$m{token_events}}) {
        my $begin = bigint($token->{start_ns});
        my $publish = bigint($token->{publication_ns});
        check($begin->is_pos && $publish->bcmp($begin) >= 0,
              'Invalid per-token device interval');
        check(!defined($previous_publication) || $begin->bcmp($previous_publication) >= 0,
              'Next token began before the preceding publication sample');
        $previous_publication = $publish;
    }
    return \%m;
}
sub host_check {
    my ($trace, $arrivals) = @_;
    send_command('hosttrace');
    my $header = read_line();
    check($header =~ /^hosttrace_begin /, "Unexpected host trace header: $header");
    my %m = $header =~ /(\w+)=([^\s]+)/g;
    check($m{clock} eq 'CLOCK_MONOTONIC', 'Host trace clock');
    for my $key (qw(request first count completed requested status aborted ended)) {
        check(defined($m{$key}) && $m{$key} eq $trace->{$key}, "Host/device request identity: $key");
    }
    my @columns = split /,/, read_line(), -1;
    my %col; @col{@columns} = (0 .. $#columns);
    check(exists($col{$_}), "Missing host trace column: $_")
        for qw(position observed_ns callback_start_ns callback_end_ns);
    my (@rows, %seen);
    my $callback_total_ms = 0;
    while (1) {
        my $line = read_line(); last if $line eq 'hosttrace_end';
        my @v = split /,/, $line, -1;
        check(@v == @columns, 'Malformed host trace CSV row');
        my $pos = $v[$col{position}];
        check($pos =~ /^\d+$/ && $pos >= $m{first} && $pos < $m{first}+$m{count}
              && !$seen{$pos}++, 'Missing, duplicate or out-of-range host position');
        my ($observed, $begin, $end) = map {bigint($_)}
            @v[@col{qw(observed_ns callback_start_ns callback_end_ns)}];
        check($observed->is_pos && $begin->bcmp($observed) >= 0 && $end->bcmp($begin) >= 0,
              'Invalid host observation/callback order');
        my $i = $pos-$m{first};
        check(defined($arrivals->[$i]), 'Missing local socket observation');
        my $arrival = bigint($arrivals->[$i]);
        check($arrival->bcmp($observed) >= 0, 'Local socket observation predates engine readback');
        my $callback_ms = $end->copy->bsub($begin)->numify * 1e-6;
        $callback_total_ms += $callback_ms;
        push @rows, {position => 0+$pos, observed_ns => "$observed",
            callback_start_ns => "$begin", callback_end_ns => "$end",
            socket_parsed_ns => "$arrival", callback_ms => $callback_ms,
            observation_to_callback_return_ms => $end->copy->bsub($observed)->numify * 1e-6,
            observation_to_socket_parse_ms => $arrival->copy->bsub($observed)->numify * 1e-6};
    }
    check(@rows == $m{count}, 'Incomplete host event coverage');
    @rows = sort {$a->{position} <=> $b->{position}} @rows;
    return {clock => 'CLOCK_MONOTONIC', events => \@rows,
        callback_total_ms => $callback_total_ms,
        host_observation_span_ms => bigint($rows[-1]{observed_ns})->bsub($rows[0]{observed_ns})->numify*1e-6,
        scope => 'Core and Unix-socket checker on one host. Callback includes core token write/flush; socket parse includes transport, scheduling and checker parsing. Neither measures GPU-to-host visibility lag; GPU and host epochs are not subtracted.'};
}
sub request_run {
    my ($ids, $new) = @_; $new //= 3;
    send_command("chat $new " . join(' ', @$ids));
    my (@tokens, @arrivals, $timing, $raw);
    while (1) {
        my $line = read_line();
        if ($line =~ /^tok (\d+)$/) { push @tokens, 0+$1; push @arrivals, $last_receive_ns }
        elsif ($line =~ /^timings (.*)$/) {
            check(!defined($raw), 'Duplicate unsolicited timing record');
            $raw = $1; $timing = $json->decode($raw);
        } elsif ($line =~ /^(?:done |error)/) { return (\@tokens, $timing, $line, $raw, \@arrivals) }
        else { die "Unexpected response: $line\n" }
    }
}
sub checked {
    my ($name, $ids, $r, $g, $stopped) = @_;
    $r //= 0; $g //= 4; $stopped //= 0;
    $state->{current_case} = $name; persist();
    my ($tokens, $t, $terminal, $raw, $arrivals) = request_run($ids);
    my @d = split / /, $terminal;
    check($d[0] eq 'done' && $d[1] == $g && $d[2] == $stopped
          && $d[3] == @$ids && $d[6] == $r && @$tokens == $g, "Terminal/count mismatch: $terminal");
    count_check($t, scalar(@$ids), $r, $g, $stopped);
    my $trace = trace_check($t, scalar(@$ids), $g, $raw);
    my $host = host_check($trace, $arrivals);
    push @{$state->{cases}}, {case => $name, status => 'PASS', tokens => $tokens,
                             timings => $t, trace => $trace, host => $host};
    persist(); print "PASS $name\n";
    return $tokens;
}

persist();
my $passed = eval {
    $sock = IO::Socket::UNIX->new(Type => SOCK_STREAM, Peer => $resolved, Timeout => 10)
        or die "Connect $resolved: $!";
    if ($lease) {
        my ($ok, $error) = verify_peer($sock, $resident_pid, $resident_start);
        die "$error\n" unless $ok;
    }
    $sock->autoflush(1); $select = IO::Select->new($sock);
    $state->{config} = ask('config');
    check($state->{config} =~ /\brev=\Q$source_sha\E(?:\s|$)/,
          'Resident engine revision differs from requested source SHA');
    my @info = split / /, ask('info');
    check($info[0] eq 'info' && $info[1] =~ /^\d+$/, 'Invalid info response');
    my $capacity = $info[1];
    control('timings 1');
    control('stop');
    control('sample 0 0 1 1');
    my @reset = request_run([90, 91], 0);
    check($reset[2] =~ /^done /, 'Snapshot reset failed');
    my $baseline = checked('four_tokens', [10, 11]);
    control('snap 1');
    check($json->encode(checked('snapshot_create', [10, 11])) eq
          $json->encode($baseline), 'Snapshot creation changed tokens');
    check($json->encode(checked('snapshot_resume', [10, 11], 1)) eq
          $json->encode($baseline), 'Snapshot resume changed tokens');
    control("stop $baseline->[0]");
    check($json->encode(checked('first_token_stop', [10, 11], 1, 1, 1)) eq
          $json->encode([$baseline->[0]]), 'First-token stop mismatch');
    control('stop');
    $state->{current_case} = 'oversized_argument'; persist();
    my ($tokens, $failed, $terminal) = request_run([10, 11], $capacity);
    check($terminal =~ /^error/ && !@$tokens, 'Oversized request was not rejected');
    my $latest_line = ask('timings');
    check($latest_line =~ /^timings /, 'Missing last-timing response');
    my $latest = $json->decode(substr($latest_line, 8));
    check($json->encode($failed) eq $json->encode($latest), 'Inline/last error timing mismatch');
    if (defined $failed) {
        check(ref($failed) eq 'HASH' && $failed->{prompt_tokens} == 2
              && $failed->{error_code} != 0, 'Invalid error timing');
        check(JSON::PP::is_bool($failed->{counts_known}) && !$failed->{counts_known}
              && JSON::PP::is_bool($failed->{counts_final}) && !$failed->{counts_final},
              'Oversized request claimed execution counts');
        check(exists($failed->{stop_token_observed})
              && (!defined($failed->{stop_token_observed}) || !$failed->{stop_token_observed}),
              'Stale stop observation');
        for my $k (qw(resumed_positions positions_completed positions_executed
                      prompt_positions_processed generated_tokens decode_intervals cache_n
                      predicted_n prompt_positions_per_second decode_tokens_per_second
                      first_observed_ms)) {
            check(exists($failed->{$k}) && !defined($failed->{$k}), "Expected unknown/null: $k");
        }
    }
    push @{$state->{cases}}, {case => 'oversized_argument', status => 'PASS',
                             terminal => $terminal, timings => $failed};
    persist(); print "PASS oversized_argument\n";
    if ($state->{config} =~ /\btwo_position=1\b/) {
        control('snap 1');
        checked('paired_prompt', [20, 21, 22, 23]);
    } else {
        push @{$state->{cases}}, {case => 'paired_prompt', status => 'SKIP',
                                 reason => 'two_position is not enabled'};
    }
    $state->{status} = 'PASS'; $state->{current_case} = undef;
    persist(); print $lease ? "PASS real leased-resident timing checks\n" : "PASS real owned-core timing checks\n";
    1;
};
my $failure = $@;
if (!$passed) {
    $state->{status} = 'FAIL'; $state->{error} = "$failure";
    eval { persist() };
}
close $sock if $sock;
close $log;
die $failure unless $passed;
