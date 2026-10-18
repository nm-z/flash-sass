#!/usr/bin/env perl
# Real six-die disk-PLE regression. The caller must own the supplied core.
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use SocketEvidence qw(connect_unix send_text read_line close_socket monotonic begin_request end_request);
use PleIoEvidence qw(put sha absolute status_fields verify_failure verify_healthy failure_detail);
use JSON::PP;

$| = 1;
$SIG{PIPE} = 'IGNORE';
$SIG{INT} = sub { die "interrupted\n" };
$SIG{TERM} = sub { die "terminated\n" };
my ($socket, $report) = @ARGV;
defined($socket) && @ARGV <= 2 or die "usage: $0 OWNED_SOCKET [NEW_REPORT.json]\n";
defined($report) && -e $report and die "report already exists: $report\n";
my $timeout = $ENV{PLE_TEST_TIMEOUT_S} // 600;
my $failure_bound = $ENV{PLE_TEST_FAILURE_BOUND_MS} // 60000;
$timeout > 0 && $failure_bound > 0 or die "positive deadlines required\n";
my ($connection, $config, $passed);
my (@wire, @cases);
sub check { $_[0] or die "$_[1]\n"; }
sub send_command {
    my ($text, $seconds) = @_;
    begin_request($connection, $seconds // $timeout);
    push @wire, {direction => 'send', line => $text, time => monotonic()};
    send_text($connection, "$text\n");
}
sub line {
    my $text = read_line($connection, 16384);
    defined($text) or die "engine closed connection\n";
    $text =~ s/[\r\n]+\z//;
    push @wire, {direction => 'receive', line => $text, time => monotonic()};
    return $text;
}
sub command {
    my ($text, $prefix, $seconds) = @_;
    send_command($text, $seconds);
    my $reply = line();
    check(index($reply, $prefix) == 0, "bad acknowledgement: $reply");
    return $reply;
}
sub control {
    my ($mode, $position) = @_;
    $mode //= 'none'; $position //= 0;
    command("ple_io_test $mode $position 3 17", 'ok: ple_io_test ');
}
sub status {
    send_command('ple_io_test status');
    return status_fields(line());
}
sub same { JSON::PP->new->encode($_[0]) eq JSON::PP->new->encode($_[1]); }
sub ask {
    my ($label, $ids, $new, $snap) = @_;
    $new //= 4;
    command("snap $snap", 'ok: snap ') if $snap;
    my $start = monotonic();
    send_command('chat '.$new.' '.join(' ', @$ids));
    my (@tokens, $done, $error);
    my $case = {case => $label, prompt_ids => $ids, n_new => $new};
    push @cases, $case;
    while (1) {
        my $text = line();
        if ($text =~ /^tok (\d+)(?:\s|$)/) { push @tokens, 0 + $1; }
        elsif ($text =~ /^done /) { $done = $text; last; }
        elsif ($text =~ /^error:/) { $error = $text; last; }
        else { die "unexpected response: $text\n"; }
    }
    $case->{elapsed_ms} = (monotonic() - $start) * 1000;
    $case->{tokens} = [@tokens]; $case->{done} = $done; $case->{error} = $error;
    my $st = status(); $case->{status} = $st;
    return (\@tokens, $done, $error, $st);
}
sub healthy {
    my ($label, $expected, $new, $snap, $resumed) = @_;
    my ($tokens, $done, $error, $s) = ask($label, [10,11], $new, $snap);
    check($done && !$error && same($tokens, $expected), "$label output differs");
    verify_healthy($s);
    check($s->{resumed} == $resumed, "$label did not use expected resume position");
    return $s;
}
sub failure {
    my ($label, $mode, $position, $ids, $new, $prelaunch, $expected_prefix) = @_;
    control($mode, $position);
    my $before = status();
    my ($tokens, $done, $error, $after) = ask($label, $ids, $new);
    check(!$done && $error, "$label missing explicit error");
    verify_failure($before, $after, $position, $mode, $prelaunch, $failure_bound);
    check($error eq 'error: '.failure_detail($after), "$label lost error metadata: $error");
    check(same($tokens, $expected_prefix), "$label wrong emitted prefix");
    print "PASS $label: failure_ms=$after->{failure_ms}, completion_mask=$after->{sync_mask}\n";
    return ($before, $after);
}
sub main {
    $socket = absolute($socket);
    check($socket ne '/tmp/flash.sock', 'requires a separately owned test core socket');
    $connection = connect_unix($socket, $timeout);
    $config = command('config', 'config ');
    check($config =~ /\bple_disk=1\b/ && $config =~ /\blayers=48\/48\b/
        && $config =~ /\bdies=6\b/, "requires complete six-die disk PLE: $config");
    my $info = command('info', 'info ');
    check($info =~ /^info (\d+)/ && $1 >= 512, "requires at least 512 positions: $info");
    control();
    command('stop', 'ok: stop ');
    # A one-token prompt cannot reuse any existing prefix snapshot.
    my (undef, $fresh_done, $fresh_error, $fresh) = ask('initial cold reset', [1], 0);
    check($fresh_done && !$fresh_error && !$fresh->{snapshot}, 'initial reset failed');
    my ($baseline, $done, $error, $s) = ask('full-read baseline', [10,11], 4);
    check($done && !$error && @$baseline == 5, 'full baseline failed');
    verify_healthy($s);
    for my $mode (qw(partial eintr)) {
        control($mode, 0);
        my ($tokens, $d, $e, $st) = ask("$mode recovery", [10,11], 4);
        check($d && !$e && same($tokens, $baseline), "$mode changed tokens");
        verify_healthy($st);
        check($st->{fault_matches} >= 2 && $st->{partial_reads} > 0, "$mode injection never executed");
        check($st->{eintr_retries} > 0, 'EINTR never retried') if $mode eq 'eintr';
    }
    command('stop '.$baseline->[0], 'ok: stop ');
    for my $mode (qw(eof eio)) {
        control($mode, 2);
        my ($tokens, $d, $e, $st) = ask("$mode unused row after early stop", [10,11], 190);
        check($d && !$e && $d =~ /^done \d+ 1 / && same($tokens, [$baseline->[0]]),
              "$mode read an unused row after early stop");
        check(!$st->{failed} && !$st->{fault_matches} && !$st->{fault_reads}, 'unused-row fault executed');
        verify_healthy($st);
    }
    command('stop', 'ok: stop ');
    control();
    my ($longbase, $ld, $le, $ls) = ask('long baseline with snapshot', [10,11], 190, 1);
    check($ld && !$le && @$longbase == 191 && $ls->{snapshot} == 1, 'long baseline failed');
    for my $mode (qw(eof eio)) {
        failure("$mode prelaunch position 0", $mode, 0, [12,13], 4, 1, []);
        control();
        $s = healthy("$mode establish matching snapshot", $baseline, 4, 1, 0);
        check($s->{snapshot} == 1, 'matching snapshot was not established');
        my ($before, $after) = failure("$mode resumed prelaunch position 1", $mode, 1, [10,11], 4, 1, []);
        check($before->{snapshot} == 1 && $after->{resumed} == 1, 'snapshot invalidation test was vacuous');
        control();
        $s = healthy("$mode cold recomputation", $baseline, 4, 1, 0);
        check($s->{snapshot} == 1, 'new snapshot was not established');
        healthy("$mode new snapshot reuse", $baseline, 4, 0, 1);
        my @prefix = @$longbase[0..158];
        ($before, $after) = failure("$mode decode position 160", $mode, 160, [10,11], 190, 0, \@prefix);
        check($before->{snapshot} == 1 && $after->{resumed} == 1, 'decode did not start with matching snapshot');
        control();
        healthy("$mode full recovery", $longbase, 190, 1, 0);
        for my $position (160,161) {
            my @prompt = (10,11, (7) x 175);
            ($before, $after) = failure("$mode prefill position $position", $mode, $position, \@prompt, 4, 0, []);
            check($before->{snapshot} == 1 && $after->{resumed} == 1, 'prefill starting parity changed');
            control();
            healthy("$mode prefill $position cold recovery", $baseline, 4, 1, 0);
            healthy("$mode prefill $position snapshot reuse", $baseline, 4, 0, 1);
        }
    }
    control();
    close_socket($connection); $connection = undef;
    $passed = 1;
    print "PASS real six-die PLE failure, snapshot invalidation and reuse matrix\n";
}
eval { main(); };
my $error = $@;
my $cleanup_error = '';
if ($connection) {
    eval {
        command('ple_io_test none 0 3 17', 'ok: ple_io_test ', 5);
        command('stop', 'ok: stop ', 5);
    };
    $cleanup_error = $@;
    eval { close_socket($connection); };
    $cleanup_error .= $@ if $@;
}
if (defined $report) {
    eval { put($report, {
        passed => $passed ? JSON::PP::true : JSON::PP::false,
        kind => 'real six-die disk-PLE core requests',
        config => $config, driver_sha256 => sha(absolute($0)),
        request_timeout_s => 0 + $timeout, failure_bound_ms => 0 + $failure_bound,
        cases => \@cases, protocol => \@wire,
        error => $error || undef, cleanup_error => $cleanup_error || undef
    }); };
    $error .= $@ if $@;
}
if ($error || $cleanup_error) {
    print STDERR $error, $cleanup_error;
    exit 1;
}
exit($passed ? 0 : 1);
