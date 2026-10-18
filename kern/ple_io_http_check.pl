#!/usr/bin/env perl
# Real TCP HTTP/SSE against a private flashd and an exclusively owned GPU core.
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use PleIoEvidence qw(sha put write_text absolute new_output status_fields verify_failure verify_healthy failure_detail);
use SocketEvidence qw(connect_unix send_text read_line close_socket monotonic begin_request);
use Getopt::Long 'GetOptions';
use JSON::PP;
use HTTP::Tiny;
use IO::Socket::INET;
use POSIX qw(WNOHANG _exit);
use Time::HiRes qw(sleep alarm);

$| = 1;
$SIG{PIPE} = 'IGNORE';
$SIG{INT} = sub { die "interrupted\n" };
$SIG{TERM} = sub { die "terminated\n" };
my ($core_path, $flashd, $gguf, $out, $base, $proc, $config);
my (@protocol, @results);
my $timeout = $ENV{PLE_TEST_TIMEOUT_S} // 600;
my $failure_bound = $ENV{PLE_TEST_FAILURE_BOUND_MS} // 60000;
my $run_status = 'FAIL';
sub check { $_[0] or die "$_[1]\n"; }
sub decode_json { JSON::PP->new->utf8->decode($_[0]); }
sub same {
    my $j = JSON::PP->new->canonical;
    return $j->encode($_[0]) eq $j->encode($_[1]);
}
sub core_request {
    my ($command, $terminal, $seconds) = @_;
    $terminal //= sub { 1 }; $seconds //= $timeout;
    my $c = connect_unix($core_path, $seconds);
    begin_request($c, $seconds);
    push @protocol, {direction => 'send', line => $command, time => monotonic()};
    send_text($c, "$command\n");
    my @lines;
    while (1) {
        my $line = read_line($c, 16384);
        defined($line) or die "engine closed during $command\n";
        $line =~ s/[\r\n]+\z//;
        push @protocol, {direction => 'receive', line => $line, time => monotonic()};
        push @lines, $line;
        if ($terminal->($line)) { close_socket($c); return \@lines; }
    }
}
sub control {
    my ($mode, $position, $seconds) = @_;
    $mode //= 'none'; $position //= 0;
    my $x = core_request("ple_io_test $mode $position 3 17", undef, $seconds)->[0];
    check(index($x, 'ok: ple_io_test ') == 0, "bad control: $x");
}
sub status { status_fields(core_request('ple_io_test status')->[0]); }
sub fresh {
    control();
    my $lines = core_request('chat 0 1', sub { $_[0] =~ /^(?:done |error)/ });
    check($lines->[-1] =~ /^done /, 'cold reset did not finish');
    my $s = status();
    check(!$s->{snapshot} && !$s->{failed}, 'cold reset retained snapshot/failure');
    return $s;
}
sub http_request {
    my ($method, $url, $options, $seconds) = @_;
    $options //= {}; $seconds //= $timeout;
    local %ENV = %ENV;
    delete @ENV{qw(http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY)};
    local $SIG{ALRM} = sub { die "whole HTTP request exceeded $seconds seconds\n" };
    alarm($seconds);
    my $reply = eval { HTTP::Tiny->new(timeout => $seconds)->request($method, $url, $options); };
    my $error = $@; alarm(0);
    die $error if $error;
    return $reply;
}
sub post {
    my ($label, $stream, $long_prompt) = @_;
    my $content = $long_prompt ? join(' ', ('x') x 180)
        : 'List the integers from 1 through 80 in order, separated by commas. Include every number.';
    my $request = {
        messages => [{role => 'user', content => $content}], max_tokens => 96,
        stream => $stream ? JSON::PP::true : JSON::PP::false, temperature => 0,
        chat_template_kwargs => {enable_thinking => JSON::PP::false}
    };
    my $start = monotonic();
    my $r = http_request('POST', "$base/v1/chat/completions", {
        headers => {'content-type' => 'application/json'},
        content => JSON::PP->new->utf8->encode($request)
    });
    my $elapsed = (monotonic() - $start) * 1000;
    my $body = $r->{content} // '';
    write_text("$out/$label.txt", $body);
    put("$out/$label.meta.json", {status => $r->{status}, headers => $r->{headers},
                                  elapsed_ms => $elapsed, request => $request});
    check($r->{status} != 599, "HTTP transport failure $label: $body");
    my ($media) = split /\s*;\s*/, lc($r->{headers}{'content-type'} // '');
    check($media eq ($stream ? 'text/event-stream' : 'application/json'), "$label content type");
    return ($r->{status}, $body, $elapsed);
}
sub sse_failure {
    my ($body, $expected_error, $need_text, $baseline_text) = @_;
    my @data = map { my $x = substr($_,5); $x =~ s/^\s+|\s+$//g; $x }
        grep { /^data:/ } split /\r\n|\n|\r/, $body;
    check(@data && pop(@data) eq '[DONE]', 'SSE lacks final transport terminator');
    my ($errors, $failed, $text) = (0, 0, '');
    for my $raw (@data) {
        check($raw ne '[DONE]', 'SSE has duplicate transport terminator');
        my $event = decode_json($raw);
        check(!exists($event->{usage}) && !exists($event->{timings}), 'failed SSE contains success metadata');
        if (exists $event->{error}) {
            $errors++; $failed = 1;
            check($event->{error}{type} eq 'engine_error'
                && $event->{error}{message} eq $expected_error, 'SSE lost exact core error');
        } else {
            check(!$failed, 'SSE data arrived after error');
            for my $choice (@{$event->{choices} // []}) {
                check(!defined($choice->{finish_reason}), 'SSE emitted successful finish_reason');
                $text .= $choice->{delta}{content} // '';
            }
        }
    }
    check($errors == 1, 'SSE must contain exactly one error');
    if ($need_text) {
        check(length($text) > 0 && index($baseline_text, $text) == 0,
              'decode SSE did not emit an identical nonempty text prefix before error');
    }
    return length($text);
}
sub poll_proc {
    return undef unless $proc;
    return $proc->{exit} if exists $proc->{exit};
    my $r = waitpid($proc->{pid}, WNOHANG);
    if ($r == $proc->{pid}) { return $proc->{exit} = $?; }
    die "waitpid private flashd: $!\n" if $r < 0;
    return undef;
}
sub stop_flashd {
    return unless $proc && !defined(poll_proc());
    kill 'TERM', $proc->{pid};
    my $end = monotonic() + 5;
    while (!defined(poll_proc()) && monotonic() < $end) { sleep .05; }
    if (!defined(poll_proc())) {
        # Only this runner's child; never the supplied core or another worker.
        kill 'KILL', $proc->{pid};
        waitpid($proc->{pid}, 0);
        $proc->{exit} = $?;
    }
}
sub start_flashd {
    my $reserve = IO::Socket::INET->new(LocalAddr => '127.0.0.1', LocalPort => 0,
                                      Proto => 'tcp', Listen => 1) or die "reserve port: $!\n";
    my $port = $reserve->sockport; close $reserve;
    $base = "http://127.0.0.1:$port";
    my @argv = ($flashd, $gguf, $core_path, "127.0.0.1:$port");
    my $pid = fork(); defined($pid) or die "fork: $!\n";
    if (!$pid) {
        open STDOUT, '>:raw', "$out/flashd.log" or _exit(126);
        open STDERR, '>&', \*STDOUT or _exit(126);
        $ENV{FLASHD_THINKING} = '0';
        exec {$argv[0]} @argv;
        _exit(127);
    }
    $proc = {pid => $pid};
    my $end = monotonic() + 60;
    while (1) {
        check(!defined(poll_proc()), 'private flashd exited; see flashd.log');
        my $r = eval { http_request('GET', "$base/health", {}, 1) };
        last if $r && $r->{status} == 200;
        check(monotonic() < $end, 'private flashd startup timeout');
        sleep .05;
    }
}
sub successful {
    my ($label, $baseline, $cold) = @_;
    my ($code, $body) = post($label, 0, 0);
    check($code == 200, "$label expected HTTP 200");
    my $payload = decode_json($body);
    check(!exists($payload->{error}) && same($payload->{choices}, $baseline->{choices}),
          "$label healthy output differs");
    my $s = status(); verify_healthy($s);
    check($s->{snapshot} > 0, "$label did not establish a snapshot");
    check($cold ? ($s->{resumed} == 0 && $payload->{timings}{cache_n} == 0)
                : ($s->{resumed} > 0 && $payload->{timings}{cache_n} == $s->{resumed}),
          "$label cache reuse mismatch");
    return $s;
}
sub matrix {
    $config = core_request('config')->[0];
    check($config =~ /\bple_disk=1\b/ && $config =~ /\blayers=48\/48\b/
        && $config =~ /\bdies=6\b/, "requires full six-die disk PLE: $config");
    my $info = core_request('info')->[0];
    check($info =~ /^info (\d+)/ && $1 >= 512, "requires 512 positions: $info");
    fresh(); start_flashd();
    my ($code, $body) = post('baseline', 0, 0);
    check($code == 200, "baseline HTTP $code");
    my $baseline = decode_json($body);
    my $prompt_n = $baseline->{usage}{prompt_tokens};
    check(defined($prompt_n) && $prompt_n > 0 && $baseline->{usage}{completion_tokens} > 20,
          'baseline ended too early for a decode failure after text');
    my $baseline_text = $baseline->{choices}[0]{message}{content} // '';
    check(length($baseline_text) > 0 && status()->{snapshot} > 0, 'baseline lacks text/snapshot');
    my $decode_position = $prompt_n + 16;
    for my $mode (qw(eof eio)) {
        for my $phase ('prelaunch', 'prefill160', 'prefill161', 'decode') {
            my $position = $phase eq 'prelaunch' ? 0 : $phase eq 'decode'
                ? $decode_position : $phase eq 'prefill160' ? 160 : 161;
            for my $stream (0,1) {
                my $label = "$mode-$phase-".($stream ? 'sse' : 'json');
                if ($phase eq 'decode') {
                    control();
                    successful("$label-establish-matching-snapshot", $baseline, 0);
                } else { fresh(); }
                control($mode, $position);
                my $before = status();
                my ($status_code, $response, $elapsed) = post($label, $stream, $phase =~ /^prefill/);
                my $after = status();
                verify_failure($before, $after, $position, $mode, $position == 0, $failure_bound);
                if ($phase eq 'decode') {
                    check($before->{snapshot} > 0 && $after->{resumed} == $before->{snapshot},
                          "$label snapshot test was vacuous");
                }
                my $detail = failure_detail($after);
                my $prefix_bytes = 0;
                if ($stream) {
                    check($status_code == 200, "$label expected SSE 200");
                    $prefix_bytes = sse_failure($response, $detail, $phase eq 'decode', $baseline_text);
                } else {
                    check($status_code == 502, "$label expected JSON 502");
                    my $payload = decode_json($response);
                    check($payload->{error}{message} eq $detail && !exists($payload->{choices})
                        && !exists($payload->{usage}) && !exists($payload->{timings}), "$label invalid JSON error");
                }
                control();
                my $cold = successful("$label-cold-recovery", $baseline, 1);
                my $warm = successful("$label-new-snapshot-reuse", $baseline, 0);
                push @results, {case => $label, status => 'PASS', http_status => $status_code,
                    elapsed_ms => $elapsed, text_prefix_bytes => $prefix_bytes,
                    before => $before, after => $after, cold => $cold, warm => $warm};
                print "PASS $label: failure_ms=$after->{failure_ms}, text_prefix_bytes=$prefix_bytes\n";
            }
        }
    }
}
GetOptions('flashd=s' => \$flashd, 'gguf=s' => \$gguf, 'core=s' => \$core_path, 'out=s' => \$out)
    or die "invalid options\n";
defined($flashd) && defined($gguf) && defined($core_path) && defined($out) && !@ARGV
    or die "usage: $0 --flashd PATH --gguf PATH --core OWNED_SOCKET --out NEW_DIRECTORY\n";
$timeout > 0 && $failure_bound > 0 or die "positive deadlines required\n";
$core_path = absolute($core_path);
check($core_path ne '/tmp/flash.sock', 'requires a separately owned test core socket');
$flashd = absolute($flashd); $gguf = absolute($gguf); $out = new_output($out);
eval { matrix(); $run_status = 'PASS'; };
my $error = $@;
my @cleanup_errors;
eval { control('none', 0, 5); };
push @cleanup_errors, $@ if $@;
eval { stop_flashd(); };
push @cleanup_errors, $@ if $@;
$run_status = 'FAIL' if @cleanup_errors;
put("$out/protocol.json", \@protocol);
put("$out/result.json", {
    status => $run_status, kind => 'real six-die PLE, real flashd and TCP HTTP/SSE',
    config => $config, driver_sha256 => sha(absolute($0)), flashd_sha256 => sha($flashd),
    core => $core_path, request_timeout_s => 0 + $timeout, failure_bound_ms => 0 + $failure_bound,
    cases => \@results, error => $error || undef, cleanup_errors => \@cleanup_errors
});
if ($error || @cleanup_errors) { print STDERR $error, @cleanup_errors; exit 1; }
print "PASS all 16 real-GPU HTTP/SSE failure and cache recovery cases\n";
exit 0;
