#!/usr/bin/env perl
use strict;
use warnings;
use Time::HiRes qw(time sleep alarm);
use IO::Socket::UNIX;
use IO::Socket::INET;
use IO::Select;
use JSON::PP qw(encode_json decode_json);
use Test::More;
use File::Temp qw(tempdir);
use POSIX qw(WNOHANG setpgid _exit);
use Socket qw(SOCK_STREAM);

my $binary = shift(@ARGV) // './flashd';
-x $binary or BAIL_OUT("flashd is not executable: $binary");
my $dir = tempdir('flashd-deadline-XXXXXX', TMPDIR => 1, CLEANUP => 1);
my $sock_path = "$dir/engine.sock";
my $mode_path = "$dir/next-mode";
my $owner_pid = $$;
my %owned;
$SIG{PIPE} = 'IGNORE';

sub slurp {
    my ($path) = @_;
    open my $f, '<', $path or return '';
    local $/;
    my $text = <$f>;
    close $f;
    return $text // '';
}
sub write_text {
    my ($path, $text) = @_;
    my $tmp = "$path.tmp.$$";
    open my $f, '>', $tmp or die "$tmp: $!";
    print {$f} $text or die "write $tmp: $!";
    close $f or die "close $tmp: $!";
    rename $tmp, $path or die "rename $tmp: $!";
}
sub save_json { write_text($_[0], encode_json($_[1])); }
sub read_json {
    my ($path) = @_;
    my $v = eval { decode_json(slurp($path)) };
    return ref($v) eq 'HASH' ? $v : undef;
}
sub wait_for {
    my ($seconds, $fn) = @_;
    my $until = time() + $seconds;
    while (time() < $until) {
        my $v = $fn->();
        return $v if $v;
        sleep 0.01;
    }
    return;
}
sub spawn_owned {
    my ($group, $fn) = @_;
    my $pid = fork();
    defined $pid or die "fork: $!";
    if (!$pid) {
        setpgid(0, 0) == 0 or die "setpgid: $!" if $group;
        my $ok = eval { $fn->(); 1 };
        warn $@ unless $ok;
        _exit($ok ? 0 : 1);
    }
    $owned{$pid} = $group;
    return $pid;
}
END {
    if ($$ == $owner_pid) {
        local $?; # Reaping a test-owned terminated server must not change our result.
        for my $pid (keys %owned) {
            kill 'TERM', $owned{$pid} ? -$pid : $pid;
        }
        my $until = time() + 0.5;
        while (%owned && time() < $until) {
            for my $pid (keys %owned) {
                my $r = waitpid($pid, WNOHANG);
                delete $owned{$pid} if $r == $pid || $r == -1;
            }
            sleep 0.01 if %owned;
        }
        for my $pid (keys %owned) {
            kill 'KILL', $owned{$pid} ? -$pid : $pid;
            waitpid($pid, 0);
        }
    }
}

# A native, tensor-free GGUF tokenizer keeps this test independent of models/GPU.
sub ggstr { return pack('Q<', length($_[0])) . $_[0]; }
sub ggkv { return ggstr($_[0]) . pack('V', $_[1]) . $_[2]; }
my @tokens = ('<|im_end|>', '</think>', 'a' .. 'z');
my @types = (3, 3, map { 1 } 1 .. 26);
my @kv = (
    ggkv('tokenizer.ggml.model', 8, ggstr('gpt2')),
    ggkv('tokenizer.ggml.pre', 8, ggstr('qwen35')),
    ggkv('tokenizer.ggml.tokens', 9,
         pack('V Q<', 8, scalar @tokens) . join('', map { ggstr($_) } @tokens)),
    ggkv('tokenizer.ggml.token_type', 9,
         pack('V Q<', 5, scalar @types) . pack('l<*', @types)),
    ggkv('tokenizer.ggml.merges', 9, pack('V Q<', 8, 0)),
    ggkv('tokenizer.ggml.eos_token_id', 4, pack('V', 0)),
    ggkv('tokenizer.ggml.add_bos_token', 7, pack('C', 0)),
    ggkv('tokenizer.chat_template', 8, ggstr('{{ messages[0].content }}')),
);
my $gguf = 'GGUF' . pack('V Q< Q<', 3, 0, scalar @kv) . join('', @kv);
$gguf .= "\0" x ((32 - length($gguf) % 32) % 32);
open my $gf, '>:raw', "$dir/tokenizer.gguf" or die $!;
print {$gf} $gguf or die $!;
close $gf or die $!;

my $reserve = IO::Socket::INET->new(
    LocalAddr => '127.0.0.1', LocalPort => 0, Listen => 1, ReuseAddr => 1,
) or die "reserve port: $!";
my $port = $reserve->sockport();
close $reserve;
write_text($mode_path, "quick\n");

sub core_handler {
    my ($c, $mode) = @_;
    $c->autoflush(1);
    local $SIG{ALRM} = sub { die "fake core handler timed out\n"; };
    alarm 7;
    my $log = "$dir/core-$mode-$$.json";
    save_json($log, { mode => $mode, accepted => 1, pid => $$ });
    if ($mode eq 'sink') {
        # Leave the engine socket unread beyond the absolute request deadline.
        sleep 2;
        my ($received, $eof) = (0, 0);
        my $sel = IO::Select->new($c);
        my $until = time() + 3;
        while (time() < $until) {
            next unless $sel->can_read(0.05);
            my $n = sysread($c, my $buf, 65536);
            if (!defined $n) {
                next if $!{EINTR};
                die "sink read: $!";
            }
            if (!$n) { $eof = 1; last; }
            $received += $n;
        }
        save_json($log, {
            mode => $mode, accepted => 1, pid => $$,
            received_bytes => $received, eof => $eof,
        });
        close $c;
        return;
    }
    while (defined(my $line = <$c>)) {
        $line =~ s/\r?\n\z//;
        if ($line eq 'info') {
            print {$c} "info 262144 0\n" or die "info write: $!";
        } elsif ($line =~ /\A(?:timings|cancel_on_eof)\s*1\z/) {
            print {$c} "ok: $line\n" or die "ack write: $!";
        } elsif ($line =~ /\A(?:stop|sample|snap)\b/) {
            print {$c} "ok\n" or die "command ack: $!";
        } elsif ($line =~ /\Achat\s/) {
            my @fields = split /\s+/, $line;
            my $np = @fields - 2;
            if ($mode eq 'stall' || $mode eq 'laggard') {
                print {$c} "tok 2\n" or die "token write: $!";
                while (defined(my $ignored = <$c>)) { }
                sleep 2.5 if $mode eq 'laggard';
                print {$c} "done 1 0 $np 1000 0.1 0 0\n";
                save_json($log, {mode => $mode, accepted => 1, pid => $$, eof => 1});
            } else {
                print {$c} "tok 2\ntok 0\ndone 2 1 $np 1.0 0.1 0 0\n"
                    or die "chat write: $!";
            }
            last;
        } else {
            die "unexpected fake core command: $line\n";
        }
    }
    close $c;
}

my $listener = IO::Socket::UNIX->new(
    Type => SOCK_STREAM, Local => $sock_path, Listen => 16,
) or die "listen $sock_path: $!";
spawn_owned(1, sub {
    $SIG{CHLD} = 'IGNORE';
    while (my $c = $listener->accept()) {
        my $mode = slurp($mode_path);
        $mode =~ s/\s+\z//;
        $mode = 'quick' unless $mode =~ /\A(?:quick|sink|stall|laggard)\z/;
        write_text($mode_path, "quick\n");
        my $pid = fork();
        defined $pid or die "handler fork: $!";
        if (!$pid) {
            close $listener;
            my $ok = eval { core_handler($c, $mode); 1 };
            warn $@ unless $ok;
            _exit($ok ? 0 : 1);
        }
        close $c;
    }
    die "fake core accept: $!";
});
close $listener;
spawn_owned(0, sub {
    $ENV{FLASHD_REQUEST_MAX_S} = 1;
    $ENV{FLASHD_THINKING} = 0;
    $ENV{FLASHD_GREEDY} = 1;
    open STDOUT, '>', "$dir/frontend.stdout" or die $!;
    open STDERR, '>', "$dir/frontend.stderr" or die $!;
    exec $binary, "$dir/tokenizer.gguf", $sock_path, "127.0.0.1:$port";
    die "exec $binary: $!";
});

sub http {
    my ($text, $stream) = @_;
    my $started = time();
    my ($raw, $error) = ('', '');
    local $SIG{ALRM} = sub { die "HTTP operation exceeded 6 seconds\n"; };
    eval {
        alarm 6;
        my $c = IO::Socket::INET->new(
            PeerAddr => '127.0.0.1', PeerPort => $port, Timeout => 1,
        ) or die "HTTP connect: $!";
        my $body = encode_json({
            model => 'deadline-test', max_tokens => 2,
            stream => $stream ? JSON::PP::true : JSON::PP::false,
            messages => [{ role => 'user', content => $text }],
        });
        my $wire = "POST /v1/chat/completions HTTP/1.1\r\n"
                 . "Host: localhost\r\nContent-Type: application/json\r\n"
                 . "Connection: close\r\nContent-Length: " . length($body)
                 . "\r\n\r\n$body";
        my $offset = 0;
        while ($offset < length($wire)) {
            my $n = syswrite($c, $wire, length($wire) - $offset, $offset);
            if (!defined $n) { next if $!{EINTR}; die "HTTP write: $!"; }
            die "HTTP write returned zero\n" unless $n;
            $offset += $n;
        }
        my $sel = IO::Select->new($c);
        while (1) {
            die "HTTP response timed out\n" if time() - $started >= 6;
            next unless $sel->can_read(0.05);
            my $n = sysread($c, my $buf, 65536);
            if (!defined $n) { next if $!{EINTR}; die "HTTP read: $!"; }
            last unless $n;
            $raw .= $buf;
        }
        close $c;
        1;
    } or $error = "$@";
    alarm 0;
    my ($status) = $raw =~ /\AHTTP\/\d\.\d\s+(\d+)/;
    my (undef, $body) = split /\r?\n\r?\n/, $raw, 2;
    return {status => 0 + ($status // 0), body => $body // '',
            elapsed => time() - $started, error => $error};
}

my $ready = wait_for(3, sub {
    my $c = IO::Socket::INET->new(
        PeerAddr => '127.0.0.1', PeerPort => $port, Timeout => 0.1,
    );
    return unless $c;
    close $c;
    return 1;
});
$ready or BAIL_OUT("frontend did not listen:\n" . slurp("$dir/frontend.stderr"));
my $quick = http('abc', 0);
is($quick->{status}, 200, 'quick request succeeds');
is($quick->{error}, '', 'quick request has no client I/O failure');
like($quick->{body}, qr/"content"\s*:\s*"a"/, 'quick response contains text');

write_text($mode_path, "sink\n");
spawn_owned(0, sub { save_json("$dir/slow-http.json", http('a' x 250000, 0)); });
my $sink_path = wait_for(2, sub {
    my @paths = glob "$dir/core-sink-*.json";
    return $paths[0];
});
$sink_path or BAIL_OUT("large request never reached the non-reading core:\n"
    . slurp("$dir/slow-http.json") . "\n" . slurp("$dir/frontend.stderr"));
sleep 0.1;
my $queued = http('abc', 0);
my $slow = wait_for(3, sub { read_json("$dir/slow-http.json") });
$slow or BAIL_OUT("large-request client never completed");
is($slow->{status}, 504, 'blocked engine write produces HTTP 504');
like($slow->{body}, qr/request_timeout/, 'blocked write reports request_timeout');
cmp_ok($slow->{elapsed}, '<', 2, 'blocked write expires within two seconds');
is($slow->{error}, '', 'blocked-write response completes normally');
is($queued->{status}, 200, 'next queued request progresses');
my $qjson = eval { decode_json($queued->{body}) } // {};
my $queue_ms = $qjson->{timings}{http}{queue_ms} // 0;
cmp_ok($queue_ms, '>', 0, 'second request actually waited in the queue');
my $sink = wait_for(4, sub {
    my $v = read_json($sink_path);
    return $v && exists($v->{received_bytes}) ? $v : undef;
});
ok($sink && $sink->{eof}, 'non-reading core eventually observes input EOF');
cmp_ok($sink ? $sink->{received_bytes} : 0, '>', 0,
       'frontend sent a nonzero portion of the protocol');
cmp_ok($sink ? $sink->{received_bytes} : 999999999, '<', 500007,
       'frontend abandoned the write before the full large chat line');
diag(sprintf('blocked write %.3f s; next request queue %.3f ms; partial engine bytes %d',
             $slow->{elapsed}, $queue_ms, $sink ? $sink->{received_bytes} : 0));

write_text($mode_path, "stall\n");
my $sse = http('abc', 1);
is($sse->{status}, 200, 'stream begins with HTTP 200');
like($sse->{body}, qr/"content"\s*:\s*"a"/, 'stream emits text before stalling');
like($sse->{body}, qr/request_timeout/, 'running stream reports request_timeout');
my $done_count = () = $sse->{body} =~ /^data:\s*\[DONE\]\s*\r?$/mg;
is($done_count, 1, 'timed-out stream emits exactly one DONE');
sub has_aborted_http {
    my ($v) = @_;
    if (ref($v) eq 'HASH') {
        return 1 if ref($v->{http}) eq 'HASH' && $v->{http}{aborted};
        for my $child (values %$v) { return 1 if has_aborted_http($child); }
    } elsif (ref($v) eq 'ARRAY') {
        for my $child (@$v) { return 1 if has_aborted_http($child); }
    }
    return 0;
}
my $aborted = 0;
for my $data ($sse->{body} =~ /^data:\s*([^\r\n]*)/mg) {
    next if $data eq '[DONE]';
    my $event = eval { decode_json($data) };
    $aborted ||= has_aborted_http($event);
}
ok($aborted, 'stream timing reports http.aborted');
cmp_ok($sse->{elapsed}, '<', 2, 'cooperative stream cancellation drains promptly');
is($sse->{error}, '', 'stream response closes without client timeout');

# Keep the first request draining long enough for a waiting request's own
# deadline to expire. It must never consume an engine connection afterward.
write_text($mode_path, "laggard\n");
spawn_owned(0, sub { save_json("$dir/draining-http.json", http('abc', 0)); });
wait_for(2, sub { my @p = glob "$dir/core-laggard-*.json"; return $p[0]; })
    or BAIL_OUT('draining request never reached its core');
my @before = glob "$dir/core-*.json";
my $expired_queue = http('abc', 0);
is($expired_queue->{status}, 504, 'queued request expires on its own deadline');
like($expired_queue->{body}, qr/request_timeout/, 'queue expiry reports request_timeout');
cmp_ok($expired_queue->{elapsed}, '<', 2, 'queue expiry does not wait for core draining');
my @after = glob "$dir/core-*.json";
is(scalar @after, scalar @before, 'expired queued request never connects to the core');
my $drained = wait_for(4, sub { read_json("$dir/draining-http.json") });
ok($drained && $drained->{status} == 504, 'noncooperative cancellation eventually returns 504');
cmp_ok($drained ? $drained->{elapsed} : 99, '<', 4.2, 'cancellation drain has a fixed bound');
is(http('abc', 0)->{status}, 200, 'engine queue recovers after cancellation drain expires');
diag(slurp("$dir/frontend.stderr")) unless Test::More->builder->is_passing;
done_testing();
