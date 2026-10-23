#!/usr/bin/env perl
# Real JSON/SSE timing evidence through a private frontend and a guarded core.
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use HTTP::Tiny;
use IO::Socket::INET;
use IO::Socket::UNIX;
use IO::Select;
use IO::Handle;
use Socket qw(SOCK_STREAM);
use Math::BigInt;
use JSON::PP;
use POSIX qw(WNOHANG isfinite);
use Time::HiRes qw(time sleep);
use Cwd qw(abs_path);
use File::Path qw(make_path);
use Digest::SHA;
use FindBin;
use lib "$FindBin::Bin/../validation/issue15";
use EngineGuard qw(borrow_lock verify_peer);

my ($flashd, $gguf, $core, $out, $source_sha, $resident_pid, $resident_start, $lease_fd, $lease);
GetOptions('flashd=s'=>\$flashd, 'gguf=s'=>\$gguf, 'socket=s'=>\$core, 'out=s'=>\$out,
           'source-sha=s'=>\$source_sha, 'resident-pid=i'=>\$resident_pid,
           'resident-start=s'=>\$resident_start, 'lease-fd=i'=>\$lease_fd)
    or die "Invalid options\n";
defined($_) && length($_) or die
    "Usage: $0 --flashd BINARY --gguf SHARD1 --socket OWNED_SOCKET --out NEW_DIRECTORY --source-sha COMMIT\n"
    for ($flashd, $gguf, $core, $out, $source_sha);
die "Unexpected arguments\n" if @ARGV;
$flashd = abs_path($flashd) // die "Missing flashd\n";
$gguf = abs_path($gguf) // die "Missing GGUF\n";
$core = abs_path($core) // die "Missing core socket\n";
die "Invalid binary, GGUF, or socket\n" unless -x $flashd && -f $gguf && -S $core;
if (defined($resident_pid) || defined($resident_start) || defined($lease_fd)) {
    die "Resident mode requires PID, start tick, and inherited lease FD\n"
        unless defined($resident_pid) && $resident_pid > 1 && defined($resident_start)
            && $resident_start =~ /\A[0-9]+\z/ && defined($lease_fd);
    my $error; ($lease, $error) = borrow_lock($lease_fd, '/tmp/flash-sass-engine-validation.lock');
    die "$error\n" unless $lease;
}
die "Use an owned shadow core\n"
    if !$lease && $core eq (abs_path('/tmp/flash.sock') // '/tmp/flash.sock');
die "Evidence directory already exists\n" if -e $out;
make_path($out); $out = abs_path($out);
delete @ENV{qw(http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY)};
$ENV{no_proxy} = '127.0.0.1,localhost';
my $json = JSON::PP->new->utf8->canonical->allow_nonref;
my $pretty = JSON::PP->new->utf8->canonical->pretty;
sub sha {
    open my $f, '<:raw', $_[0] or die "Hash $_[0]: $!";
    my $h = Digest::SHA->new(256); $h->addfile($f); close $f; return $h->hexdigest;
}
my $state = {
    status=>'RUNNING', source_sha=>$source_sha, flashd=>$flashd, gguf=>$gguf, socket=>$core,
    driver_sha256=>sha(abs_path($0)), flashd_sha256=>sha($flashd), cases=>[],
    capture=>'HTTP::Tiny response bodies after transfer decoding, observed status and headers'
};
$state->{resident} = {pid => $resident_pid, start_tick => $resident_start,
                     shared_lock => '/tmp/flash-sass-engine-validation.lock'} if $lease;
my ($pid, $reaped);
STDOUT->autoflush(1);
$SIG{PIPE} = 'IGNORE';
$SIG{INT} = $SIG{TERM} = sub { die "Interrupted\n" };

sub check { die "$_[1]\n" unless $_[0] }
sub save {
    my ($name, $bytes) = @_;
    open my $f, '>:raw', "$out/$name" or die "$name: $!";
    print {$f} $bytes or die "write $name: $!";
    close $f or die "close $name: $!";
}
sub checkpoint {
    save('results.json.tmp', $pretty->encode($state));
    rename "$out/results.json.tmp", "$out/results.json" or die "rename results: $!";
}
sub reap {
    return 1 if !$pid || $reaped;
    my $r = waitpid($pid, WNOHANG);
    return $reaped = 1 if $r == $pid || $r == -1;
    return 0;
}

sub core_reply {
    my ($command,$end)=@_;
    my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$core) or die "Connect core: $!";
    if ($lease) {
        my ($ok, $error) = verify_peer($s, $resident_pid, $resident_start);
        die "$error\n" unless $ok;
    }
    $s->autoflush(1);print {$s} "$command\n" or die "Send core command: $!";
    my $select=IO::Select->new($s);my ($buffer,@lines)=('');
    while(1) {
        while(index($buffer,"\n")<0) {
            check($select->can_read(60),'Trace reply timed out');
            my $n=sysread($s,my $chunk,65536);check(defined($n)&&$n>0,'Short trace reply');$buffer.=$chunk;
        }
        my $line=substr($buffer,0,index($buffer,"\n")+1,'');$line=~s/\r?\n\z//;
        push @lines,$line;
        last if !$end || $line eq $end || $line=~/^error:/;
    }
    close $s;return \@lines;
}
sub difference_ms {return Math::BigInt->new($_[0])->bsub($_[1])->numify*1e-6;}
sub request_traces {
    my ($name,$t)=@_;my %events;
    for my $kind(qw(trace hosttrace)) {
        my $lines=core_reply($kind eq 'trace'?'trace 0':'hosttrace',"${kind}_end");
        save("$name.$kind.protocol",join("\n",@$lines)."\n");
        my $header=shift @$lines;check($header=~/^${kind}_begin /,"Missing $kind for HTTP request");
        my %meta=$header=~/(\w+)=([^\s]+)/g;
        check(defined($t->{request_id})&&$meta{request} eq $t->{request_id},"HTTP/$kind request identity differs");
        check($meta{count}==$t->{generated_tokens}&&$meta{first}==$t->{prompt_tokens}-1
              &&$meta{completed}==$t->{positions_completed}&&$meta{status} eq 'completed'
              &&$meta{aborted}==0,'HTTP trace coverage/status');
        check($kind eq 'trace' ? $meta{die}==0&&$meta{ctas}==1
                              : $meta{clock} eq 'CLOCK_MONOTONIC','Trace clock domain');
        my @columns=split /,/,shift @$lines;my %col;@col{@columns}=(0..$#columns);
        check(pop(@$lines) eq "${kind}_end",'Missing trace terminator');my %seen;
        for my $line(@$lines) {
            my @v=split /,/,$line,-1;check(@v==@columns,'Malformed trace row');
            my $p=$v[$col{position}];check($p>=$meta{first}&&$p<$meta{first}+$meta{count},'Unexpected output position');
            if($kind eq 'trace') {
                my $phase=$v[$col{phase}];check($phase<$meta{nph}&&!$seen{"$p,$phase"}++,'Duplicate/out-of-range phase');
                $events{$p}{device_start_ns}=$v[$col{start_ns}] if $phase==0;
                $events{$p}{device_publish_ns}=$v[$col{publication_ns}] if $phase==$meta{nph}-1;
            } else {
                check(!$seen{$p}++,'Duplicate host event');
                @{$events{$p}}{qw(observed_ns callback_start_ns callback_end_ns)}=
                    @v[@col{qw(observed_ns callback_start_ns callback_end_ns)}];
            }
        }
        check(@$lines==$meta{count}*($kind eq 'trace'?$meta{nph}:1),'Incomplete trace rows');
    }
    my @events;
    for my $p(sort {$a<=>$b} keys %events) {
        my $e=$events{$p};
        for my $key(qw(device_start_ns device_publish_ns observed_ns callback_start_ns callback_end_ns)) {
            check(defined($e->{$key})&&$e->{$key}=~/^[1-9][0-9]*$/,"Missing event: $key");
        }
        my $work=difference_ms($e->{device_publish_ns},$e->{device_start_ns});
        my $callback=difference_ms($e->{callback_end_ns},$e->{callback_start_ns});
        my $observation=difference_ms($e->{callback_start_ns},$e->{observed_ns});
        check($work>=0&&$callback>=0&&$observation>=0,'Event order within clock domain');
        push @events,{position=>0+$p,%$e,device_work_ms=>$work,callback_ms=>$callback,
                      observation_to_callback_return_ms=>$observation+$callback};
    }
    check($t->{device_first_start_ns} eq $events[0]{device_start_ns}
          &&$t->{device_first_publish_ns} eq $events[0]{device_publish_ns}
          &&$t->{device_last_traced_publish_ns} eq $events[-1]{device_publish_ns},
          'Canonical HTTP device stamps differ from request trace');
    return \@events;
}
sub write_completion {
    my ($request_id)=@_;my $deadline=time()+2;
    while(1) {
        open my $f,'<',"$out/flashd.log" or die "Read flashd log: $!";
        my @matches;
        while(my $line=<$f>) {
            next unless $line=~/^flashd_request_timing: (.*)$/;
            my $value=$json->decode($1);push @matches,$value if $value->{request_id} eq $request_id;
        }
        close $f;
        if(@matches){check(@matches==1,'Duplicate completion log');return $matches[0];}
        check(time()<$deadline,'Missing correlated completion log');sleep .02;
    }
}

checkpoint();
my $passed = eval {
    my $config=core_reply('config')->[0];save('core-config.txt',"$config\n");
    my %config=$config=~/(\w+)=([^\s]+)/g;
    check(defined($config{rev})&&$config{rev} eq $source_sha,'Core source revision mismatch');
    check($config{layers} eq '48/48'&&$config{trace_tokens}>=4,'Full model/tracing configuration required');
    $state->{config}=\%config;
    my $reserve = IO::Socket::INET->new(
        LocalAddr=>'127.0.0.1', LocalPort=>0, Listen=>1, Proto=>'tcp')
        or die "Reserve port: $!";
    my $port = $reserve->sockport; close $reserve;
    $state->{argv} = [$flashd, $gguf, $core, "127.0.0.1:$port"];
    $pid = fork(); defined($pid) or die "fork: $!";
    if (!$pid) {
        open STDOUT, '>', "$out/flashd.log" or POSIX::_exit(126);
        open STDERR, '>&', \*STDOUT or POSIX::_exit(126);
        $ENV{FLASHD_GREEDY} = '1'; $ENV{FLASHD_THINKING} = '0';
        $ENV{FLASHD_EXPECT_CORE_PID} = "$resident_pid" if $lease;
        exec {$flashd} $flashd, $gguf, $core, "127.0.0.1:$port"
            or do { print STDERR "exec flashd: $!\n"; POSIX::_exit(127) };
    }
    $state->{flashd_pid} = $pid; checkpoint();
    my $base = "http://127.0.0.1:$port";
    my $probe = HTTP::Tiny->new(timeout=>1, keep_alive=>0, max_redirect=>0);
    my $deadline = time()+60;
    while (1) {
        check(!reap(), 'flashd exited; see flashd.log');
        last if $probe->get("$base/health")->{status} == 200;
        check(time() < $deadline, 'flashd startup timed out');
        sleep 0.05;
    }
    my $http = HTTP::Tiny->new(timeout=>180, keep_alive=>0, max_redirect=>0);
    for my $stream (0, 1) {
        my $name = $stream ? 'sse' : 'json';
        $state->{current_case} = $name; checkpoint(); print "RUN $name\n";
        my $body = {messages=>[{role=>'user', content=>'Return the single word hello.'}],
            max_tokens=>4, temperature=>0, stream=>$stream ? JSON::PP::true : JSON::PP::false,
            chat_template_kwargs=>{enable_thinking=>JSON::PP::false}};
        save("$name.request.json", $pretty->encode($body));
        my $r = $http->post("$base/v1/chat/completions", {
            headers=>{'content-type'=>'application/json'}, content=>$json->encode($body)});
        save("$name.body", $r->{content});
        save("$name.response.json", $pretty->encode({
            status=>$r->{status}, reason=>$r->{reason}, headers=>$r->{headers}}));
        check($r->{status} == 200, "$name returned HTTP $r->{status}");
        my @events;
        if ($stream) {
            my @data = map { /^data:\s*(.*)$/ ? $1 : () } split /\r?\n/, $r->{content};
            check(@data && pop(@data) eq '[DONE]', 'SSE missing terminal [DONE]');
            @events = map { $json->decode($_) } @data;
        } else { @events = ($json->decode($r->{content})) }
        my (@final, $text_seen);
        for my $e (@events) {
            check(ref($e) eq 'HASH' && !exists($e->{error}), "$name contains an error event");
            for my $c (@{$e->{choices} // []}) {
                my $d = $c->{delta} // {};
                $text_seen ||= (defined($d->{content}) && length($d->{content}))
                            || (defined($d->{reasoning_content}) && length($d->{reasoning_content}));
                push @final, [$e, $c] if defined $c->{finish_reason};
            }
        }
        check(@final == 1, "$name must contain exactly one completion");
        my ($payload, $choice) = @{$final[0]};
        my $finish = $choice->{finish_reason};
        check($finish eq 'stop' || $finish eq 'length', "Unexpected finish reason: $finish");
        check(ref($payload->{timings}) eq 'HASH', 'Missing timings');
        my ($t, $h) = @{$payload->{timings}}{qw(engine http)};
        check(ref($t) eq 'HASH' && ref($h) eq 'HASH', 'Missing canonical/HTTP timings');
        save("$name.timings.json", $pretty->encode($payload->{timings}));
        check($t->{schema_version} == 1 && $t->{clock} eq 'CLOCK_MONOTONIC'
              && $h->{clock} eq 'CLOCK_MONOTONIC', 'Timing schema/clock');
        check($t->{counts_known} && $t->{counts_final} && defined($t->{error_code})
              && $t->{error_code} == 0 && defined($t->{aborted}) && !$t->{aborted},
              'Engine did not report a successful final count');
        check(JSON::PP::is_bool($h->{complete}) && $h->{complete}, 'HTTP completion is not complete');
        check(defined($t->{stop_token_observed})
              && !!$t->{stop_token_observed} == ($finish eq 'stop'), 'Stop/finish mismatch');
        my ($p, $cache, $g) = @{$t}{qw(prompt_tokens resumed_positions generated_tokens)};
        check(defined($p) && defined($cache) && defined($g) && $p >= 1
              && $cache >= 0 && $cache < $p && $g >= 1 && $g <= 4
              && ($finish eq 'stop' || $g == 4), 'Unexpected prompt/cache/generated count');
        check($h->{raw_token_events} == $g && $payload->{usage}{completion_tokens} == $g
              && $payload->{usage}{prompt_tokens} == $p, 'Received/engine/usage counts differ');
        check($t->{positions_completed} == $p-1+$g
              && $t->{positions_executed} == $p-1+$g-$cache
              && $t->{prompt_positions_processed} == $p-$cache
              && $t->{decode_intervals} == $g-1, 'Canonical position counts differ');
        my %aliases = (prompt_n=>'prompt_tokens', cache_n=>'resumed_positions',
            predicted_n=>'generated_tokens', prompt_ms=>'first_observed_ms',
            predicted_ms=>'post_first_ms', decode_ms=>'post_first_ms',
            prompt_per_second=>'prompt_positions_per_second',
            predicted_per_second=>'decode_tokens_per_second');
        for my $k (keys %aliases) {
            check(exists($t->{$k}) && $json->encode($t->{$k}) eq
                  $json->encode($t->{$aliases{$k}}), "Canonical alias mismatch: $k");
        }
        for my $k (grep { /_ms$/ } keys %$h) {
            check(!defined($h->{$k}) || (isfinite($h->{$k}) && $h->{$k} >= 0),
                  "Invalid HTTP observation: $k");
        }
        check(exists($h->{first_text_send_ms}), 'Missing first-text observation field');
        check(($stream && $text_seen) ? defined($h->{first_text_send_ms})
                                     : !defined($h->{first_text_send_ms}),
              'First-text observation does not match emitted text');
        check(!defined($h->{response_write_complete_ms}),
              'Final payload claims its own completed write');
        my $post=write_completion($h->{request_id});
        check($post->{write_attempted}&&!$post->{failed}&&!$post->{client_closed}
              &&$post->{clock} eq 'CLOCK_MONOTONIC','Final response write did not succeed');
        my $write_ms=$post->{response_write_returned_ms}-$h->{completion_record_created_ms};
        check($write_ms>=0,'Response write predates record construction');
        save("$name.post-write.json",$pretty->encode($post));
        my $traces=request_traces($name,$t);
        push @{$state->{cases}}, {case=>$name, status=>'PASS', finish_reason=>$finish,
            timings=>$payload->{timings},token_events=>$traces,
            overhead=>{record_construction_to_response_write_return_ms=>$write_ms,
                first_observation_to_text_send_ms=>$h->{first_observation_to_text_send_ms},
                scope=>'Host CLOCK_MONOTONIC intervals. First observation to text includes decoding, hidden/empty outputs, buffering and successful server socket writes. Record construction to write return includes serialization/logging and final writes. Neither is client receipt or an absolute GPU-to-host latency.'}};
        checkpoint(); print "PASS $name\n";
    }
    $state->{status} = 'PASS'; $state->{current_case} = undef; checkpoint();
    print "PASS real HTTP JSON and SSE timing checks\n";
    1;
};
my $failure = $@;
if (!$passed) {
    $state->{status} = 'FAIL'; $state->{error} = "$failure";
    eval { checkpoint() };
}
if (!reap()) {
    kill 'TERM', $pid;
    my $deadline = time()+5;
    sleep 0.05 while !reap() && time() < $deadline;
    if (!reap()) { kill 'KILL', $pid; waitpid($pid, 0); $reaped = 1 }
}
die $failure unless $passed;
