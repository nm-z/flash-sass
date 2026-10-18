#!/usr/bin/env perl
use strict;use warnings;use IO::Socket::INET;use JSON::PP;use File::Temp qw(tempdir);use FindBin;use Test::More;use POSIX ();
my$script="$FindBin::Bin/depth.sh";my$tmp=tempdir(CLEANUP=>1);
sub readfile{my($p)=@_;open my$f,'<:raw',$p or die"$p: $!";local$/;my$s=<$f>;close$f;return defined($s)?$s:'';}
sub canonical{my($generated)=@_;return{schema_version=>1,counts_known=>JSON::PP::true,counts_final=>JSON::PP::true,error_code=>0,aborted=>JSON::PP::false,generated_tokens=>$generated,decode_intervals=>$generated-1,prompt_tokens=>100};}
my @prefill_cases = (
    ['prefill-inconsistent', 2, {prompt_per_second => 999}],
    ['prefill-zero-rate', 2, {prompt_per_second => 0}],
    ['prefill-missing-rate', 2, {}],
    ['prefill-negative-rate', 2, {prompt_per_second => -1}],
    ['prefill-string-rate', 2, {prompt_per_second => '50'}],
    ['prefill-cached', 0, {cache_n => 40, prompt_per_second => 30}],
    ['prefill-cached-wrong-rate', 2, {cache_n => 40}],
    ['prefill-invalid-cache', 2, {cache_n => 101}],
    ['prefill-negative-cache', 2, {cache_n => -1}],
    ['prefill-fully-cached', 0, {cache_n => 100, prompt_per_second => 0}],
    ['prefill-fully-cached-rate', 2, {cache_n => 100}],
    # Actual durations .149 ms and .051 ms both serialize as .1 ms.
    ['prefill-rounded-low', 0, {prompt_ms => .1, prompt_per_second => 671140.94}],
    ['prefill-rounded-high', 0, {prompt_ms => .1, prompt_per_second => 1960784.31}],
    ['prefill-rounded-invalid', 2, {prompt_ms => .1, prompt_per_second => 600000}],
    # A positive .04 ms duration rounds to zero, while its rate remains known.
    ['prefill-rounded-zero', 0, {prompt_ms => 0, prompt_per_second => 2500000}],
    ['prefill-rounded-zero-invalid', 2, {prompt_ms => 0, prompt_per_second => 999}],
    ['prefill-zero-null', 0, {prompt_ms => 0, prompt_per_second => undef}],
    ['prefill-zero-legacy', 0, {prompt_ms => 0, prompt_per_second => 0}],
    ['prefill-rate-rounds-zero', 0, {prompt_n => 1, prompt_ms => 250000, prompt_per_second => 0}],
);
my @decode_cases = (
    ['decode-rounded-inconsistent', 2, {predicted_ms => .1, decode_ms => .1, predicted_per_second => 1}],
    # Actual durations .149 ms and .051 ms both serialize as .1 ms.
    ['decode-rounded-low', 0, {predicted_ms => .1, decode_ms => .1, predicted_per_second => 67114.09}],
    ['decode-rounded-high', 0, {predicted_ms => .1, decode_ms => .1, predicted_per_second => 196078.43}],
    ['decode-rounded-too-fast', 2, {predicted_ms => .1, decode_ms => .1, predicted_per_second => 250000}],
    ['decode-rounded-zero', 0, {predicted_ms => 0, decode_ms => 0, predicted_per_second => 250000}],
    ['decode-rounded-zero-invalid', 2, {predicted_ms => 0, decode_ms => 0, predicted_per_second => 999}],
    ['decode-unknown-rate', 2, {predicted_ms => 0, decode_ms => 0, predicted_per_second => undef}],
    ['decode-rate-rounds-zero', 0, {predicted_n => 2, decode_intervals => 1, predicted_ms => 250000, decode_ms => 250000, predicted_per_second => 0}],
);
my @protocol_cases = (
    ['engine-array', 2, {engine => [canonical(11)]}],
    ['engine-null', 2, {engine => undef}],
    ['http-array', 2, {http => [{complete => JSON::PP::false, aborted => JSON::PP::true}]}],
    ['http-null', 2, {http => undef}],
    ['http-incomplete', 2, {http => {complete => JSON::PP::false}}],
    ['http-aborted', 2, {http => {aborted => JSON::PP::true}}],
    ['http-complete', 0, {http => {complete => JSON::PP::true, aborted => JSON::PP::false}}],
);
sub timing_fixture {
    my ($mode, $body) = @_;
    for my $case (@prefill_cases, @decode_cases, @protocol_cases) {
        next unless $case->[0] eq $mode;
        @{$body->{timings}}{keys %{$case->[2]}} = values %{$case->[2]};
        delete $body->{timings}{prompt_per_second} if $mode eq 'prefill-missing-rate';
        return;
    }
}

sub server{my($mode)=@_;my$s=IO::Socket::INET->new(LocalAddr=>'127.0.0.1',LocalPort=>0,Listen=>1,ReuseAddr=>1) or die$!;my$port=$s->sockport;my$pid=fork();defined($pid) or die'fork';if(!$pid){$SIG{PIPE}='IGNORE';alarm 12;my$c=$s->accept or die$!;my$header='';while($header!~/\r\n\r\n\z/){my$n=sysread($c,my$b,1);$n or die'short HTTP headers';$header.=$b;length($header)<65536 or die'long headers';}my($len)=$header=~/\r\nContent-Length:\s*(\d+)/i;defined($len) or die'no content length';if($header=~/\r\nExpect:\s*100-continue/i){print {$c} "HTTP/1.1 100 Continue\r\n\r\n";$c->flush;}my$raw='';while(length($raw)<$len){my$n=sysread($c,my$b,$len-length($raw));$n or die'short body';$raw.=$b;}my$req=decode_json($raw);my($code)=$req->{messages}[0]{content}=~/vault is ([0-9a-f]{6})/;defined($code) or die'fixture request missing code';my$body={choices=>[{message=>{content=>"$code\n1 2 3"}}],timings=>{prompt_n=>100,prompt_ms=>2000.0,prompt_per_second=>50.0,predicted_n=>11,predicted_ms=>400.0,predicted_per_second=>25.0,decode_intervals=>10,decode_ms=>400.0}};my$status=200;
if($mode eq'wrong'){$body->{choices}[0]{message}{content}='wrong code';}elsif($mode eq'missing-answer'){delete$body->{choices};}elsif($mode eq'api-error'){$body={error=>{message=>'fixture error'}};}elsif($mode eq'missing-timings'){delete$body->{timings};}elsif($mode eq'bad-rate'){$body->{timings}{predicted_per_second}=123.0;}elsif($mode eq'http-error'){$status=503;$body={error=>'unavailable'};}elsif($mode=~/\Asingle-(null|zero)\z/){my$null=$1 eq'null';my$t=$body->{timings};$t->{predicted_n}=1;$t->{decode_intervals}=0;$t->{predicted_ms}=0;$t->{decode_ms}=0;$t->{predicted_per_second}=$null?undef:0;$t->{engine}=canonical(1) if$null;}elsif($mode eq'no-output'){my$t=$body->{timings};$t->{predicted_n}=0;$t->{decode_intervals}=0;$t->{predicted_ms}=0;$t->{decode_ms}=0;$t->{predicted_per_second}=undef;}elsif($mode eq'all-unknown'){$_=undef for values%{$body->{timings}};}elsif($mode=~/\Acanonical-(success|unknown|aborted)\z/){$body->{timings}{engine}=canonical(11);$body->{timings}{engine}{counts_known}=JSON::PP::false if$mode eq'canonical-unknown';$body->{timings}{engine}{aborted}=JSON::PP::true if$mode eq'canonical-aborted';}
timing_fixture($mode,$body);
my$payload=$mode eq'malformed'?'{broken':encode_json($body);my$reason=$status==200?'OK':'Service Unavailable';print {$c} "HTTP/1.1 $status $reason\r\nContent-Type: application/json\r\nContent-Length: ".length($payload)."\r\nConnection: close\r\n\r\n$payload";close$c;close$s;POSIX::_exit(0);}close$s;return($pid,"http://127.0.0.1:$port");}
sub run_depth{my($name,$url,$timeout)=@_;my$out="$tmp/$name";my$pid=fork();defined($pid) or die'fork';if(!$pid){$ENV{DEPTH_OUT}=$out;$ENV{DEPTH_TIMEOUT}=$timeout;open STDOUT,'>',"$tmp/$name.stdout" or die$!;open STDERR,'>',"$tmp/$name.stderr" or die$!;exec'bash',$script,'2','50',$url;die'exec';}my$timed=0;{local$SIG{ALRM}=sub{$timed=1;kill'TERM',$pid;};alarm 15;waitpid($pid,0);alarm 0;}die'depth fixture timeout' if$timed;my$status=$?;die'depth signaled' if$status&127;return($status>>8,readfile("$tmp/$name.stdout"),readfile("$tmp/$name.stderr"),$out);}
my@cases=(['success',0],['wrong',1],['missing-answer',2],['api-error',2],['missing-timings',2],['bad-rate',2],['malformed',2],['http-error',2],['single-null',0],['single-zero',0],['no-output',2],['all-unknown',2],['canonical-success',0],['canonical-unknown',2],['canonical-aborted',2]);
push @cases, map { [$_->[0], $_->[1]] } (@prefill_cases, @decode_cases, @protocol_cases);
my @accepted;
for my$case(@cases){my($mode,$expected)=@$case;my($pid,$url)=server($mode);my($rc,$stdout,$stderr,$out)=run_depth($mode,$url,3);waitpid($pid,0);is($?,0,"$mode HTTP server completed");is($rc,$expected,"$mode exit");push @accepted,$mode if $rc==0;ok(-f"$out/request.json"&&-f"$out/response.json","$mode retains HTTP artifacts");if(!$expected){like($stdout,qr/PASS/,"$mode correct response");if($mode eq'success'){like($stdout,qr/prompt 100 tok/,"prompt count");like($stdout,qr/10 intervals in 400\.0 ms/,"interval duration");like($stdout,qr/prefill 50 tok\/s  decode 25 tok\/s/,"successful prefill and decode rates");}}elsif($expected==1){like($stdout,qr/FAIL/,"wrong answer fails");unlike($stdout,qr/PASS/,"wrong answer never passes");}elsif($expected==2){is($stdout,'',"$mode emits no successful measurement");}diag($stderr) if$rc!=$expected;}
is_deeply(\@accepted, [map { $_->[0] } grep { $_->[1] == 0 } @cases],
    'only valid measurements enter the successful sample set');
my$closed=IO::Socket::INET->new(LocalAddr=>'127.0.0.1',LocalPort=>0,Listen=>1) or die$!;my$port=$closed->sockport;close$closed;my($rc,$stdout,$stderr)=run_depth('transport',"http://127.0.0.1:$port",1);is($rc,2,'transport exit');like($stderr,qr/transport failed/,'transport diagnostic');done_testing();
