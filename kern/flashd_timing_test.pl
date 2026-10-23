#!/usr/bin/env perl
use strict;use warnings;use FindBin;use File::Temp qw(tempdir);use File::Path 'make_path';
use IO::Socket::INET;use IO::Socket::UNIX;use IO::Select;use JSON::PP;
use POSIX qw(_exit WNOHANG);use Socket qw(SOCK_STREAM);use Test::More;use Time::HiRes qw(time sleep);
my $frontend=$ARGV[0]//"$FindBin::Bin/flashd_timing_test";
my $fixture=$ARGV[1]//"$FindBin::Bin/timings_test";my $json=JSON::PP->new->canonical;
my $dir;
if(defined $ARGV[2]){die "output already exists" if -e $ARGV[2];make_path($ARGV[2]);$dir=$ARGV[2];}else{$dir=tempdir(CLEANUP=>1);}
my $socket_path="$dir/engine.sock";my $mode_path="$dir/mode.json";my $log_path="$dir/frontend.log";
my $fake_log_path="$dir/fake-engine.log";my $gguf_path="$dir/tokenizer.gguf";my $parent_pid=$$;my @children;
$SIG{PIPE}='IGNORE';
END {local $?;if($$==$parent_pid){kill 'TERM',@children if @children;waitpid($_,0) for @children;}}
sub write_file {my($path,$text)=@_;open my $f,'>:raw',$path or die "open $path: $!";print {$f} $text or die "write $path: $!";close $f or die "close $path: $!";}
sub read_file {my($path)=@_;open my $f,'<:raw',$path or die "open $path: $!";local $/;my $text=<$f>;close $f;return defined($text)?$text:'';}
sub set_mode {my($name,$mode)=@_;write_file($mode_path,$json->encode({name=>$name,mode=>$mode}));}
sub gguf_string {my($s)=@_;return pack('Q<',length($s)).$s;}
sub gguf_kv {my($key,$type,$payload)=@_;return gguf_string($key).pack('V',$type).$payload;}
sub create_tokenizer {
 my @tokens=('<|im_end|>','</think>','a'..'z');my @types=(3,3,(1)x26);
 my @kv=(
 gguf_kv('tokenizer.ggml.model',8,gguf_string('gpt2')),
 gguf_kv('tokenizer.ggml.pre',8,gguf_string('qwen35')),
 gguf_kv('tokenizer.ggml.tokens',9,pack('VQ<',8,scalar @tokens).join('',map {gguf_string($_)}@tokens)),
 gguf_kv('tokenizer.ggml.token_type',9,pack('VQ<',5,scalar @types).pack('l<*',@types)),
 gguf_kv('tokenizer.ggml.merges',9,pack('VQ<',8,0)),
 gguf_kv('tokenizer.ggml.eos_token_id',4,pack('V',0)),
 gguf_kv('tokenizer.ggml.add_bos_token',7,pack('C',0)),
 gguf_kv('tokenizer.chat_template',8,gguf_string('{{ messages[0].content }}')));
 my $bytes='GGUF'.pack('VQ<Q<',3,0,scalar @kv).join('',@kv);
 $bytes.="\0"x((32-length($bytes)%32)%32);write_file($gguf_path,$bytes);
}
sub fixture_json {
 my($mode,$prompt,$generated,$resumed)=@_;
 open my $f,'-|',$fixture,$mode,$prompt,$generated,$resumed or die "start timing fixture: $!";
 local $/;my $text=<$f>;close $f or die "timing fixture failed with status $?";
 die "timing fixture returned no JSON" unless defined $text;$text=~s/\r?\n\z//;$json->decode($text);return $text;
}
sub send_all {my($socket,$bytes)=@_;my $offset=0;while($offset<length $bytes){my $n=syswrite($socket,$bytes,length($bytes)-$offset,$offset);return 0 unless defined($n)&&$n>0;$offset+=$n;}return 1;}
sub fake_engine {
 my($listener)=@_;open STDOUT,'>>:raw',$fake_log_path or die $!;open STDERR,'>&',\*STDOUT or die $!;$|=1;local $SIG{PIPE}='IGNORE';
 while(my $connection=$listener->accept){
  $connection->autoflush(1);my $mode=$json->decode(read_file($mode_path));
  while(my $line=<$connection>){
   print "[$mode->{name}] $line";
   if($line eq "info\n"){print {$connection} "info 256 0\n";}
   elsif($line eq "timings 1\n" || $line eq "timings on\n"){
    print {$connection} $mode->{mode} eq 'legacy'
     ?"error: gen|chat <n_new> <id>... | stop [<id> [<id>]] | info | cubin <path> | quit\n":"ok: timings 1\n";
   }elsif($line eq "cancel_on_eof 1\n"){
    print {$connection} $mode->{mode}=~/\A(?:legacy|legacycap)\z/
     ?"error: gen|chat <n_new> <id>... | stop [<id> [<id>]] | info | cubin <path> | quit\n":"ok: cancel_on_eof 1\n";
   }elsif($line=~/\A(?:stop|sample|snap)\b/){print {$connection} "ok\n";}
   elsif($line=~/\Achat /){
    my @words=split /\s+/,$line;my $prompt=@words-2;my $kind=$mode->{mode};
    my $resumed=($kind eq 'cold'||$kind eq 'prefillabort')?0:$prompt-1;
    my $generated=($kind eq 'one'||$kind eq 'stop')?1:($kind eq 'partial'||$kind eq 'abort')?2:($kind eq 'unknown'||$kind eq 'nooutput'||$kind eq 'prefillabort')?0:$kind=~/cap|cancel/?5:3;
    last if $kind eq 'nooutput';
    my $canonical=fixture_json(($kind eq 'legacy'||$kind eq 'multi_crlf')?'multi':$kind eq 'frontendcap'?'abort':$kind,$prompt,$generated,$resumed);
    write_file("$dir/$mode->{name}.expected.json",$canonical."\n");sleep .08;
    for my $i(0..($kind eq 'cancelbusy'?99:$generated-1)){
     my $token=($kind eq 'stop'&&$i==$generated-1)?0:2+($i%26);last unless print {$connection} "tok $token\n";sleep .01;
     if($i==3 && $kind eq 'cancelstall'){sleep .5;last;}
     if($i==3 && $kind eq 'cancelfragment'){
      for(1..100){last unless print {$connection} 'x';sleep .01;}last;
     }
    }
    if($kind eq 'frontendcap'){
     my $unexpected=<$connection>;die "expected half-close cancellation" if defined $unexpected;
     print "[$mode->{name}] half-close cancellation received\n";
    }
    last if $kind=~/^cancel/;
    print {$connection} 'timings '.$canonical.($kind eq 'multi_crlf'?"\r\n":"\n") if $kind ne 'legacy';
    if($kind eq 'partial'||$kind eq 'unknown'){print {$connection} "error: device fault; the core rebuilds the contexts and the engine\n";}
    else{my $ended=$kind eq 'stop'?1:0;my $run_ms=($kind eq 'one'||$kind eq 'stop')?'12.3':'56.8';print {$connection} "done $generated $ended $prompt $run_ms 12.3 $resumed $resumed\n";}
    last;
   }else{die "unexpected fake-engine command: $line";}
  }close $connection;
 }
}
sub reserve_port {my $socket=IO::Socket::INET->new(LocalAddr=>'127.0.0.1',LocalPort=>0,Listen=>1,Proto=>'tcp',ReuseAddr=>1) or die "reserve TCP port: $!";my $port=$socket->sockport;close $socket;return $port;}
sub start_frontend {
 my($port)=@_;my $pid=fork();die "fork frontend: $!" unless defined $pid;
 if(!$pid){
  open STDOUT,'>>:raw',$log_path or _exit(120);open STDERR,'>&',\*STDOUT or _exit(121);
  $ENV{FLASHD_TEST_ANSWER_MAX}=3;$ENV{FLASHD_TEST_CANCEL_DRAIN_MS}=200;$ENV{FLASHD_THINKING}=0;$ENV{FLASHD_GREEDY}=1;delete $ENV{FLASHVIS_SOCK};
  exec {$frontend} $frontend,$gguf_path,$socket_path,"127.0.0.1:$port" or _exit(122);
 }
 push @children,$pid;my $deadline=time+10;
 while(time<$deadline){
  my $socket=IO::Socket::INET->new(PeerAddr=>'127.0.0.1',PeerPort=>$port,Proto=>'tcp',Timeout=>.1);
  if($socket){close $socket;return;}
  if(waitpid($pid,WNOHANG)==$pid){die "frontend exited at startup:\n".(-e $log_path?read_file($log_path):'');}
  sleep .02;
 }die "frontend did not listen:\n".(-e $log_path?read_file($log_path):'');
}
sub dechunk {
 my($wire)=@_;my $body='';
 while(1){
  $wire=~s/\A([0-9a-fA-F]+)(?:;[^\r\n]*)?\r\n// or die "malformed HTTP chunk size";my $size=hex($1);
  if(!$size){die "malformed final chunk" unless $wire=~/\A\r\n/;return $body;}
  die "short HTTP chunk" if length($wire)<$size+2;$body.=substr($wire,0,$size,'');
  die "missing chunk CRLF" unless substr($wire,0,2,'') eq "\r\n";
 }
}
sub http_request {
 my($port,$stream,$max_tokens)=@_;
 my $body=$json->encode({messages=>[{role=>'user',content=>'abcdefghijklmnopqrst'}],stream=>$stream?JSON::PP::true:JSON::PP::false,max_tokens=>$max_tokens,temperature=>0});
 my $socket=IO::Socket::INET->new(PeerAddr=>'127.0.0.1',PeerPort=>$port,Proto=>'tcp',Timeout=>5) or die "connect frontend: $!";
 my $request="POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: ".length($body)."\r\nConnection: close\r\n\r\n".$body;
 send_all($socket,$request) or die "write HTTP request";my $select=IO::Select->new($socket);my $wire='';
 while(1){die "HTTP response timeout" unless $select->can_read(10);my $n=sysread($socket,my $chunk,65536);die "HTTP response read: $!" unless defined $n;last unless $n;$wire.=$chunk;}
 close $socket;my($headers,$payload)=split /\r\n\r\n/,$wire,2;die "missing HTTP response headers" unless defined $payload;
 $headers=~/\AHTTP\/1\.1 ([0-9]{3}) / or die "invalid HTTP status";my $status=0+$1;
 if($headers=~/Transfer-Encoding:\s*chunked/i){$payload=dechunk($payload);}
 elsif($headers=~/Content-Length:\s*([0-9]+)/i){die "wrong HTTP content length" unless length($payload)==$1;}
 my @events;my $response;
 if($stream){
  # An SSE client joins only data fields. An unprefixed protocol newline
  # inside JSON must not be silently accepted as part of the event payload.
  my(@data,@messages);
  for my $line(split /\r\n|\r|\n/,$payload,-1){
   if($line eq ''){push @messages,join("\n",@data) if @data;@data=();next;}
   push @data,defined($1)?$1:'' if $line=~/\Adata(?:: ?(.*))?\z/;
  }
  die "SSE missing terminal [DONE]" unless @messages && pop(@messages) eq '[DONE]';
  @events=map {$json->decode($_)} @messages;
  ($response)=grep {exists $_->{timings}}reverse @events;
  die "SSE response has no final timing record:\n$payload" unless $response;
 }
 else{$response=$json->decode($payload);}
 return {status=>$status,headers=>$headers,payload=>$payload,response=>$response,events=>\@events,wire=>$wire};
}
sub assert_aliases {my($timing,$engine,$label)=@_;for my $key(qw(prompt_n cache_n prompt_ms prompt_per_second predicted_n predicted_ms predicted_per_second decode_intervals decode_ms)){is_deeply($timing->{$key},$engine->{$key},"$label alias $key equals canonical source");}}
sub check_case {
 my($port,$mode,$stream)=@_;my $name=$mode.($stream?'-stream':'-plain');set_mode($name,$mode);
 my $max_tokens=($mode eq 'one'||$mode eq 'stop')?1:$mode=~/cap|cancel/?5:3;
 my $r=http_request($port,$stream,$max_tokens);write_file("$dir/$name.http",$r->{wire});
 my $response=$r->{response};my $t=$response->{timings};ok(ref($t) eq 'HASH',"$name has timing object");return unless ref($t) eq 'HASH';
 my $h=$t->{http};is($h->{clock},'CLOCK_MONOTONIC',"$name host clock");is($h->{start_event},'request_worker_entry',"$name host origin");
 like($h->{request_id},qr/\A\d+-\d+\z/,"$name request log identity");
 ok(!defined($h->{response_write_complete_ms}),"$name does not predict future write completion");
 for my $key(qw(request_read_ms prequeue_ms queue_ms tokenization_ms setup_ms engine_submitted_ms completion_record_created_ms)){ok(defined($h->{$key})&&$h->{$key}>=0,"$name $key is nonnegative");}
 if($mode eq 'legacy'){
  ok(!defined($t->{engine}),"$name engine JSON unavailable");ok(!$h->{engine_timing_available},"$name availability false");is($r->{status},200,"$name old engine supported");is($t->{predicted_n},3,"$name legacy output count");return;
 }
 if($mode eq 'nooutput'){
  ok(!defined($t->{engine}),"$name no stale canonical JSON");ok(!$h->{engine_timing_available},"$name unavailable");is($h->{raw_token_events},0,"$name no observed tokens");
  for my $key(qw(first_token_observed_ms first_text_send_ms ttft_ms socket_decode_tokens_per_second)){ok(!defined($h->{$key}),"$name $key null");}
  ok(!defined($t->{predicted_per_second}),"$name unknown engine rate null");ok(exists($response->{error}),"$name error response");is($r->{status},$stream?200:502,"$name error transport status");return;
 }
 if($mode eq 'legacycap'||$mode=~/^cancel/){
  ok($h->{aborted},"$name frontend cap is cancellation");ok(!$h->{complete},"$name incomplete");ok(!$h->{engine_timing_available},"$name no invented final engine snapshot");ok(!defined($t->{engine}),"$name engine JSON null");cmp_ok($h->{raw_token_events},'>=',4,"$name fourth token triggers cap three");ok(!defined($h->{socket_decode_tokens_per_second}),"$name cancelled socket rate null");ok(!defined($t->{predicted_per_second}),"$name unknown final engine rate null");
  if($mode=~/cancel/){ok($h->{cancel_drain_timed_out},"$name drain deadline enforced");cmp_ok($h->{cancel_drain_ms},'<',450,"$name bounded drain");}
  else{ok(!defined($h->{cancel_drain_ms}),"$name legacy engine not half-closed");}
  return;
 }
 my $expected_text=read_file("$dir/$name.expected.json");$expected_text=~s/\r?\n\z//;my $expected=$json->decode($expected_text);
 is_deeply($t->{engine},$expected,"$name exact canonical values forwarded");
 ok(index($r->{payload},'"engine":'.$expected_text)>=0,"$name raw JSON forwarded without reformatting");
 ok($h->{engine_timing_available},"$name engine timing available");assert_aliases($t,$expected,$name);
 is($expected->{prompt_tokens},20,"$name actual tokenizer prompt size");
 if($mode eq 'unknown'){ok(!defined($expected->{resumed_positions}),"$name unknown resumed count is null");}
 else{is($expected->{resumed_positions},($mode eq 'prefillabort'||$mode eq 'cold')?0:19,"$name resumed count");}
 if($mode eq 'partial'||$mode eq 'unknown'){
  ok(exists($response->{error}),"$name engine failure");is($r->{status},$stream?200:502,"$name failure transport");ok(!$h->{complete},"$name failure incomplete");
  ok(!defined($expected->{decode_tokens_per_second}),"$name engine decode rate null");ok(!defined($expected->{prompt_positions_per_second}),"$name engine prompt rate null");ok(!defined($h->{socket_decode_tokens_per_second}),"$name failed socket rate null");ok(!defined($t->{predicted_per_second}),"$name legacy decode rate null");
 }else{is($r->{status},200,"$name successful transport");ok(!exists($response->{error}),"$name no error");}
 if($mode eq 'partial'){is($expected->{error_code},-2,"$name fault preserved");ok(!$expected->{counts_final},"$name partial counters");is($expected->{generated_tokens},2,"$name partial output count");is($h->{raw_token_events},2,"$name observed count");ok(!defined($expected->{stop_token_observed}),"$name unknown stop null");}
 if($mode eq 'unknown'){ok(!$expected->{counts_known},"$name unknown counters");ok(!defined($expected->{generated_tokens}),"$name generated count null");ok(!defined($expected->{positions_completed}),"$name positions null");is($h->{raw_token_events},0,"$name zero observations");ok(!defined($h->{first_token_observed_ms}),"$name first observation null");ok(!defined($h->{ttft_ms}),"$name delivery null");}
 if($mode eq 'abort'||$mode eq 'prefillabort'||$mode eq 'frontendcap'){ok($expected->{aborted},"$name engine abort");ok($h->{aborted},"$name HTTP reflects abort");ok(!$h->{complete},"$name abort incomplete");ok(!$expected->{stop_token_observed},"$name abort is not model stop");ok(!defined($h->{socket_decode_tokens_per_second}),"$name aborted socket rate null");is($response->{choices}[0]{finish_reason},'cancelled',"$name cancellation finish reason");}
 if($mode eq 'prefillabort'){
  is($expected->{positions_completed},7,"$name actual prompt progress");is($expected->{prompt_positions_processed},7,"$name prompt-only work");
  is($expected->{generated_tokens},0,"$name no generated output");is($response->{usage}{completion_tokens},0,"$name nonnegative zero usage");
  for my $key(qw(first_observed_ms post_first_ms decode_tokens_per_second)){ok(!defined($expected->{$key}),"$name $key unavailable");}
  ok(!defined($h->{first_token_observed_ms})&&!defined($h->{first_text_send_ms}),"$name no host observation/delivery");
 }
 if($mode eq 'frontendcap'){
  ok($h->{cancel_requested}&&!$h->{cancel_drain_timed_out},"$name cancellation drained");
  is($expected->{generated_tokens},5,"$name final core count includes in-flight output");is($h->{raw_token_events},5,"$name drains all protocol outputs");
  is($response->{usage}{completion_tokens},5,"$name usage preserves final core count");
  if(!$stream){is($response->{choices}[0]{message}{content},'abc',"$name content stops at frontend budget");}
 }
 if($mode eq 'one'||$mode eq 'stop'){is($expected->{generated_tokens},1,"$name one generated token");is($expected->{decode_intervals},0,"$name zero decode intervals");ok(!defined($expected->{decode_tokens_per_second}),"$name canonical rate null");ok(!defined($t->{predicted_per_second}),"$name legacy rate null");ok(!defined($h->{socket_decode_tokens_per_second}),"$name socket rate null");}
 if($mode eq 'stop'){ok($expected->{stop_token_observed},"$name stop observed");is($response->{choices}[0]{finish_reason},'stop',"$name stop finish reason");ok(!defined($h->{first_text_send_ms}),"$name hidden stop not delivery");ok(!defined($h->{ttft_ms}),"$name role not delivery");}
 if($mode eq 'multi'||$mode eq 'multi_crlf'){is($expected->{first_observed_ms},12.3456789,"$name unrounded first");is($expected->{run_ms},56.7890123,"$name unrounded run");ok($expected->{decode_tokens_per_second}>0,"$name engine rate");ok($h->{socket_decode_tokens_per_second}>0,"$name socket rate");ok($h->{complete},"$name complete");}
 ok($h->{first_token_observed_ms}>=60,"$name first token includes engine delay") if $mode ne 'unknown'&&$mode ne 'prefillabort';
 if($stream&&$mode ne 'stop'&&$mode ne 'unknown'&&$mode ne 'prefillabort'){
  ok(defined($h->{first_text_send_ms}),"$name first delivery measured");ok($h->{first_text_send_ms}>=60,"$name role excluded");
  is($h->{ttft_ms},$h->{first_text_send_ms},"$name TTFT aliases delivery");ok($h->{first_text_send_ms}>=$h->{first_token_observed_ms},"$name delivery follows observation");
  my @role=grep {exists($_->{choices})&&exists($_->{choices}[0]{delta})&&exists($_->{choices}[0]{delta}{role})}@{$r->{events}};is(scalar @role,1,"$name actual role event");
 }elsif(!$stream){ok(!defined($h->{first_text_send_ms}),"$name nonstream future delivery null");ok(!defined($h->{ttft_ms}),"$name nonstream future TTFT null");}
}
ok(-x $frontend,'compiled actual frontend exists') or BAIL_OUT("build $frontend");
ok(-x $fixture,'compiled canonical C fixture exists') or BAIL_OUT("build $fixture");
is(system($frontend,'--unit'),0,'actual C role/keepalive/admitted-error tests');
create_tokenizer();set_mode('startup','multi');
my $listener=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Local=>$socket_path,Listen=>16) or die "listen fake engine: $!";
my $fake_pid=fork();die "fork fake engine: $!" unless defined $fake_pid;
if(!$fake_pid){eval{fake_engine($listener)};if($@){print STDERR $@;_exit(123);}_exit(0);}
push @children,$fake_pid;close $listener;my $port=reserve_port();start_frontend($port);
for my $stream(0,1){for my $mode(qw(multi multi_crlf cold nooutput one stop partial abort prefillabort unknown legacy frontendcap legacycap cancelstall cancelbusy cancelfragment)){subtest $mode.($stream?' streaming':' nonstreaming')=>sub{check_case($port,$mode,$stream);};}}
diag("frontend log:\n".read_file($log_path)) if !Test::More->builder->is_passing;
done_testing;
