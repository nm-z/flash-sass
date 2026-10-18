#!/usr/bin/env perl
use strict;use warnings;use FindBin;use lib $FindBin::Bin;
use SocketEvidence qw(connect_unix send_text read_line close_socket monotonic begin_request end_request);
use IO::Socket::UNIX;use Socket 'SOCK_STREAM';use File::Temp 'tempdir';
use POSIX '_exit';use Time::HiRes 'sleep';use Test::More;
my $tmp=tempdir(CLEANUP=>1);my $seq=0;my @children;
END {local $?;for my $pid(@children){kill 'KILL',$pid;waitpid($pid,0);}}
sub server{
 my($body)=@_;my $path="$tmp/sock".++$seq;
 my $listen=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Local=>$path,Listen=>1) or die $!;
 my $pid=fork();defined($pid) or die $!;
 if(!$pid){@children=();alarm 5;my $s=$listen->accept or _exit(2);close $listen;my $ok=eval{$body->($s);1};close $s;_exit($ok?0:3);}
 close $listen;push @children,$pid;return($path,$pid);
}
sub reaped{my($pid,$label)=@_;waitpid($pid,0)==$pid or die $!;my $status=$?;@children=grep{$_!=$pid}@children;is($status,0,$label);}
my($path,$pid)=server(sub{
 my($s)=@_;binmode $s;my $request='';
 while(length($request)<11){my $n=sysread($s,my $b,11-length($request));$n or die "short request";$request.=$b;}
 $request eq "alpha\nbeta\n" or die "changed request";
 syswrite($s,'one')==3 or die $!;sleep .01;
 my $rest="\0two\nthree\ntail";syswrite($s,$rest)==length($rest) or die $!;
});
my $c=connect_unix($path,1);send_text($c,"alpha\nbeta\n");
is(read_line($c),"one\0two\n",'fragmented line retains NUL');
is(read_line($c),"three\n",'coalesced second line');
is(read_line($c),'tail','unterminated final line at EOF');
ok(!defined(read_line($c)),'EOF exhausted');close_socket($c);reaped($pid,'fragment server completed');
pipe(my $gate_r,my $gate_w) or die $!;
($path,$pid)=server(sub{my($s)=@_;close $gate_w;syswrite($s,'unterminated')==12 or die $!;sysread($gate_r,my $release,1)==1 or die $!;syswrite($s,"\n")==1 or die $!;close $gate_r;});
close $gate_r;$c=connect_unix($path,.2);my $start=monotonic();my $ok=eval{read_line($c);1};my $elapsed=monotonic()-$start;
ok(!$ok && $@=~/socket read timeout/,'partial line times out');
ok($elapsed>=.15 && $elapsed<2,'read deadline is bounded');
syswrite($gate_w,'G')==1 or die $!;close $gate_w;$c->{timeout}=1;
is(read_line($c),"unterminated\n",'partial buffer survives timeout');ok(!defined(read_line($c)),'EOF after resumed line');close_socket($c);reaped($pid,'timeout server completed');
($path,$pid)=server(sub{my($s)=@_;syswrite($s,"12345\n")==6 or die $!;});
$c=connect_unix($path,1);$ok=eval{read_line($c,6);1};ok(!$ok && $@=~/oversized socket response/,'optional response cap enforced');close_socket($c);reaped($pid,'cap server completed');
pipe(my $hold_r,my $hold_w) or die $!;
($path,$pid)=server(sub{close $hold_w;sysread($hold_r,my $release,1)==1 or die $!;close $hold_r;});
close $hold_r;$c=connect_unix($path,.2);$start=monotonic();$ok=eval{send_text($c,'x'x4_000_000);1};$elapsed=monotonic()-$start;
ok(!$ok && $@=~/socket write timeout/,'blocked large write times out');
ok($elapsed>=.15 && $elapsed<2,'nonblocking write deadline is bounded');close_socket($c);syswrite($hold_w,'G')==1 or die $!;close $hold_w;reaped($pid,'write deadline server completed');
($path,$pid)=server(sub{
 my($s)=@_;local $SIG{PIPE}='IGNORE';
 for(1..40){last unless syswrite($s,"valid line\n");sleep .01;}
});
$c=connect_unix($path,1);begin_request($c,.15);$start=monotonic();my $lines=0;
$ok=eval{while(defined(my $line=read_line($c))){die "changed line\n" unless $line eq "valid line\n";$lines++;}1};
$elapsed=monotonic()-$start;
ok(!$ok && $@=~/socket read timeout/,'regular valid lines cannot extend whole-request deadline');
ok($lines>2 && $elapsed<.8,'total request duration bounded despite progress');
$c->{buffer}="buffered\0line\n";$c->{deadline}=monotonic()-1;
$ok=eval{read_line($c);1};ok(!$ok && $@=~/timeout/,'expired request rejects complete buffered line');
$ok=eval{send_text($c,'');1};ok(!$ok && $@=~/timeout/,'expired request rejects even empty send');
end_request($c);is(read_line($c),"buffered\0line\n",'deadline rejection preserves buffered bytes');
close_socket($c);reaped($pid,'total-deadline server completed');
done_testing;
