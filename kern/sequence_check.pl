#!/usr/bin/env perl
use strict;use warnings;
use FindBin;use lib $FindBin::Bin;
use SocketEvidence qw(connect_unix send_text read_line close_socket monotonic);
use QsaEvidence 'put';use IO::Select;use JSON::PP;use POSIX '_exit';use File::Temp 'tempfile';
$|=1;$SIG{PIPE}='IGNORE';$SIG{INT}=sub{die "interrupted\n";};$SIG{TERM}=sub{die "terminated\n";};
my($sockpath,$report)=@ARGV;defined($sockpath) && @ARGV<=2 or die "usage: $0 SOCKET [REPORT.json]\n";
my @records;my $passed=0;
sub check{my($ok,$why)=@_;$ok or die "$why\n";}
sub same_tokens{my($a,$b)=@_;return JSON::PP->new->encode($a->{tokens}) eq JSON::PP->new->encode($b->{tokens});}
sub ask{
 my(%a)=@_;my $slot=$a{slot};my $ids=$a{ids};my $new=$a{new}//6;my $snap=$a{snap}//0;my $c=connect_unix($sockpath,120);my @commands;
 push @commands,"sample $a{sample} .7 8 .9" if defined($a{sample});push @commands,"stop $a{stop}" if defined($a{stop});push @commands,"snap $snap" if $snap;push @commands,"img $a{image} 1" if $a{image};push @commands,'chat_slot '.$slot.' '.$new.' '.join(' ',@$ids);
 if($a{ready}){syswrite($a{ready},'R',1)==1 or die "barrier ready: $!\n";close $a{ready};my $n=sysread($a{gate},my $byte,1);defined($n)&&$n==1 or die "barrier release failed\n";close $a{gate};}
 send_text($c,join("\n",@commands)."\n");my(@tokens,@raw);my $first;my $started=monotonic();
 while(defined(my $line=read_line($c))){$line=~s/\n\z//;push @raw,$line;my @v=split ' ',$line;next unless @v;die "$line\n" if $v[0] eq 'error:';
  if($v[0] eq 'tok'){defined($v[1]) && $v[1]=~/\A[+-]?\d+\z/ or die "invalid token line: $line\n";push @tokens,0+$v[1];$first//=monotonic();}
  if($v[0] eq 'done'){@v>=9 or die "short done line: $line\n";for my $i(1,2,6,8){$v[$i]=~/\A[+-]?\d+\z/ or die "invalid done integer: $line\n";}check($v[1]==@tokens,"done token count: $line");
   my $result={tokens=>\@tokens,ended=>0+$v[2],resumed=>0+$v[6],slot=>0+$v[8],first=>$first,done=>monotonic()};push @records,{request=>{slot=>$slot,ids=>$ids,new=>$new,snap=>$snap,sample=>$a{sample},stop=>$a{stop},image=>$a{image}?JSON::PP::true:JSON::PP::false},started=>$started,result=>$result,raw=>\@raw};close_socket($c);return $result;
  }
 }die "engine closed without done\n";
}
sub pair{
 my($left,$right)=@_;pipe(my $ready_r,my $ready_w) or die "ready pipe: $!\n";pipe(my $gate_r,my $gate_w) or die "gate pipe: $!\n";my @children;
 for my $arg($left,$right){pipe(my $r,my $w) or die "result pipe: $!\n";my $pid=fork();defined($pid) or die "fork: $!\n";
  if(!$pid){close $r;close $ready_r;close $gate_w;close $_->{read} for @children;my $answer;
   eval{my $v=ask(%$arg,ready=>$ready_w,gate=>$gate_r);$answer={result=>$v,record=>$records[-1]};};if($@){$answer={error=>$@};syswrite($ready_w,'E',1) if defined(fileno($ready_w));}
   close $ready_w if defined(fileno($ready_w));close $gate_r if defined(fileno($gate_r));my $s=JSON::PP->new->encode($answer);my $offset=0;while($offset<length($s)){my $n=syswrite($w,$s,length($s)-$offset,$offset);_exit(2) unless defined($n)&&$n>0;$offset+=$n;}close $w;_exit(0);
  }close $w;push @children,{pid=>$pid,read=>$r,data=>''};
 }close $ready_w;close $gate_r;my @results;
 eval{
  my $rs=IO::Select->new($ready_r);my $ready='';while(length($ready)<2){$rs->can_read(120) or die "client barrier timeout\n";my $n=sysread($ready_r,my $s,2-length($ready));defined($n)&&$n>0 or die "client barrier closed\n";$ready.=$s;}close $ready_r;syswrite($gate_w,'GG',2)==2 or die "release clients: $!\n";close $gate_w;
  my $sel=IO::Select->new(map {$_->{read}}@children);my %byfd=map {fileno($_->{read})=>$_}@children;
  while($sel->count){my @r=$sel->can_read(120);@r or die "concurrent clients timeout\n";for my $f(@r){my $n=sysread($f,my $s,65536);defined($n) or die "read client result: $!\n";if(!$n){$sel->remove($f);close $f;}else{$byfd{fileno($f)}{data}.=$s;}}}
  my @newrecords;my $pair_error='';for my $child(@children){waitpid($child->{pid},0)==$child->{pid} or die "wait client: $!\n";my $status=$?;$child->{reaped}=1;if($status){$pair_error.="client process failed status=$status\n";next;}my $answer=JSON::PP->new->decode($child->{data});if(exists($answer->{error})){$pair_error.=$answer->{error};next;}push @results,$answer->{result};push @newrecords,$answer->{record};}
  push @records,sort {$a->{result}{done}<=>$b->{result}{done}}@newrecords;die $pair_error if length($pair_error);
 };
 if($@){my $error=$@;for my $child(@children){unless($child->{reaped}){kill 'KILL',$child->{pid};waitpid($child->{pid},0);}}die $error;}return @results;
}
sub main{
 my $c=connect_unix($sockpath,120);send_text($c,"info\n");my $line=read_line($c);defined($line) or die "missing info\n";close_socket($c);my @i=split ' ',$line;check(@i>=5 && $i[0] eq 'info',"invalid info: $line");for(@i[1..4]){/\A[+-]?\d+\z/ or die "invalid info integer: $line\n";}my($cap,$images,$nslots,$quantum)=map {0+$_}@i[1..4];check($nslots>=2,'at least two sequence slots required');
 my $A=[1,2,3,4,5];my $B=[65,66,67,68,69];
 for my $sampling(undef,12345){my $a={slot=>0,ids=>$A,sample=>$sampling};my $b={slot=>1,ids=>$B,sample=>defined($sampling)?98765:undef};my @serial=(ask(%$a),ask(%$b));my @inter=pair($a,$b);
  for my $n(0,1){check(same_tokens($inter[$n],$serial[$n]),'serial/interleaved tokens differ');check($inter[$n]{slot}==$n,'wrong concurrent slot');}$a->{snap}=2;$b->{snap}=2;my @snap=pair($a,$b);my @resumed=pair($a,$b);
  for my $n(0,1){check(same_tokens($snap[$n],$serial[$n]),'snapshot tokens differ');check(same_tokens($resumed[$n],$serial[$n]),'resumed tokens differ');check($resumed[$n]{resumed}==2,'wrong resumed prefix');}print 'serial/interleaved/snapshot match, sample=',defined($sampling)?$sampling:'none',"\n";
 }
 my $length=$cap-8;$length=96 if $length>96;my @longids=$length>0?(map {100+($_%100)}(0..$length-1)):();
 my($long,$short)=pair({slot=>0,ids=>\@longids,new=>1},{slot=>1,ids=>[@$B[0,1]],new=>2});check($short->{done}<$long->{done},'short decode did not finish during other prefill');print "short decode completed while other slot was prefilling\n";
 my $first=ask(slot=>0,ids=>$A)->{tokens}[0];check(defined($first),'baseline has no token');my $ended=ask(slot=>0,ids=>$A,stop=>$first);check($ended->{ended}==1 && @{$ended->{tokens}}==1,'stop did not end after first token');
 my $cancel=connect_unix($sockpath,120);send_text($cancel,'chat_slot 0 1 '.join(' ',@longids)."\n");close_socket($cancel);
 check(@{ask(slot=>1,ids=>$B)->{tokens}}>0,'other slot after cancellation has no tokens');check(@{ask(slot=>0,ids=>$A)->{tokens}}>0,'cancelled slot reuse has no tokens');print "stop, cancelled request and slot reuse passed\n";
 if($images){my($f,$path)=tempfile('flash-sequence-image-XXXXXX',SUFFIX=>'.bin',TMPDIR=>1,UNLINK=>1);binmode $f;print {$f} pack('V5',1,1,1,7,2560),pack('V*',(0)x2560),pack('V3',0,1,2) or die "write image fixture: $!\n";close $f or die "close image fixture: $!\n";my $a={slot=>0,ids=>$A,image=>$path};my $b={slot=>1,ids=>$B};my @base=(ask(%$a),ask(%$b));my @inter=pair($a,$b);for my $n(0,1){check(same_tokens($base[$n],$inter[$n]),'image/plain slots not isolated');}unlink $path or die "unlink image fixture: $!\n";print "image/M-RoPE and plain-text slots isolated\n";}
 $passed=1;print "sequence_check: passed\n";return 0;
}
my $rc=eval{main()};my $error=$@;if(defined($report)){eval{put($report,{passed=>$passed?JSON::PP::true:JSON::PP::false,requests=>\@records});};$error.=$@ if $@;}if($error){print STDERR $error;exit 1;}exit $rc;
