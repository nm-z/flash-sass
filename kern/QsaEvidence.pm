package QsaEvidence;
use strict;
use warnings;
use Exporter 'import';
use JSON::PP;
use Digest::SHA;
use IO::Select;
use Time::HiRes qw(time sleep clock_gettime CLOCK_MONOTONIC);
use POSIX qw(strftime WNOHANG setpgid _exit);
use Cwd 'abs_path';
use File::Path 'make_path';
use File::Basename 'dirname';
our @EXPORT_OK = qw(sha put write_text run absolute new_output utc csv_rows);
my $json=JSON::PP->new->canonical->pretty;
sub sha {
 my($path)=@_;open my $f,'<:raw',$path or die "open $path: $!\n";
 my $h=Digest::SHA->new(256);$h->addfile($f);close $f or die "close $path: $!\n";return $h->hexdigest;
}
sub write_text {
 my($path,$text)=@_;open my $f,'>:raw',$path or die "write $path: $!\n";
 print {$f} $text or die "write $path: $!\n";close $f or die "close $path: $!\n";
}
sub put {my($path,$value)=@_;write_text($path,$json->encode($value));}
sub absolute {my($p)=@_;my $a=abs_path($p);defined($a) or die "resolve $p: $!\n";return $a;}
sub new_output {
 my($p)=@_;-e $p and die "output already exists: $p\n";
 make_path(dirname($p));mkdir($p) or die "mkdir $p: $!\n";return absolute($p);
}
sub utc {return strftime('%Y-%m-%dT%H:%M:%SZ',gmtime);}
# Capture both streams concurrently. Each child owns a private process group;
# timeouts and interruption terminate that group and reap the direct child.
sub run {
 my($cmd,$env,$timeout,$cwd)=@_;
 pipe(my $or,my $ow) or die "stdout pipe: $!\n";
 pipe(my $er,my $ew) or die "stderr pipe: $!\n";
 my $pid=fork();defined($pid) or die "fork: $!\n";
 if(!$pid) {
  close $or;close $er;
  eval {
   setpgid(0,0)==0 or die "setpgid: $!\n";
   open STDIN,'<','/dev/null' or die "stdin: $!\n";
   open STDOUT,'>&',$ow or die "stdout: $!\n";
   open STDERR,'>&',$ew or die "stderr: $!\n";
   close $ow;close $ew;
   %ENV=%$env;
   chdir($cwd) or die "chdir $cwd: $!\n" if defined $cwd;
   exec {$cmd->[0]} @$cmd;
   die "exec $cmd->[0]: $!\n";
  };
  my $error=$@||"child setup failed\n";
  print {$ew} $error if defined fileno($ew);
  print STDERR $error unless defined fileno($ew);
  _exit(127);
 }
 close $ow;close $ew;binmode $or;binmode $er;
 setpgid($pid,$pid);
 my $sel=IO::Select->new($or,$er);my %which=(fileno($or)=>0,fileno($er)=>1);
 my @buf=('','');my $start=clock_gettime(CLOCK_MONOTONIC);my($status,$timed)=(undef,0);
 my $stop=sub {kill 'KILL',-$pid;kill 'KILL',$pid unless defined $status;};
 local $SIG{INT}=sub{$stop->();die "interrupted\n";};
 local $SIG{TERM}=sub{$stop->();die "terminated\n";};
 eval {
  while($sel->count || !defined($status)) {
   if(!$timed && defined($timeout) && clock_gettime(CLOCK_MONOTONIC)-$start >= $timeout){$timed=1;$stop->();}
   for my $fh($sel->can_read(0.05)) {
    my $n=sysread($fh,my $chunk,65536);
    if(!defined $n){next if $!{EINTR};die "read child: $!\n";}
    if(!$n){$sel->remove($fh);close $fh;next;}
    $buf[$which{fileno($fh)}].=$chunk;
   }
   if(!defined($status)){my $w=waitpid($pid,WNOHANG);if($w==$pid){$status=$?;}elsif($w<0){next if $!{EINTR};die "waitpid: $!\n";}}
   last if $timed && defined($status) && clock_gettime(CLOCK_MONOTONIC)-$start >= $timeout+2;
   sleep(0.001) unless $sel->count || defined($status);
  }
 };
 if($@){my $error=$@;$stop->();waitpid($pid,0) unless defined $status;die $error;}
 if($timed){$buf[1].="\nRUNNER TIMEOUT\n";return($buf[0],$buf[1],124);}
 my $rc=($status&127)?-($status&127):($status>>8);return($buf[0],$buf[1],$rc);
}
sub csv_rows {
 my($path)=@_;open my $f,'<:raw',$path or die "open $path: $!\n";local $/;my $s=<$f>;close $f;
 my(@records,@row);my $field='';my $quoted=0;my $closed=0;my $at_start=1;
 for(my $i=0;$i<length($s);$i++) {
  my $c=substr($s,$i,1);
  if($quoted){if($c eq '"'){if(substr($s,$i+1,1) eq '"'){$field.='"';$i++;}else{$quoted=0;$closed=1;}}else{$field.=$c;}next;}
  if($c eq '"'){die "invalid CSV quote in $path\n" unless $at_start;$quoted=1;$at_start=0;next;}
  if($c eq ',' || $c eq "\n" || $c eq "\r") {
   push @row,$field;$field='';$at_start=1;$closed=0;
   if($c ne ','){ $i++ if $c eq "\r" && substr($s,$i+1,1) eq "\n";push @records,[@row];@row=(); }
   next;
  }
  die "data after closing CSV quote in $path\n" if $closed;
  $field.=$c;$at_start=0;
 }
 die "unterminated CSV quote in $path\n" if $quoted;
 if(@row || length($field) || !$at_start){push @row,$field;push @records,[@row];}
 return [] unless @records;my $header=shift @records;my %seen;
 for(@$header){die "duplicate CSV header in $path\n" if $seen{$_}++;}
 my @result;for my $r(@records){die "CSV column count in $path\n" unless @$r==@$header;my %h;@h{@$header}=@$r;push @result,\%h;}
 return \@result;
}
1;
