package SocketEvidence;
use strict;use warnings;
use Exporter 'import';use IO::Socket::UNIX;use IO::Select;use Socket 'SOCK_STREAM';
use Fcntl qw(F_GETFL F_SETFL O_NONBLOCK);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
our @EXPORT_OK=qw(connect_unix send_text read_line close_socket monotonic);
sub monotonic{return clock_gettime(CLOCK_MONOTONIC);}
sub connect_unix{my($path,$timeout)=@_;$timeout//=120;my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$path,Timeout=>$timeout) or die "connect $path: $!\n";binmode $s;my $flags=fcntl($s,F_GETFL,0);defined($flags) or die "socket flags: $!\n";fcntl($s,F_SETFL,$flags|O_NONBLOCK) or die "socket nonblocking: $!\n";return {socket=>$s,buffer=>'',timeout=>$timeout,eof=>0};}
sub send_text{
 my($c,$text)=@_;my $offset=0;my $deadline=monotonic()+$c->{timeout};my $sel=IO::Select->new($c->{socket});
 while($offset<length($text)){my $left=$deadline-monotonic();$left>0 && $sel->can_write($left) or die "socket write timeout\n";my $n=syswrite($c->{socket},$text,length($text)-$offset,$offset);if(!defined $n){next if $!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK};die "socket write: $!\n";}$n>0 or die "socket write returned zero\n";$offset+=$n;}
}
sub read_line{
 my($c,$limit)=@_;my $deadline=monotonic()+$c->{timeout};my $sel=IO::Select->new($c->{socket});
 while(1){if(defined($limit)){my $nl=index($c->{buffer},"\n");die "oversized socket response\n" if ($nl>=0?$nl+1>=$limit:length($c->{buffer})>=$limit);}if($c->{buffer}=~s/\A([^\n]*\n)//){return $1;}if($c->{eof}){return undef unless length($c->{buffer});my $last=$c->{buffer};$c->{buffer}='';return $last;}my $left=$deadline-monotonic();$left>0 && $sel->can_read($left) or die "socket read timeout\n";my $n=sysread($c->{socket},my $chunk,65536);if(!defined $n){next if $!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK};die "socket read: $!\n";}if(!$n){$c->{eof}=1;next;}$c->{buffer}.=$chunk;}
}
sub close_socket{my($c)=@_;close $c->{socket} or die "socket close: $!\n";}
1;
