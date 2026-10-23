#!/usr/bin/env perl
use strict; use warnings; use IO::Socket::UNIX; use Socket qw(SOCK_STREAM); use JSON::PP; use Time::HiRes qw(time);
my ($sock,$report)=@ARGV; die "socket and fresh report required\n" unless $sock && $report && !-e $report && $sock ne '/tmp/flash.sock';
my @cases; my $result={status=>'RUNNING',socket=>$sock,cases=>\@cases};
sub save { open my $f,'>',$report or die $!; print {$f} JSON::PP->new->canonical->pretty->encode($result); close $f or die $!; }
$SIG{ALRM}=sub{die "request timeout\n"}; $SIG{PIPE}='IGNORE'; $|=1;
eval { for my $n (4,6,8,32) { alarm 90; my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$sock) or die "connect: $!"; $s->autoflush(1); my @ids=map {20+($_%100)} 0..$n-1; my $cmd='chat 3 '.join(' ',@ids); my $start=time; print {$s} "$cmd\n" or die $!; my @raw; my @tokens; my $done;
while(my $line=<$s>){ chomp $line; push @raw,$line; die "$line\n" if $line =~ /^error/; push @tokens,0+$1 if $line =~ /^tok (\d+)/; if($line =~ /^done /){ $done=$line;last; } }
die "no done" unless $done; my @d=split ' ',$done; die "wrong generated count" unless @tokens==4 && $d[1]==4 && $d[2]==0 && $d[3]==$n; close $s; push @cases,{prompt=>$n,command=>$cmd,seconds=>time-$start,tokens=>\@tokens,raw=>\@raw,status=>'PASS'}; save(); print "PASS paired prompt $n\n"; alarm 0;
} $result->{status}='PASS'; save(); };
if($@){ $result->{status}='FAIL';$result->{error}=$@;save();die $@; }
