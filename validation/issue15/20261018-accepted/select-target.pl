#!/usr/bin/env perl
use strict; use warnings;
BEGIN {
    die "Frozen source directory required\n" unless defined($ARGV[0])
        && $ARGV[0] =~ m{\A/home/nate/claude/flash-sass-resolve/timing-[0-9a-f]{40}\z};
    unshift @INC, "$ARGV[0]/validation/issue15";
}
use EngineGuard qw(borrow_lock process_start);
use IO::Socket::UNIX; use IO::Handle; use Socket qw(SOCK_STREAM SOL_SOCKET SO_PEERCRED);
use JSON::PP; use Digest::SHA; use POSIX qw(strftime);
my ($source,$revision,$out,$layout,$fd)=@ARGV;
die "source revision output layout lease-fd required\n" unless defined($fd) && @ARGV==5;
die "Invalid source SHA\n" unless $revision=~/\A[0-9a-f]{40}\z/;
my ($lease,$error)=borrow_lock($fd,'/tmp/flash-sass-engine-validation.lock');
die "$error\n" unless $lease;
my $selection={utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),source_sha=>$revision,
    shared_lock=>'/tmp/flash-sass-engine-validation.lock'};
my @argv;
my $socket='/tmp/flash.sock';
my $s=-S $socket ? IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$socket,Timeout=>5) : undef;
if ($s) {
    my $cred=getsockopt($s,SOL_SOCKET,SO_PEERCRED);
    die "Missing resident peer credentials\n" unless defined($cred) && length($cred)==length(pack('i!I!I!',0,0,0));
    my ($pid,$uid)=unpack('i!I!I!',$cred);
    die "Resident socket owner differs\n" unless $uid==$> && $pid>1;
    my ($start,$start_error)=process_start($pid);die "$start_error\n" unless defined($start);
    close $s;
    open my $f,'<:raw',"/proc/$pid/maps" or die "Read mapped engine: $!";
    my %images;
    while (<$f>) { my @v=split /\s+/; $images{$v[5]}=1 if defined($v[5]) && $v[5]=~m{\A/tmp/flashcore-\Q$pid\E-gen[0-9]+\.so\z}; }
    close $f; die "Missing or ambiguous resident engine image\n" unless keys(%images)==1;
    my ($image)=keys %images;
    open $f,'<:raw',$image or die "Read resident image: $!";
    my $h=Digest::SHA->new(256);$h->addfile($f);close $f;my $sha=$h->hexdigest;
    @$selection{qw(mode resident_pid resident_start_tick baseline_engine_sha256 private_image)}
        =('resident',$pid,$start,$sha,$image);
    @argv=('perl',"$source/validation/issue15/run_resident.pl",'--source',$source,'--source-sha',$revision,
        '--out',"$out/evidence",'--resident-pid',$pid,'--resident-start',$start,
        '--baseline-engine-sha256',$sha,'--lease-fd',fileno($lease));
} else {
    $selection->{mode}='owned-ipc';
    @argv=('perl',"$source/validation/issue15/run_target.pl",'--source',$source,'--source-sha',$revision,
        '--out',"$out/evidence",'--ipc-layout',$layout,'--lease-fd',fileno($lease));
}
$selection->{argv}=\@argv;
open my $f,'>:raw',"$out/selection.json" or die "Write selection: $!";
print {$f} JSON::PP->new->canonical->pretty->encode($selection) or die "Write selection: $!";
close $f or die "Close selection: $!";
print "$selection->{utc} SELECTED $selection->{mode}\n";
STDOUT->flush();
exec {$argv[0]} @argv or die "Exec selected validation: $!";
