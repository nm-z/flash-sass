#!/usr/bin/env perl
# Reproduce issue #15 on archy's existing six-die IPC weight layout.
# Owns only the cores/frontends it starts; never reloads or stops /tmp/flash.sock.
use strict;use warnings;
use Getopt::Long qw(GetOptions);
use Cwd qw(abs_path);
use File::Path qw(make_path);
use FindBin;
use lib $FindBin::Bin;
use EngineGuard qw(take_lock idle_reason);
use IO::Socket::UNIX;use IO::Select;use IO::Handle;
use Socket qw(SOCK_STREAM);
use POSIX qw(WNOHANG _exit strftime);
use Time::HiRes qw(time sleep);
use JSON::PP;use Digest::SHA;
my($source,$out,$revision,$preflight_only);
GetOptions('source=s'=>\$source,'out=s'=>\$out,'source-sha=s'=>\$revision,'preflight-only'=>\$preflight_only) or die "Invalid options\n";
defined($_)&&length($_) or die "Usage: $0 --source BUILT_SOURCE --out NEW_DIRECTORY --source-sha FULL_COMMIT\n" for($source,$out,$revision);
die "Full source SHA required\n" unless $revision=~/\A[0-9a-f]{40}\z/;
$source=abs_path($source)//die "Missing source\n";
die "Evidence directory already exists\n" if -e $out;
make_path($out);$out=abs_path($out);STDOUT->autoflush(1);
my $pretty=JSON::PP->new->canonical->pretty;
my $state={status=>'RUNNING',source_sha=>$revision,source=>$source,commands=>[],cores=>[],
    started_utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime)};
sub save {my($name,$s)=@_;open my $f,'>:raw',"$out/$name" or die "$name: $!";print {$f} $s or die "write $name: $!";close $f or die "close $name: $!";}
sub persist {save('results.json.tmp',$pretty->encode($state));rename "$out/results.json.tmp","$out/results.json" or die "rename: $!";}
sub sha {open my $f,'<:raw',$_[0] or die "hash $_[0]: $!";my $s=Digest::SHA->new(256);$s->addfile($f);close $f;return $s->hexdigest;}
my %owned_children;
sub spawn {
    my($log,$argv)=@_;my $pid=fork();defined($pid) or die "fork: $!";
    if(!$pid){open STDIN,'<','/dev/null' or _exit(125);open STDOUT,'>:raw',"$out/$log" or _exit(126);open STDERR,'>&',\*STDOUT or _exit(126);chdir "$source/kern" or _exit(126);exec {$argv->[0]} @$argv or _exit(127);}
    $owned_children{$pid}=1;return $pid;
}
sub exit_code {return ($_[0]&127)?128+($_[0]&127):$_[0]>>8;}
sub stop_owned {
    my($pid)=@_;return if !defined($pid)||!delete($owned_children{$pid});
    return if waitpid($pid,WNOHANG)!=0;
    kill 'TERM',$pid;my $end=time()+5;
    while(time()<$end){return if waitpid($pid,WNOHANG)!=0;sleep .05;}
    kill 'KILL',$pid;waitpid($pid,0);
}
sub command {
    my($name,$timeout,@argv)=@_;my $start=time();my $pid=spawn("$name.log",\@argv);my($rc,$timed_out);
    while(1){if(waitpid($pid,WNOHANG)==$pid){$rc=exit_code($?);delete $owned_children{$pid};last;}if(time()-$start>$timeout){$timed_out=1;stop_owned($pid);$rc=124;last;}sleep .05;}
    push @{$state->{commands}},{name=>$name,argv=>\@argv,exit_code=>$rc,timed_out=>$timed_out?JSON::PP::true:JSON::PP::false,seconds=>time()-$start};persist();
    print "$name exit=$rc\n";die "$name failed (exit $rc)\n" if $rc;
}
sub read_log {open my $f,'<:raw',"$out/$_[0].log" or die "read $_[0]: $!";local $/;my $s=<$f>;close $f;return $s;}
sub ensure_idle {
    my($name)=@_;
    my $ok=eval {
        command("$name-status",10,'nvidia-smi');
        command("$name-gpus",10,'nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm','--format=csv');
        command("$name-compute",10,'nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader,nounits');
        command("$name-processes",10,'ps','-eo','pid=,ppid=,comm=,args=');
        my $reason=idle_reason(read_log("$name-status"),read_log("$name-compute"),read_log("$name-processes"));
        die "$reason\n" if defined $reason;
        1;
    };
    if(!$ok){$state->{status}='BLOCKED';die "GPU preflight blocked: $@";}
}
sub query {
    my($socket)=@_;my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$socket) or return;
    $s->autoflush(1);print {$s} "config\n" or return;my $sel=IO::Select->new($s);return unless $sel->can_read(1);
    my $n=sysread($s,my $b,8192);close $s;return defined($n)&&$n>0?$b:undef;
}
my ($owned_pid,$lock);
$SIG{PIPE}='IGNORE';$SIG{TERM}=$SIG{INT}=$SIG{HUP}=sub{die "Interrupted\n"};
persist();
my $ok=eval {
    open my $host,'-|','hostname' or die "hostname: $!";my $hostname=<$host>;close $host;chomp $hostname;
    die "This recorded hardware layout requires archy\n" unless $hostname eq 'archy';$state->{hostname}=$hostname;
    $state->{shared_lock}='/tmp/flash-sass-engine-validation.lock';
    my $lock_error;($lock,$lock_error)=take_lock($state->{shared_lock});
    if(!$lock){$state->{status}='BLOCKED';die "$lock_error\n";}
    $state->{lock_acquired_utc}=strftime('%Y-%m-%dT%H:%M:%SZ',gmtime);persist();
    ensure_idle('preflight');
    if($preflight_only){$state->{status}='PREFLIGHT_ONLY';}
    else {
    delete $ENV{$_} for grep {/\A(?:FLASH_|MODEL_)/} keys %ENV;
    my %env=(CUDA_VISIBLE_DEVICES=>'0,1,2,3,4,5,7',CUDA_DEVICE_ORDER=>'PCI_BUS_ID',
        SPLIT_SHARES=>'112,97,81,116,97,97',ATTN_DIE=>'1',ATTN_MAP=>'1,1,1,1,4,4,4,4,5,5,5,5',
        FLASH_ANCHOR_DIE=>'7',FLASH_MAXT=>'512',FLASH_ENGINE_MAXT=>'512',FLASH_MAXIMG=>'1',
        FLASH_PLE_DISK=>'1',FLASH_PLE_OFF=>'192',FLASH_NBLK=>'512',FLASH_KV_HOST=>'0',FLASH_WATCHDOG_S=>'60',
        FLASH_TRACE=>'4',FLASH_CUBIN=>"$source/kern/die_trace.cubin");
    @ENV{keys %env}=values %env;$state->{environment}=\%env;
    $state->{runner_sha256}=sha(abs_path($0));
    $state->{artifacts}={map {$_=>sha("$source/kern/$_")} qw(flashcore libengine.so flashd die.cubin die_trace.cubin)};
    command('session-fixtures',30,"$source/kern/trace_session_test");
    command('output-token-regression',30,"$source/kern/output_tokens_test",'/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf');
    for my $paired(0,1) {
        ensure_idle("before-pair$paired");
        my $socket="/tmp/flash-owned-issue15-".substr($revision,0,10)."-$paired-$$.sock";
        die "Owned socket already exists\n" if -e $socket;
        $ENV{FLASH_SOCK}=$socket;$ENV{FLASH_TWO_POSITION}="$paired";
        my @argv=("$source/kern/flashcore",'/home/nate/flash-ple/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf',"$source/kern/libengine.so",map {"/tmp/hold-h$_.manifest"} 0..5);
        $owned_pid=spawn("core-pair$paired.log",\@argv);
        my $record={pid=>$owned_pid,paired=>$paired,socket=>$socket,argv=>\@argv};push @{$state->{cores}},$record;persist();
        my $end=time()+90;
        while(1) {
            if(waitpid($owned_pid,WNOHANG)==$owned_pid){$record->{exit_code}=exit_code($?);delete $owned_children{$owned_pid};$owned_pid=undef;die "Owned core exited at startup (exit $record->{exit_code}); see core-pair$paired.log\n";}
            my $config=query($socket);
            if(defined($config)&&$config=~/^config /){$record->{config}=$config;persist();last;}
            die "Owned core startup timeout\n" if time()>$end;sleep .1;
        }
        command("cli-pair$paired",600,'perl',"$source/kern/timings_core_check.pl",'--socket',$socket,'--out',"$out/cli-pair$paired",'--source-sha',$revision);
        command("http-pair$paired",600,'perl',"$source/kern/timings_http_real.pl",'--flashd',"$source/kern/flashd",'--gguf','/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf','--socket',$socket,'--out',"$out/http-pair$paired",'--source-sha',$revision);
        stop_owned($owned_pid);$owned_pid=undef;$record->{stopped}=JSON::PP::true;persist();
    }
    command('gpu-after',10,'nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm','--format=csv');
    $state->{status}='PASS';
    }
    1;
};
my $failure=$@;stop_owned($_) for keys %owned_children;
if(!$ok){
    $state->{status}='FAIL' unless $state->{status} eq 'BLOCKED';$state->{error}="$failure";
    eval {command('gpu-after-failure',10,'nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm','--format=csv')}
        if $lock && $state->{status} ne 'BLOCKED';
}
if($lock){close $lock or die "release shared lock: $!";$state->{lock_released_utc}=strftime('%Y-%m-%dT%H:%M:%SZ',gmtime);}
$state->{finished_utc}=strftime('%Y-%m-%dT%H:%M:%SZ',gmtime);persist();
if(!$ok){warn $failure;exit($state->{status} eq 'BLOCKED'?75:1);}
print $preflight_only?"PREFLIGHT ONLY: no target request was executed\n":"PASS target CLI/HTTP timing validation\n";
