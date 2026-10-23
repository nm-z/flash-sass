#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use EngineGuard qw(take_lock idle_reason borrow_lock process_start verify_peer);
use File::Temp qw(tempdir);
use Fcntl qw(:DEFAULT :flock F_GETFD FD_CLOEXEC);
use Socket qw(AF_UNIX SOCK_STREAM);
use Test::More;

my $dir = tempdir(CLEANUP => 1);
my $path = "$dir/shared.lock";
my ($first, $error) = take_lock($path);
ok($first && !defined($error), 'first validator owns the shared lock');
my $inode = (stat $path)[1];
my ($second, $busy) = take_lock($path);
ok(!$second && $busy =~ /unavailable/, 'second validator is blocked');
ok(!(fcntl($first, F_GETFD, 0) & FD_CLOEXEC), 'owned exec children retain the lock');

pipe(my $read, my $write) or die "pipe: $!";
my $pid = fork();
defined($pid) or die "fork: $!";
if (!$pid) {
    close $write;
    sysread($read, my $signal, 1) == 1 or die "parent signal";
    close $first;
    exit 0;
}
close $read;
close $first;
($second, $busy) = take_lock($path);
ok(!$second, 'parent close does not release an owned child lock');
syswrite($write, "x") == 1 or die "signal child";
close $write;
waitpid($pid, 0) == $pid or die "waitpid";
is($?, 0, 'owned child exited');
($second, $error) = take_lock($path);
ok($second && !defined($error), 'lock becomes available after the child exits');
close $second;
is((stat $path)[1], $inode, 'release preserves the lock inode');
symlink($path, "$dir/link.lock") or die "symlink: $!";
my ($linked, $link_error) = take_lock("$dir/link.lock");
ok(!$linked && $link_error, 'unexpected symlink cannot create a second lock');

my ($owner, $owner_error) = take_lock($path);
BAIL_OUT($owner_error // 'lease owner missing') unless $owner;
my ($borrowed, $borrow_error) = borrow_lock(fileno($owner), $path);
ok($borrowed && !defined($borrow_error), 'inherited open-file description is accepted');
sysopen(my $independent, $path, O_RDWR) or die "open independent: $!";
my ($bad, $bad_error) = borrow_lock(fileno($independent), $path);
ok(!$bad, 'independent descriptor to the same inode is rejected');
like($bad_error, qr/not the supervisor's locked description/, 'open-description mismatch is explicit');
close $independent;
sysopen(my $other, "$dir/other.lock", O_RDWR | O_CREAT | O_EXCL, 0600) or die $!;
($bad, $bad_error) = borrow_lock(fileno($other), $path);
ok(!$bad && $bad_error =~ /different inode/, 'different lock inode is rejected');
close $other;
($bad, $bad_error) = borrow_lock(fileno($owner), "$dir/link.lock");
ok(!$bad && $bad_error =~ /symlink/, 'borrowed lock rejects a symlink path');
($bad, $bad_error) = borrow_lock(0, $path);
ok(!$bad && $bad_error =~ /Invalid/, 'standard input cannot masquerade as a borrowed lease');
close $borrowed if $borrowed;
my ($contender, $contender_error) = take_lock($path);
ok(!$contender, 'closing a borrowed handle preserves the supervisor lock');
close $owner;
sysopen(my $unlocked, $path, O_RDWR) or die $!;
($bad, $bad_error) = borrow_lock(fileno($unlocked), $path);
ok(!$bad && $bad_error =~ /not already held/, 'an unlocked descriptor is not an inherited lease');
close $unlocked;
($contender, $contender_error) = take_lock($path);
ok($contender, 'lease is available after owner cleanup');
close $contender;

socketpair(my $left, my $right, AF_UNIX, SOCK_STREAM, 0) or die "socketpair: $!";
my ($start, $start_error) = process_start($$);
BAIL_OUT($start_error // 'process start missing') unless defined($start);
my ($peer_ok, $peer_error) = verify_peer($left, $$, $start);
ok($peer_ok && !defined($peer_error), 'actual Unix peer PID, UID and start tick match');
($peer_ok, $peer_error) = verify_peer($left, $$ + 1, $start);
ok(!$peer_ok && $peer_error =~ /Unexpected peer PID/, 'wrong socket peer is rejected');
($peer_ok, $peer_error) = verify_peer($left, $$, $start + 1);
ok(!$peer_ok && $peer_error =~ /start tick changed/, 'reused PID with different start tick is rejected');
close $left; close $right;

my $status = "NVIDIA-SMI fixture\nProcesses:\n";
my $ps = " 100 1 init /sbin/init\n 101 100 perl fixture.pl\n";
ok(!defined(idle_reason($status, '', $ps)), 'successful empty compute query and idle ps permit the next step');
like(idle_reason($status, "974018, ./hold, GPU-fixture\n", $ps),
    qr/Existing GPU compute process/, 'a resident weight holder blocks even at zero utilization');
like(idle_reason($status, '', "$ps 222 1 flashcore ./flashcore\n"),
    qr/Existing worker.*222/, 'ps-only core blocks a launch');
like(idle_reason($status, '', "$ps 223 1 qsa_test ./qsa_test\n"),
    qr/Existing worker.*223/, 'a kernel harness not yet reported by NVML blocks a launch');
like(idle_reason($status, "driver unavailable\n", $ps),
    qr/Unrecognized compute/, 'diagnostic text cannot masquerade as an empty query');
like(idle_reason('driver unavailable', '', $ps),
    qr/Unrecognized nvidia/, 'missing GPU status blocks a launch');
like(idle_reason($status, '', "permission denied\n"),
    qr/Unrecognized ps/, 'invalid process snapshot blocks a launch');
like(idle_reason($status, '', ''),
    qr/Empty ps/, 'missing process snapshot blocks a launch');
my %ipc=(222=>'hold',224=>'fence',225=>'flashd');
my $ipc_ps="$ps 222 1 hold ./hold model 0 /tmp/hold-h0.manifest\n 224 1 fence ./fence 2\n 225 1 flashd ./flashd model /tmp/flash.sock\n";
my $ipc_apps="222, ./hold, GPU-abcd-1234\n224, ./fence, GPU-abcd-5678\n";
ok(!defined(idle_reason($status,$ipc_apps,$ipc_ps,\%ipc)),
    'explicit identified IPC holders, memory fence and separate frontend permit an owned core');
like(idle_reason($status,$ipc_apps,$ipc_ps),qr/Existing GPU compute process/,
    'default policy still rejects the same resident allocations');
like(idle_reason($status,$ipc_apps,"$ipc_ps 226 1 flashcore ./flashcore\n",\%ipc),
    qr/Existing worker.*226/, 'another core blocks explicit IPC mode');
like(idle_reason($status,"$ipc_apps 226, ./flashcore, GPU-abcd-1234\n",$ipc_ps,\%ipc),
    qr/Existing GPU compute process.*226/, 'another GPU worker blocks explicit IPC mode');
like(idle_reason($status,"222, ./flashcore, GPU-abcd-1234\n",$ipc_ps,\%ipc),
    qr/Existing GPU compute process/, 'approved PID cannot change GPU process type');
like(idle_reason($status,$ipc_apps,"$ps 222 1 flashcore ./flashcore\n",\%ipc),
    qr/identity changed/, 'approved PID cannot change ps process type');
like(idle_reason($status,$ipc_apps,$ps,\%ipc),qr/IPC process missing/,
    'missing approved resident state blocks a launch');
like(idle_reason($status,'',$ps,{226=>'flashcore'}),qr/Invalid resident IPC process policy/,
    'policy cannot permit a competing core');
done_testing();
