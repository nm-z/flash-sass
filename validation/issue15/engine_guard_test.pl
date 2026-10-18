#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use EngineGuard qw(take_lock idle_reason);
use File::Temp qw(tempdir);
use Fcntl qw(F_GETFD FD_CLOEXEC);
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
done_testing();
