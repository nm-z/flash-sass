#!/usr/bin/env perl
use strict;
use warnings;

# Exercise the production die watchdog without touching the shared flashd core.
# Start a private flashcore on the resident holders, begin a long prefill, then
# stop only that core's host process after its CUDA kernel is active. The kernel
# runs into the PLE ready-flag wait, expires at its 300-second launch limit, and
# the resumed host must report ctl[11] and rebuild its contexts.
#
# Usage (run only inside `ssh archy gpuq run -- ...`):
#   perl validation/issue83/timeout_recovery.pl FLASHCORE PLE_GGUF LIBENGINE CUBIN MANIFEST0 ... MANIFEST5

use Fcntl qw(F_GETFL F_SETFL O_NONBLOCK);
use IO::Select;
use IO::Socket::UNIX;
use POSIX qw(:sys_wait_h);
use Socket qw(SOCK_STREAM);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC sleep);

$| = 1;
sub mono { clock_gettime(CLOCK_MONOTONIC) }
sub say_event { print STDOUT "issue83: $_[0]\n"; }
sub check { die "issue83: $_[1]\n" unless $_[0]; }

my ($core_bin, $ple_file, $engine_lib, $cubin, @manifest) = @ARGV;
check(defined($cubin) && @manifest == 6,
      'expected flashcore, PLE GGUF, libengine.so, die.cubin and six manifests');
for my $path ($core_bin, $ple_file, $engine_lib, $cubin, @manifest) {
    check(defined($path) && length($path), 'empty input path');
}
check(-x $core_bin, "flashcore is not executable: $core_bin");
check(-r $ple_file, "PLE GGUF is not readable: $ple_file");
check(-r $engine_lib, "engine library is not readable: $engine_lib");
check(-r $cubin, "die cubin is not readable: $cubin");
for my $path (@manifest) { check(-r $path, "holder manifest is not readable: $path"); }

my $core_pid;
my $core_reaped = 0;
my $core_status;
my $core_stopped = 0;
my $socket_path = "/tmp/issue83-watchdog-$$.sock";
my $log_read;
my $log_select;
my $log_buffer = '';
my $core_output = '';

sub drain_core_output {
    return unless $log_read;
    while (1) {
        my $chunk = '';
        my $n = sysread($log_read, $chunk, 8192);
        if (!defined $n) {
            last if $!{EAGAIN} || $!{EWOULDBLOCK};
            next if $!{EINTR};
            die "issue83: read core output: $!\n";
        }
        if ($n == 0) {
            $log_select->remove($log_read) if $log_select;
            last;
        }
        $core_output .= $chunk;
        $log_buffer .= $chunk;
        while ($log_buffer =~ s/\A([^\n]*\n)//) {
            print STDOUT "core: $1";
        }
    }
}

sub reap_core {
    return 1 if $core_reaped;
    return 0 unless $core_pid;
    my $waited = waitpid($core_pid, WNOHANG);
    if ($waited == $core_pid) {
        $core_reaped = 1;
        $core_status = $?;
        drain_core_output();
        return 1;
    }
    return 0;
}

sub connect_core {
    return IO::Socket::UNIX->new(Type => SOCK_STREAM, Peer => $socket_path);
}

sub read_line_until {
    my ($fh, $seconds) = @_;
    my $sel = IO::Select->new($fh);
    my $deadline = mono() + $seconds;
    my $buf = '';
    while (mono() < $deadline) {
        drain_core_output();
        check(!reap_core(), 'flashcore exited while the test was waiting');
        my @ready = $sel->can_read(0.1);
        next unless @ready;
        my $chunk = '';
        my $n = sysread($fh, $chunk, 4096);
        next if !defined($n) && ($!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK});
        check(defined($n), "read core socket: $!");
        return undef if $n == 0 && !length($buf);
        last if $n == 0;
        $buf .= $chunk;
        my $newline = index($buf, "\n");
        return substr($buf, 0, $newline + 1) if $newline >= 0;
    }
    die "issue83: timed out waiting for a core socket line\n";
}

sub pmon_sm_for_core {
    open my $fh, '-|', 'nvidia-smi', 'pmon', '-c', '1', '-s', 'u'
        or die "issue83: start nvidia-smi pmon: $!\n";
    my @lines = <$fh>;
    close $fh or die "issue83: nvidia-smi pmon failed\n";
    for my $line (@lines) {
        next unless $line =~ /^\s*\d+\s+\Q$core_pid\E\s+\S+\s+(\d+|N\/A|-)\b/;
        return $1 =~ /^\d+$/ ? 0 + $1 : 0;
    }
    return undef;
}

sub read_socket_to_eof {
    my ($fh, $seconds) = @_;
    my $sel = IO::Select->new($fh);
    my $deadline = mono() + $seconds;
    my $data = '';
    while (mono() < $deadline) {
        drain_core_output();
        my @ready = $sel->can_read(0.1);
        if (!@ready) {
            check(!reap_core(), 'flashcore exited before replying to the test client');
            next;
        }
        my $chunk = '';
        my $n = sysread($fh, $chunk, 8192);
        next if !defined($n) && ($!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK});
        check(defined($n), "read test response: $!");
        last if $n == 0;
        $data .= $chunk;
        print STDOUT "client: $chunk";
    }
    check(mono() < $deadline, 'timed out waiting for context recovery response');
    return $data;
}

sub stop_core_gracefully {
    return unless $core_pid && !$core_reaped;
    if ($core_stopped) { kill 'CONT', $core_pid; $core_stopped = 0; }
    if (-S $socket_path) {
        eval {
            my $quit = connect_core();
            if ($quit) { $quit->autoflush(1); print {$quit} "quit\n"; close $quit; }
        };
    }
    my $deadline = mono() + 15;
    while (mono() < $deadline && !reap_core()) { sleep(0.1); }
    if (!$core_reaped) {
        kill 'TERM', $core_pid;
        $deadline = mono() + 5;
        while (mono() < $deadline && !reap_core()) { sleep(0.1); }
    }
    if (!$core_reaped) {
        kill 'KILL', $core_pid;
        waitpid($core_pid, 0);
        $core_reaped = 1;
    }
}

$SIG{INT} = sub { die "issue83: interrupted\n"; };
$SIG{TERM} = sub { die "issue83: terminated\n"; };
$SIG{HUP} = sub { die "issue83: hung up\n"; };

END {
    stop_core_gracefully() if $core_pid && !$core_reaped;
    unlink $socket_path if defined($socket_path) && -S $socket_path;
}

my $log_write;
pipe($log_read, $log_write) or die "issue83: create core log pipe: $!\n";
my $flags = fcntl($log_read, F_GETFL, 0);
defined($flags) or die "issue83: read core log flags: $!\n";
fcntl($log_read, F_SETFL, $flags | O_NONBLOCK)
    or die "issue83: make core log pipe nonblocking: $!\n";
$log_select = IO::Select->new($log_read);

$core_pid = fork();
defined($core_pid) or die "issue83: fork flashcore: $!\n";
if ($core_pid == 0) {
    close $log_read;
    open STDIN, '<', '/dev/null' or POSIX::_exit(126);
    open STDOUT, '>&', $log_write or POSIX::_exit(126);
    open STDERR, '>&', $log_write or POSIX::_exit(126);
    close $log_write;
    $ENV{CUDA_DEVICE_ORDER} = 'PCI_BUS_ID';
    $ENV{CUDA_VISIBLE_DEVICES} = '0,1,2,3,4,5,6';
    $ENV{SPLIT_SHARES} = '112,97,81,116,97,97';
    $ENV{ATTN_MAP} = '1,1,1,1,4,4,4,4,5,5,5,5';
    $ENV{ATTN_DIE} = '1';
    $ENV{FLASH_ANCHOR_DIE} = '6';
    $ENV{FLASH_MAXT} = '4096';
    $ENV{FLASH_ENGINE_MAXT} = '4096';
    $ENV{FLASH_MAXIMG} = '4096';
    $ENV{FLASH_KV_HOST} = '0';
    $ENV{FLASH_KV_TRACE} = '0';
    $ENV{FLASH_TRACE} = '0';
    $ENV{FLASH_ROPE_OWNER} = '-1';
    $ENV{FLASH_SNAPSHOT_HOST} = '0';
    $ENV{FLASH_PLE_DISK} = '1';
    $ENV{FLASH_PLE_OFF} = '192';
    $ENV{FLASH_PLE_READERS} = '1';
    $ENV{FLASH_TWO_POSITION} = '0';
    $ENV{FLASH_PREFILL_CHUNK} = '0';
    $ENV{FLASH_WATCHDOG_S} = '600';
    $ENV{FLASH_CUBIN} = $cubin;
    $ENV{FLASH_SOCK} = $socket_path;
    exec { $core_bin } $core_bin, $ple_file, $engine_lib, @manifest
        or do { print STDERR "issue83: exec $core_bin: $!\n"; POSIX::_exit(127); };
}
close $log_write;
say_event("started isolated flashcore pid=$core_pid socket=$socket_path");

my $startup_deadline = mono() + 180;
while (mono() < $startup_deadline && !-S $socket_path) {
    drain_core_output();
    check(!reap_core(), 'flashcore exited during initialization');
    sleep(0.1);
}
check(-S $socket_path, 'flashcore socket did not become ready within 180 seconds');

my $config_socket = connect_core();
check($config_socket, 'could not connect to the isolated flashcore');
$config_socket->autoflush(1);
print {$config_socket} "config\n" or die "issue83: send config: $!\n";
my $config = read_line_until($config_socket, 15);
check(defined($config) && $config =~ /^config /, 'flashcore did not return its config');
check(index($config, 'rev=270aa639dd9dc86eb16db698e318aaa7e7d3fd2c') >= 0,
      "unexpected engine revision: $config");
check(index($config, 'maxt=4096') >= 0, "unexpected engine capacity: $config");
check(index($config, 'ple_disk=1') >= 0, "engine is not using disk-backed PLE: $config");
check(index($config, "cubin=$cubin") >= 0, "engine did not load the requested cubin: $config");
say_event("isolated engine ready: $config");
close $config_socket;

my $client = connect_core();
check($client, 'could not connect to flashcore for the test request');
$client->autoflush(1);
print {$client} "timings 1\n" or die "issue83: enable timing records: $!\n";
my $timing_ack = read_line_until($client, 10);
check(defined($timing_ack) && $timing_ack =~ /^ok: timings 1/, 'engine did not enable timings');

my $prompt_positions = 2046;
my $request = 'gen 0' . (' 1' x $prompt_positions) . "\n";
print {$client} $request or die "issue83: send long prefill request: $!\n";
say_event("sent one $prompt_positions-position prefill; waiting for the production kernel to be active");

my $active_deadline = mono() + 180;
my $active_at;
my $last_sm;
while (mono() < $active_deadline) {
    drain_core_output();
    check(!reap_core(), 'flashcore exited before the test kernel became active');
    $last_sm = pmon_sm_for_core();
    if (defined($last_sm) && $last_sm > 0) { $active_at = mono(); last; }
}
check(defined($active_at), 'nvidia-smi never observed this flashcore running a kernel');
say_event(sprintf('nvidia-smi observed flashcore GPU activity (SM=%d); pausing only this core host', $last_sm));
kill 'STOP', $core_pid or die "issue83: SIGSTOP isolated flashcore: $!\n";
$core_stopped = 1;
my $stopped_at = mono();
my $last_progress = $stopped_at;
my $idle_samples = 0;
my $gpu_idle_at;
my $max_wait = 330;
while (mono() - $stopped_at < $max_wait) {
    my $elapsed = mono() - $stopped_at;
    my $sm = pmon_sm_for_core();
    if ($elapsed >= 298 && (!defined($sm) || $sm == 0)) { $idle_samples++; }
    else { $idle_samples = 0; }
    if ($idle_samples >= 3) { $gpu_idle_at = mono(); last; }
    if (mono() - $last_progress >= 60) {
        say_event(sprintf('host remains paused; GPU wait %.1f seconds, pmon SM=%s',
                          $elapsed, defined($sm) ? $sm : 'idle'));
        $last_progress = mono();
    }
    sleep(1);
}
check(defined($gpu_idle_at), 'GPU kernel did not retire within 330 seconds of the forced wait');
my $gpu_wait_seconds = $gpu_idle_at - $stopped_at;
check($gpu_wait_seconds >= 298 && $gpu_wait_seconds <= 330,
      sprintf('GPU wait duration %.1f seconds is outside the expected 300-second window', $gpu_wait_seconds));
say_event(sprintf('GPU kernel stopped after %.1f seconds while the host was paused; resuming host to inspect ctl[11]',
                  $gpu_wait_seconds));
kill 'CONT', $core_pid or die "issue83: SIGCONT isolated flashcore: $!\n";
$core_stopped = 0;

my $response = read_socket_to_eof($client, 120);
close $client;
check($response =~ /error: device fault; the core rebuilds the contexts and the engine/,
      'engine did not report the kernel time-limit failure');
check($response =~ /error_code\s*[:=]\s*-2/ || $response =~ /"error_code":-2/,
      'timing record did not classify the limit as a device fault');
check($response =~ /ok: contexts rebuilt, engine reloaded/,
      'flashcore did not confirm the context rebuild');

my $reloaded_socket = connect_core();
check($reloaded_socket, 'core did not accept a client after rebuilding its contexts');
$reloaded_socket->autoflush(1);
print {$reloaded_socket} "config\n" or die "issue83: query reloaded engine: $!\n";
my $reloaded_config = read_line_until($reloaded_socket, 15);
check(defined($reloaded_config)
      && index($reloaded_config, 'rev=270aa639dd9dc86eb16db698e318aaa7e7d3fd2c') >= 0
      && index($reloaded_config, "cubin=$cubin") >= 0,
      "rebuilt core did not reload the tested master engine: $reloaded_config");
close $reloaded_socket;

drain_core_output();
my $core_logs = $core_output . $log_buffer;
check($core_logs =~ /engine: die \d+ reached the kernel time limit/,
      'engine did not read ctl[11] after the kernel ended');
check($core_logs =~ /flashcore: device fault; rebuilding the die contexts and the engine/,
      'flashcore did not enter its context rebuild path');
say_event(sprintf('PASS: 300-second die watchdog, ctl[11] error and context rebuild observed; GPU wait %.1f seconds',
                  $gpu_wait_seconds));

stop_core_gracefully();
check($core_reaped && $core_status == 0, 'isolated flashcore did not exit cleanly after quit');
unlink $socket_path if -S $socket_path;
