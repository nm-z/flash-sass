package EngineGuard;
use strict;
use warnings;
use Exporter 'import';
use Fcntl qw(:DEFAULT :flock F_GETFD F_SETFD FD_CLOEXEC S_ISREG);
use Errno qw(EAGAIN EWOULDBLOCK);
use Socket qw(SOL_SOCKET SO_PEERCRED);
our @EXPORT_OK = qw(take_lock idle_reason borrow_lock process_start verify_peer);

# Do not unlink this file: all validators must lock the same persistent inode.
# Children inherit the descriptor, retaining the lock if the supervisor dies.
sub take_lock {
    my ($path) = @_;
    sysopen(my $lock, $path, O_RDWR | O_CREAT | O_NOFOLLOW, 0600)
        or return (undef, "Cannot open shared lock $path: $!");
    unless (-f $lock) {
        close $lock;
        return (undef, "Shared lock is not a regular file");
    }
    unless (flock($lock, LOCK_EX | LOCK_NB)) {
        my $reason = "Shared engine validation lock is unavailable: $!";
        close $lock;
        return (undef, $reason);
    }
    my $flags = fcntl($lock, F_GETFD, 0);
    unless (defined($flags) && defined(fcntl($lock, F_SETFD, $flags & ~FD_CLOEXEC))) {
        close $lock;
        return (undef, "Cannot retain the shared lock in owned children");
    }
    return ($lock, undef);
}

# Borrow a supervisor's already-locked open-file description. Merely opening
# the same pathname is insufficient. Closing the returned handle is safe;
# LOCK_UN would also unlock the supervisor and must never be used here.
sub borrow_lock {
    my ($fd, $path) = @_;
    return (undef, 'Invalid inherited lock descriptor')
        unless defined($fd) && "$fd" =~ /\A[0-9]+\z/ && $fd >= 3;
    return (undef, 'Missing shared lock path') unless defined($path) && length($path);
    my ($held, $probe);
    my $reject = sub {
        my ($error) = @_;
        close($probe) if defined($probe) && defined(fileno($probe));
        close($held) if defined($held) && defined(fileno($held));
        return (undef, $error);
    };
    my $same = sub {my ($a,$b)=@_; return @$a && @$b && $a->[0]==$b->[0] && $a->[1]==$b->[1];};
    open($held, '+<&', 0 + $fd) or return (undef, "Cannot duplicate lock descriptor $fd: $!");
    my @held_stat = stat($held);
    my @path_stat = lstat($path);
    return $reject->('Inherited lock is not a regular file')
        unless @held_stat && S_ISREG($held_stat[2]);
    return $reject->('Shared lock path is missing, nonregular, or a symlink')
        unless @path_stat && S_ISREG($path_stat[2]);
    return $reject->('Inherited lock descriptor names a different inode')
        unless $same->(\@held_stat, \@path_stat);
    sysopen($probe, $path, O_RDWR | O_NOFOLLOW)
        or return $reject->("Cannot open independent lock probe: $!");
    my @probe_stat = stat($probe);
    my @current_stat = lstat($path);
    return $reject->('Shared lock path changed during verification')
        unless @current_stat && S_ISREG($current_stat[2]) && @probe_stat && S_ISREG($probe_stat[2])
            && $same->(\@held_stat, \@probe_stat) && $same->(\@held_stat, \@current_stat);
    if (flock($probe, LOCK_EX | LOCK_NB)) {
        return $reject->('Shared lock was not already held');
    }
    my ($number, $message) = (0 + $!, "$!");
    return $reject->("Independent lock probe failed unexpectedly: $message")
        unless $number == EWOULDBLOCK || $number == EAGAIN;
    close($probe) or return $reject->("Cannot close independent lock probe: $!");
    undef $probe;
    flock($held, LOCK_EX | LOCK_NB)
        or return $reject->("Descriptor is not the supervisor's locked description: $!");
    my $flags = fcntl($held, F_GETFD, 0);
    return $reject->("Cannot retain borrowed descriptor: $!")
        unless defined($flags) && defined(fcntl($held, F_SETFD, $flags & ~FD_CLOEXEC));
    @current_stat = lstat($path);
    return $reject->('Shared lock path changed after verification')
        unless @current_stat && S_ISREG($current_stat[2]) && $same->(\@held_stat, \@current_stat);
    return ($held, undef);
}

sub process_start {
    my ($pid) = @_;
    return (undef, 'Invalid PID') unless defined($pid) && "$pid" =~ /\A[1-9][0-9]*\z/;
    open(my $fh, '<', "/proc/$pid/stat") or return (undef, "Cannot read process $pid: $!");
    my $text = do {local $/; <$fh>}; close($fh);
    return (undef, "Empty process stat for $pid") unless defined($text);
    my ($tail) = $text =~ /\A\Q$pid\E \(.*\) (.*)\z/s;
    return (undef, "Malformed process stat for $pid") unless defined($tail);
    my @fields = split(/\s+/, $tail);
    return (undef, "Missing process start tick for $pid")
        unless @fields >= 20 && $fields[19] =~ /\A[0-9]+\z/;
    return ($fields[19], undef);
}

sub verify_peer {
    my ($socket, $pid, $expected_start) = @_;
    return (undef, 'Invalid expected peer PID') unless defined($pid) && "$pid" =~ /\A[1-9][0-9]*\z/;
    return (undef, 'Missing expected peer start tick')
        unless defined($expected_start) && "$expected_start" =~ /\A[0-9]+\z/;
    my $cred = getsockopt($socket, SOL_SOCKET, SO_PEERCRED);
    return (undef, "Cannot read SO_PEERCRED: $!") unless defined($cred);
    return (undef, 'Unexpected SO_PEERCRED size') unless length($cred)==length(pack('i!I!I!',0,0,0));
    my ($peer_pid, $peer_uid) = unpack('i!I!I!', $cred);
    return (undef, "Unexpected peer PID $peer_pid; expected $pid") unless $peer_pid==$pid;
    return (undef, "Unexpected peer UID $peer_uid") unless $peer_uid==$>;
    my ($start, $error) = process_start($pid);
    return (undef, $error) unless defined($start);
    return (undef, "Peer PID $pid start tick changed") unless $start eq "$expected_start";
    return (1, undef);
}

# Inputs are complete outputs from successful commands. Failed commands are
# rejected by the runner before reaching this parser; unfamiliar output blocks.
sub idle_reason {
    my ($status, $apps, $processes) = @_;
    return "Unrecognized nvidia-smi status"
        unless $status =~ /NVIDIA-SMI/ && $status =~ /Processes:/;
    for my $line (split /\n/, $apps) {
        next if $line =~ /\A\s*\z/;
        return "Existing GPU compute process: $line"
            if $line =~ /\A\s*[1-9][0-9]*\s*,/;
        return "Unrecognized compute-process query output: $line";
    }
    my $rows = 0;
    for my $line (split /\n/, $processes) {
        next if $line =~ /\A\s*\z/;
        return "Unrecognized ps output"
            unless $line =~ /\A\s*([0-9]+)\s+([0-9]+)\s+(\S+)\s+(.*)\z/;
        my ($pid, $name) = ($1, $3);
        $rows++;
        return "Existing worker in ps: PID $pid ($name)"
            if $name =~ /\A(?:flashcore|flashd|hold|fence|memtest|.+_(?:test|bench))\z/;
    }
    return "Empty ps snapshot" unless $rows;
    return undef;
}
1;
