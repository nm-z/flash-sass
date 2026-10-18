package EngineGuard;
use strict;
use warnings;
use Exporter 'import';
use Fcntl qw(:DEFAULT :flock F_GETFD F_SETFD FD_CLOEXEC);
our @EXPORT_OK = qw(take_lock idle_reason);

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
