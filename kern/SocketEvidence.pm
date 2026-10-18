package SocketEvidence;
use strict;
use warnings;
use Exporter 'import';
use IO::Socket::UNIX;
use IO::Select;
use Socket 'SOCK_STREAM';
use Fcntl qw(F_GETFL F_SETFL O_NONBLOCK);
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
our @EXPORT_OK = qw(connect_unix send_text read_line close_socket monotonic begin_request end_request);

sub monotonic { return clock_gettime(CLOCK_MONOTONIC); }
sub connect_unix {
    my ($path, $timeout) = @_;
    $timeout //= 120;
    $timeout > 0 or die "socket timeout must be positive\n";
    my $s = IO::Socket::UNIX->new(Type => SOCK_STREAM, Peer => $path, Timeout => $timeout)
        or die "connect $path: $!\n";
    binmode $s;
    my $flags = fcntl($s, F_GETFL, 0);
    defined($flags) or die "socket flags: $!\n";
    defined(fcntl($s, F_SETFL, $flags | O_NONBLOCK)) or die "socket nonblocking: $!\n";
    return {socket => $s, buffer => '', timeout => $timeout, eof => 0};
}
# One deadline covers sending and every response line, including buffered lines.
sub begin_request {
    my ($c, $seconds) = @_;
    $seconds //= $c->{timeout};
    $seconds > 0 or die "request timeout must be positive\n";
    return $c->{deadline} = monotonic() + $seconds;
}
sub end_request { delete $_[0]{deadline}; }
sub deadline {
    my ($c) = @_;
    my $d = monotonic() + $c->{timeout};
    return defined($c->{deadline}) && $c->{deadline} < $d ? $c->{deadline} : $d;
}
sub remaining {
    my ($deadline, $operation) = @_;
    my $left = $deadline - monotonic();
    $left > 0 or die "socket $operation timeout\n";
    return $left;
}
sub send_text {
    my ($c, $text) = @_;
    my $offset = 0;
    my $end = deadline($c);
    my $select = IO::Select->new($c->{socket});
    remaining($end, 'write');
    while ($offset < length($text)) {
        my $left = remaining($end, 'write');
        local $! = 0;
        unless ($select->can_write($left)) {
            next if $!{EINTR};
            die "socket write timeout\n";
        }
        remaining($end, 'write');
        my $n = syswrite($c->{socket}, $text, length($text) - $offset, $offset);
        unless (defined $n) {
            next if $!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK};
            die "socket write: $!\n";
        }
        $n > 0 or die "socket write returned zero\n";
        $offset += $n;
    }
    remaining($end, 'write');
}
sub read_line {
    my ($c, $limit) = @_;
    my $end = deadline($c);
    my $select = IO::Select->new($c->{socket});
    while (1) {
        remaining($end, 'read');
        my $nl = index($c->{buffer}, "\n");
        if (defined $limit) {
            die "oversized socket response\n"
                if ($nl >= 0 ? $nl + 1 >= $limit : length($c->{buffer}) >= $limit);
        }
        return substr($c->{buffer}, 0, $nl + 1, '') if $nl >= 0;
        if ($c->{eof}) {
            return undef unless length($c->{buffer});
            my $last = $c->{buffer}; $c->{buffer} = ''; return $last;
        }
        my $left = remaining($end, 'read');
        local $! = 0;
        unless ($select->can_read($left)) {
            next if $!{EINTR};
            die "socket read timeout\n";
        }
        remaining($end, 'read');
        my $n = sysread($c->{socket}, my $chunk, 65536);
        unless (defined $n) {
            next if $!{EINTR} || $!{EAGAIN} || $!{EWOULDBLOCK};
            die "socket read: $!\n";
        }
        if (!$n) { $c->{eof} = 1; next; }
        $c->{buffer} .= $chunk;
    }
}
sub close_socket {
    my ($c) = @_;
    return unless $c && $c->{socket};
    close $c->{socket} or die "socket close: $!\n";
    delete $c->{socket};
}
1;
