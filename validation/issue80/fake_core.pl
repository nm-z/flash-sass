#!/usr/bin/env perl
# A stand-in for flashcore on a Unix socket: "silent" reads requests and never answers; "answer" replies to chat with one token and done.
# Each connection's close is logged with its duration, so the caller can see flashd hang up on a request it cancelled.
use strict; use warnings; use IO::Socket::UNIX; use Time::HiRes 'time';
my ($path, $mode, $log) = @ARGV; die "usage: $0 SOCKET silent|answer LOG\n" unless $log && $mode =~ /\A(silent|answer)\z/;
unlink $path; my $srv = IO::Socket::UNIX->new(Type => SOCK_STREAM, Local => $path, Listen => 8) or die "listen $path: $!\n";
open my $lf, '>>', $log or die "$log: $!\n"; $lf->autoflush(1);
$SIG{CHLD} = 'IGNORE';
while (my $c = $srv->accept) {
	next if fork;
	my $t0 = time; my $chat = 0;
	while (my $l = <$c>) {
		if ($l =~ /\Ainfo/) { print $c "info 65536 4096\n"; }
		if ($l =~ /\Achat /) { $chat = 1; if ($mode eq 'answer') { print $c "tok 9707\n", "done 1 1 4 12.5 10.0 0 0\n"; } }
	}
	printf $lf "closed chat=%d after %.1f s\n", $chat, time - $t0; exit 0;
}
