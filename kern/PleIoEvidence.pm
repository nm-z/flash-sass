package PleIoEvidence;
use strict;
use warnings;
use Exporter 'import';
use JSON::PP;
use Digest::SHA;
use Cwd 'abs_path';
use File::Path 'make_path';
use File::Basename 'dirname';
use Errno 'EIO';

our @EXPORT_OK = qw(sha put write_text absolute new_output status_fields verify_failure verify_healthy failure_detail);
my $json = JSON::PP->new->utf8->canonical->pretty;
sub sha {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "open $path: $!\n";
    my $h = Digest::SHA->new(256);
    $h->addfile($f);
    close $f or die "close $path: $!\n";
    return $h->hexdigest;
}
sub write_text {
    my ($path, $bytes) = @_;
    open my $f, '>:raw', $path or die "write $path: $!\n";
    print {$f} $bytes or die "write $path: $!\n";
    close $f or die "close $path: $!\n";
}
sub put {
    my ($path, $value) = @_;
    write_text($path, $json->encode($value));
}
sub absolute {
    my ($path) = @_;
    my $result = abs_path($path);
    defined($result) or die "resolve $path: $!\n";
    return $result;
}
sub new_output {
    my ($path) = @_;
    -e $path and die "output already exists: $path\n";
    make_path(dirname($path));
    mkdir($path) or die "mkdir $path: $!\n";
    return absolute($path);
}
sub status_fields {
    my ($line) = @_;
    $line =~ /^ple_io_test / or die "invalid PLE status: $line\n";
    my %s = map { $_->[0] => 0 + $_->[1] }
        map { [split /=/, $_, 2] } grep { /^[a-z][a-z0-9_]*=-?\d+(?:\.\d+)?$/ } split /\s+/, $line;
    for my $key (qw(launches failed pos head snapshot offset bytes expected errnum eof dies launched_mask sync_mask error_mask resumed failure_ms read_calls partial_reads eintr_retries fault_matches fault_reads), map { ("count$_", "error$_") } 0..5) {
        exists $s{$key} or die "status lacks $key: $line\n";
    }
    $s{dies} == 6 or die "six dies required: $line\n";
    return \%s;
}
sub verify_healthy {
    my ($s) = @_;
    !$s->{failed} && !$s->{error_mask} && $s->{launched_mask} == 63 && $s->{sync_mask} == 63
        or die "healthy request has incomplete/error die masks\n";
    $s->{read_calls} > 0 or die "healthy request read no PLE rows\n";
    for (0..5) { !$s->{"error$_"} or die "healthy request retained die $_ error\n"; }
}
sub verify_failure {
    my ($before, $s, $position, $mode, $prelaunch, $bound_ms) = @_;
    $s->{failed} == 1 && $s->{pos} == $position && $s->{head} == 3 && $s->{offset} >= 0
        && $s->{bytes} == 17 && $s->{expected} == 90 && $s->{snapshot} == 0
        && $s->{eof} == ($mode eq 'eof' ? 1 : 0) && $s->{errnum} == ($mode eq 'eof' ? 0 : EIO)
        or die "incorrect failure metadata or snapshot state\n";
    $s->{fault_reads} == 1 && $s->{fault_matches} >= 2 or die "injected fault was not observed\n";
    $s->{failure_ms} >= 0 && $s->{failure_ms} < $bound_ms or die "PLE failure exceeded termination bound\n";
    if ($prelaunch) {
        $s->{launches} == $before->{launches} && !$s->{launched_mask} && !$s->{sync_mask} && !$s->{error_mask}
            or die "prelaunch failure dispatched kernels\n";
    } else {
        $s->{launches} > $before->{launches} && $s->{launched_mask} == 63 && $s->{sync_mask} == 63 && $s->{error_mask} == 63
            or die "midrun failure did not finish every die normally\n";
    }
    for (0..5) {
        $s->{"error$_"} == ($prelaunch ? 0 : 1) && $s->{"count$_"} <= $position
            or die "die $_ consumed beyond the failing position or has wrong error\n";
    }
}
sub failure_detail {
    my ($s) = @_;
    local $! = $s->{errnum};
    my $reason = $s->{eof} ? 'unexpected EOF' : "$!";
    return "PLE I/O position $s->{pos} head $s->{head} offset $s->{offset} bytes $s->{bytes}/$s->{expected}: $reason";
}
1;
