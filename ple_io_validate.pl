#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP;
use Math::BigInt;
my ($local, $nfs, $batches, $readers) = @ARGV;
@ARGV == 4
    && $batches =~ /\A[1-9][0-9]*\z/ && $batches <= 512
    && $readers =~ /\A[1-9][0-9]*\z/ && $readers >= 2 && $readers <= 16
    or die "usage: $0 LOCAL.csv NFS.csv BATCHES(1..512) READERS(2..16)\n";
my @columns = qw(
    phase trial readers resident_pages pages hint_ns read_ns total_ns hash
    bytes errors hint_error offsets schema_version instrumentation comparison
    sample_set repeat order cache_class valid failure_mask residency_error
    reference_errors parity_errors audit_recorded audit_dropped
);
my @numbers = qw(
    trial readers resident_pages pages hint_ns read_ns total_ns bytes errors
    hint_error schema_version sample_set repeat order valid failure_mask
    residency_error reference_errors parity_errors audit_recorded audit_dropped
);
my @zero = qw(
    errors hint_error failure_mask residency_error reference_errors
    parity_errors audit_recorded audit_dropped
);
my $header = join ',', @columns;
sub bad {
    my ($where, $message) = @_;
    die "ple_io_validate: $where: $message\n";
}
sub uint {
    my ($value, $where, $name, $max) = @_;
    $max //= '18446744073709551615';
    defined($value) && $value =~ /\A(?:0|[1-9][0-9]*)\z/
        && (length($value) < length($max)
            || (length($value) == length($max) && $value le $max))
        or bad($where, "$name is not a supported unsigned integer");
}
# Exact decimal strings avoid overflow or floating-point rounding.
sub median {
    my @v = sort { length($a) <=> length($b) || $a cmp $b } @_;
    return undef unless @v;
    my $sum = Math::BigInt->new($v[int($#v / 2)])->badd($v[int(@v / 2)]);
    my $half = $sum->is_odd;
    $sum->bdiv(2);
    return $sum->bstr . ($half ? '.5' : '');
}
sub read_csv {
    my ($path) = @_;
    open my $fh, '<', $path or bad($path, "open failed: $!");
    -f $fh or bad($path, 'input is not a regular file');
    my @stat = stat $fh;
    my $first = <$fh>;
    defined($first) && $first =~ s/\r?\n\z//
        or bad($path, 'missing or unterminated header');
    $first eq $header or bad($path, 'unexpected CSV header/schema');
    my (@sets, @ranges, @pair, @serial, @parallel, @resident);
    my %cache_counts = map {
        $_ => { map { $_ => 0 } qw(client_absent client_mixed client_resident) }
    } qw(unprimed warm);
    my $count = 0;
    while (my $line = <$fh>) {
        my $where = "$path:" . ($count + 2);
        $count < 18 * $batches or bad($where, 'extra CSV record');
        $line =~ s/\r?\n\z// or bad($where, 'unterminated CSV record');
        # The benchmark's only quoted field contains semicolons, not commas.
        my @fields = split /,/, $line, -1;
        @fields == @columns or bad($where, 'expected exactly 27 columns');
        my %r;
        @r{@columns} = @fields;
        uint($r{$_}, $where, $_) for @numbers;
        $r{schema_version} eq '2' or bad($where, 'schema_version must be 2');
        $r{instrumentation} eq 'none' or bad($where, 'instrumented rows are not normal timing evidence');
        $r{valid} eq '1' or bad($where, 'invalid benchmark record');
        for my $name (@zero) { $r{$name} eq '0' or bad($where, "$name must be zero"); }
        $r{bytes} eq '1440' or bad($where, 'bytes must be 1440');
        Math::BigInt->new($r{hint_ns})->badd($r{read_ns})->bcmp($r{total_ns}) == 0
            or bad($where, 'total_ns differs from hint_ns + read_ns');
        $r{hash} =~ /\A[0-9a-f]{16}\z/ or bad($where, 'invalid 64-bit fingerprint');
        $r{offsets} =~ /\A"([0-9]+(?:;[0-9]+){15})"\z/
            or bad($where, 'offsets must contain exactly 16 quoted integers');
        my @offsets = split /;/, $1;
        uint($_, $where, 'offset', '9223372036854775718') for @offsets;
        $r{pages} >= 16 && $r{pages} <= 32 && $r{resident_pages} <= $r{pages}
            or bad($where, 'invalid page residency counts');
        my $cache = $r{resident_pages} == 0 ? 'client_absent'
            : $r{resident_pages} == $r{pages} ? 'client_resident' : 'client_mixed';
        $r{cache_class} eq $cache or bad($where, 'cache_class disagrees with residency counts');
        my ($phase, $trial, $order, $arm, $set, $repeat, $comparison);
        if ($count < 2 * $batches) {
            $phase = 'unprimed'; $trial = int($count / 2); $order = $count % 2;
            $arm = $order ^ ($trial & 1); $set = 2 * $trial + $arm; $repeat = 0;
            $comparison = 'disjoint_inputs';
        } else {
            my $i = $count - 2 * $batches;
            $phase = 'warm'; $trial = int($i / 2); $order = $i % 2;
            $set = $trial % (2 * $batches); $repeat = int($trial / (2 * $batches));
            $arm = $order ^ (($set + $repeat) & 1); $comparison = 'same_inputs';
        }
        my %expected = (
            phase => $phase, trial => $trial, order => $order, readers => $arm ? $readers : 1,
            sample_set => $set, repeat => $repeat, comparison => $comparison,
        );
        for my $name (sort keys %expected) {
            $r{$name} eq "$expected{$name}" or bad($where, "$name=$r{$name}; expected $expected{$name}");
        }
        $cache_counts{$phase}{$cache}++;
        my $identity = join "\t", @r{qw(offsets hash pages)};
        if ($phase eq 'unprimed') {
            $sets[$set] = $identity;
            push @ranges, map { [0 + $_, $where] } @offsets;
        } else {
            $identity eq $sets[$set] or bad($where, 'warm offsets/fingerprint/pages changed');
            $pair[$arm] = [$r{total_ns}, $cache eq 'client_resident'];
            if ($order == 1) {
                my $both = $pair[0][1] && $pair[1][1] ? 1 : 0;
                push @resident, $both;
                if ($both) { push @serial, $pair[0][0]; push @parallel, $pair[1][0]; }
            }
        }
        $count++;
    }
    close $fh or bad($path, "close failed: $!");
    $count == 18 * $batches or bad($path, "truncated CSV: $count rows; expected " . 18 * $batches);
    @ranges = sort { $a->[0] <=> $b->[0] } @ranges;
    for my $i (1 .. $#ranges) {
        $ranges[$i][0] > $ranges[$i - 1][0] + 89 or bad($ranges[$i][1], 'unprimed byte ranges overlap');
    }
    return {
        inode => join(':', @stat[0, 1]), sets => \@sets, resident => \@resident,
        summary => {
            file => $path, rows => $count, cache_class_rows => \%cache_counts,
            warm_pairs => 8 * $batches, resident_matched_warm_pairs => scalar(@serial),
            median_total_ns => { serial => median(@serial), parallel => median(@parallel) },
        },
    };
}
my $l = read_csv($local);
my $n = read_csv($nfs);
$l->{inode} ne $n->{inode} or bad('inputs', 'LOCAL.csv and NFS.csv refer to the same file');
# Warm rows were bound to their original unprimed sample set.
for my $set (0 .. 2 * $batches - 1) {
    $l->{sets}[$set] eq $n->{sets}[$set] or bad('inputs', "local/NFS mismatch for sample_set=$set");
}
my $common = grep { $l->{resident}[$_] && $n->{resident}[$_] } 0 .. 8 * $batches - 1;
print JSON::PP->new->canonical->pretty->encode({
    status => 'pass', csv_schema_version => 2, batches => 0 + $batches, parallel_readers => 0 + $readers,
    matched_inputs => JSON::PP::true, cross_filesystem_resident_warm_pairs => $common,
    interpretation => 'Descriptive medians use warm pairs resident in both arms. '
        . 'Unprimed arms use disjoint inputs; no cold same-input speedup is inferred.',
    local => $l->{summary}, nfs => $n->{summary},
});
