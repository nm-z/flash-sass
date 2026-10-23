#!/usr/bin/env bash
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
validator=${1:-"$test_dir/ple_io_validate.pl"}
work=$(mktemp -d "${TMPDIR:-/tmp}/flash-ple-validator-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
# Synthetic parser fixtures only: these numbers are not measurements.
perl - "$work" <<'PERL'
use strict;
use warnings;
my ($dir) = @ARGV;
my @columns = qw(
    phase trial readers resident_pages pages hint_ns read_ns total_ns hash
    bytes errors hint_error offsets schema_version instrumentation comparison
    sample_set repeat order cache_class valid failure_mask residency_error
    reference_errors parity_errors audit_recorded audit_dropped
);
for my $fs (qw(local nfs)) {
    open my $out, '>', "$dir/$fs.csv" or die $!;
    print {$out} join(',', @columns), "\n";
    for my $i (0 .. 35) {
        my $warm = $i >= 4;
        my $trial = int(($warm ? $i - 4 : $i) / 2);
        my $order = $i % 2;
        my $set = $warm ? $trial % 4 : undef;
        my $repeat = $warm ? int($trial / 4) : 0;
        my $arm = $order ^ (($warm ? $set + $repeat : $trial) & 1);
        $set = 2 * $trial + $arm unless $warm;
        my $resident = $warm ? 16 : ($arm ? 8 : 0);
        $resident = 8 if $warm && $fs eq 'local' && $trial % 4 == 3 && $arm == 1;
        $resident = 0 if $warm && $fs eq 'nfs' && $trial % 4 == 2 && $arm == 0;
        my $class = !$resident ? 'client_absent' : $resident == 16 ? 'client_resident' : 'client_mixed';
        my $total = 100 + 100 * $arm + ($fs eq 'nfs' ? 50 : 0);
        my $offsets = '"' . join(';', map { ($set * 16 + $_) * 1048576 + 8192 + 192 } 0 .. 15) . '"';
        my @row = (
            $warm ? 'warm' : 'unprimed', $trial, $arm ? 16 : 1, $resident, 16,
            10, $total - 10, $total, sprintf('%016x', 256 + $set), 1440, 0, 0, $offsets,
            2, 'none', $warm ? 'same_inputs' : 'disjoint_inputs',
            $set, $repeat, $order, $class, 1, 0, 0, 0, 0, 0, 0,
        );
        print {$out} join(',', @row), "\n";
    }
    close $out or die $!;
}
PERL
checks=0
fail() { printf 'not ok %d - %s\n' "$checks" "$1"; exit 1; }
perl "$validator" "$work/local.csv" "$work/nfs.csv" 2 16 > "$work/result.json"
perl -MJSON::PP - "$work/result.json" <<'PERL'
use strict;
use warnings;
local $/;
my $j = decode_json(<>);
die "status/count mismatch\n"
    unless $j->{status} eq 'pass' && $j->{matched_inputs}
    && $j->{local}{rows} == 36 && $j->{nfs}{rows} == 36
    && $j->{local}{warm_pairs} == 16
    && $j->{local}{resident_matched_warm_pairs} == 12
    && $j->{nfs}{resident_matched_warm_pairs} == 12
    && $j->{cross_filesystem_resident_warm_pairs} == 8;
die "median mismatch\n"
    unless $j->{local}{median_total_ns}{serial} eq '100'
    && $j->{local}{median_total_ns}{parallel} eq '200'
    && $j->{nfs}{median_total_ns}{serial} eq '150'
    && $j->{nfs}{median_total_ns}{parallel} eq '250';
PERL
checks=$((checks + 1))
printf 'ok %d - complete paired records and resident-only summaries\n' "$checks"
reject() {
    local mutation=$1
    perl - "$work/nfs.csv" "$work/bad.csv" "$mutation" <<'PERL'
use strict;
use warnings;
my ($input, $output, $mode) = @ARGV;
open my $in, '<', $input or die $!;
my @line = <$in>;
close $in or die $!;
chomp @line;
my @header = split /,/, $line[0], -1;
my %column;
@column{@header} = 0 .. $#header;
sub edit {
    my ($row, $key, $value) = @_;
    my @v = split /,/, $line[$row], -1;
    $v[$column{$key}] = $value;
    $line[$row] = join ',', @v;
}
sub shift_offset {
    my ($row, $amount) = @_;
    my @v = split /,/, $line[$row], -1;
    $v[$column{offsets}] =~ s/\A"|"\z//g;
    my @o = split /;/, $v[$column{offsets}];
    $o[0] += $amount;
    edit($row, 'offsets', '"' . join(';', @o) . '"');
}
if ($mode =~ /\A([^=]+)=(.*)\z/) { edit(1, $1, $2); }
elsif ($mode eq 'missing') { pop @line; }
elsif ($mode eq 'duplicate') { $line[-1] = $line[-2]; }
elsif ($mode eq 'extra') { push @line, $line[1]; }
elsif ($mode eq 'bad_header') { $line[0] .= ',extra'; }
elsif ($mode eq 'warm_hash') { edit(5, 'hash', 'ffffffffffffffff'); }
elsif ($mode eq 'warm_offsets') { shift_offset(5, 90); }
elsif ($mode eq 'cross_hash' || $mode eq 'cross_offsets' || $mode eq 'overlap') {
    for my $i (1 .. $#line) {
        my @v = split /,/, $line[$i], -1;
        my $set = $v[$column{sample_set}];
        if ($mode eq 'cross_hash' && $set == 0) { edit($i, 'hash', 'ffffffffffffffff'); }
        elsif ($mode eq 'cross_offsets' && $set == 0) { shift_offset($i, 4096); }
        elsif ($mode eq 'overlap' && $set == 1) { shift_offset($i, -16 * 1048576); }
    }
} elsif ($mode ne 'unterminated') { die "unknown mutation: $mode\n"; }
open my $out, '>', $output or die $!;
print {$out} join("\n", @line);
print {$out} "\n" unless $mode eq 'unterminated';
close $out or die $!;
PERL
    checks=$((checks + 1))
    if perl "$validator" "$work/local.csv" "$work/bad.csv" 2 16 > "$work/rejected.json" 2> "$work/rejected.err"; then
        fail "accepted $mutation"
    fi
    [[ ! -s "$work/rejected.json" && -s "$work/rejected.err" ]] ||
        fail "missing diagnostic or published success for $mutation"
    printf 'ok %d - rejects %s\n' "$checks" "$mutation"
}
for mutation in \
    'schema_version=1' 'instrumentation=pread_audit' 'valid=0' \
    'failure_mask=1' 'errors=1' 'hint_error=5' 'residency_error=38' \
    'reference_errors=1' 'parity_errors=1' 'audit_recorded=16' \
    'audit_dropped=1' 'bytes=1439' 'total_ns=999' \
    'cache_class=client_resident' 'resident_pages=17' 'pages=0' \
    'readers=15' 'comparison=same_inputs' 'sample_set=99' \
    'repeat=1' 'order=1' 'hash=xyz' \
    warm_hash warm_offsets cross_hash cross_offsets overlap \
    missing duplicate extra unterminated bad_header
do
    reject "$mutation"
done
checks=$((checks + 1))
if perl "$validator" "$work/local.csv" "$work/local.csv" 2 16 > "$work/rejected.json" 2> "$work/rejected.err"; then
    fail 'accepted the same CSV file for both filesystems'
fi
printf 'ok %d - rejects identical input file identities\n' "$checks"
printf '1..%d\n' "$checks"
