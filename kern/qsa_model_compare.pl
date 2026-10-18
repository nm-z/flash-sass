#!/usr/bin/env perl
# Bind selected native reference rows to a complete candidate prefix before
# invoking the fixed issue #11 numerical gate. Provenance is reported separately.
use strict;
use warnings;
use JSON::PP;
use Digest::SHA qw(sha256_hex);
use FindBin qw($Bin);
use Fcntl qw(O_WRONLY O_CREAT O_EXCL);

my $historical = @ARGV && $ARGV[0] eq '--historical-v1' ? shift(@ARGV) : '';
die "usage: qsa_model_compare.pl [--historical-v1] REF_DIR CAND_DIR CAND_FROZEN_MANIFEST NEW_OUT_DIR\n" unless @ARGV == 4;
my ($ref, $cand, $manifest_path, $out) = @ARGV;
my $json = JSON::PP->new->canonical->pretty;
my %identities;
sub raw {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "open $path: $!\n";
    local $/;
    my $data = <$f>;
    close $f or die "close $path: $!\n";
    die "empty file $path\n" unless defined $data && length $data;
    $identities{$path} = {bytes => length($data), sha256 => sha256_hex($data)};
    return $data;
}
sub object { return $json->decode(raw($_[0])); }
sub new_file {
    my ($name) = @_;
    sysopen my $f, "$out/$name", O_WRONLY | O_CREAT | O_EXCL, 0600 or die "create $name: $!\n";
    binmode $f;
    return $f;
}
sub save {
    my ($name, $data) = @_;
    my $f = new_file($name);
    print {$f} $data or die "write $name: $!\n";
    close $f or die "close $name: $!\n";
}
sub hash_file {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "open $path: $!\n";
    my $digest = Digest::SHA->new(256)->addfile($f)->hexdigest;
    my $size = -s $f;
    close $f or die "close $path: $!\n";
    return {bytes => $size, sha256 => $digest};
}
sub exact_artifact {
    my ($name, $path, $manifest) = @_;
    my $expected = $manifest->{artifacts}{$name} or die "manifest lacks $name\n";
    my $actual = hash_file($path);
    die "candidate $name differs from frozen manifest\n"
        unless $actual->{bytes} == $expected->{bytes} && $actual->{sha256} eq $expected->{sha256};
    $identities{$path} = $actual;
}
my $ri = object("$ref/capture_info.json");
my $ci = object("$cand/capture_info.json");
my $cm = object($manifest_path);
die "reference is not a completed native QSA capture\n"
    unless $ri->{schema} eq 'flash-sass.qsa-model-reference.v1' && $ri->{complete} &&
           $ri->{teacher_forced} && $ri->{prefix_decoded_in_full} && $ri->{dtype} eq '<f4' && $ri->{byte_order} eq 'little';
die "candidate must be completed v2, or explicitly selected historical v1\n"
    unless ($ci->{schema} eq 'flash-sass.raw-model-capture.v2' && $ci->{complete}) ||
           ($historical && $ci->{schema} eq 'flash-sass.raw-model-capture.v1');
die "candidate is not teacher forced or has a different vocabulary\n"
    unless $ci->{teacher_forced} && $ci->{byte_order} eq 'little' && $ci->{vocab_size} == $ri->{vocab_size};
die "unsupported candidate frozen manifest\n"
    unless $cm->{schema} eq 'flash-sass.frozen-logits.v1' && $cm->{dtype} eq '<f4' &&
           $cm->{byte_order} eq 'little' && $cm->{positions} == $ci->{positions} && $cm->{vocab_size} == $ri->{vocab_size};
my $prefix = raw("$ref/all_input_ids.u32");
my $candidate_ids = raw("$cand/ids.u32");
die "full teacher-forced prefixes differ\n" unless $prefix eq $candidate_ids;
die "full-prefix length differs from capture metadata\n"
    unless length($prefix) == 4 * $ri->{input_positions} && $ri->{input_positions} == $ci->{positions};
my $vocab_ids = raw("$ref/vocab_ids.u32");
die "reference vocabulary IDs are not canonical ordered columns\n"
    unless $vocab_ids eq pack('L<*', 0 .. $ri->{vocab_size} - 1);
my $cv = $cm->{artifacts}{'vocab_ids.u32'} or die "candidate lacks vocabulary-column identity\n";
die "vocabulary-column identities differ\n"
    unless length($vocab_ids) == $cv->{bytes} && sha256_hex($vocab_ids) eq $cv->{sha256};
exact_artifact('ids.u32', "$cand/ids.u32", $cm);
exact_artifact('logits.f32', "$cand/logits.f32", $cm);
my @positions = unpack 'L<*', raw("$ref/positions.u32");
my $ids = raw("$ref/ids.u32");
die "selected row count differs from metadata\n"
    unless @positions == $ri->{positions} && length($ids) == 4 * @positions &&
           $identities{"$ref/positions.u32"}{bytes} == 4 * @positions;
my $previous = -1;
for my $r (0 .. $#positions) {
    my $p = $positions[$r];
    die "selected positions must strictly increase within the full prefix\n" unless $p > $previous && $p < $ri->{input_positions};
    die "selected token does not match its full-prefix position\n" unless substr($ids, 4 * $r, 4) eq substr($prefix, 4 * $p, 4);
    $previous = $p;
}
die "selected rows omit the final prefix position\n" unless $previous + 1 == $ri->{input_positions};
my $row_bytes = 4 * $ri->{vocab_size};
die "logit byte counts differ from capture shapes\n"
    unless -s "$cand/logits.f32" == $ci->{positions} * $row_bytes && -s "$ref/logits.f32" == @positions * $row_bytes;
$identities{"$ref/logits.f32"} = hash_file("$ref/logits.f32");

# Summarize the actual native snapshots, preserving their individual identities.
# These are reference diagnostics; no candidate per-layer comparison is implied.
my (%seen, %wanted, @layers);
$wanted{$_ + 1} = 1 for @positions;
my @records = map { $json->decode($_) } split /\n/, raw("$ref/tensors.jsonl");
die "tensor index count differs from completion metadata\n" unless @records == $ri->{diagnostic_tensors};
for my $t (@records) {
    my ($length, $layer, $node, $file) = @{$t}{qw(visible_length layer node file)};
    die "invalid diagnostic index entry\n" unless $wanted{$length} && $layer >= 0 && $layer < 48 &&
        $file =~ /^L[0-9]{6}-B[0-9]{2}-[A-Za-z0-9_]+\.bin$/ && $file eq sprintf('L%06d-B%02d-%s.bin', $length, $layer, $node);
    die "duplicate tensor index entry\n" if $seen{$length}{$layer}{$node}++;
    my $d = raw("$ref/$file");
    die "tensor byte size differs from index\n" unless length($d) == $t->{bytes};
    if ($node eq 'attn_pregate' || $node eq 'attn_output' || $node eq 'l_last') {
        die "summary requires native f32 vectors\n" unless $t->{dtype} eq 'f32' && length($d) % 4 == 0;
        my @x = unpack 'f<*', $d;
        my ($sum_sq, $max_abs) = (0, 0);
        for my $x (@x) { die "nonfinite activation\n" if $x != $x || abs($x) > 3.4028234663852886e38; $sum_sq += $x * $x; $max_abs = abs($x) if abs($x) > $max_abs; }
        push @layers, {visible_length => $length, layer => $layer, node => $node,
                       elements => scalar(@x), rms => sqrt($sum_sq / @x), max_abs => $max_abs,
                       sha256 => sha256_hex($d)};
    }
    if ($node eq 'indexer_sel') {
        my $selected_file = sprintf('L%06d-B%02d-selected_ids.u32', $length, $layer);
        my $selected = raw("$ref/$selected_file");
        my @selected = unpack 'L<*', $selected;
        die "selected-ID byte count mismatch\n" unless length($selected) == 4 * @selected && @selected == $t->{selected_positions};
        my $last = -1;
        for my $p (@selected) { die "invalid selected prefix position\n" unless $p > $last && $p < $length; $last = $p; }
    }
}
for my $length (keys %wanted) {
    for my $layer (0 .. 47) {
        my @required = ('l_last');
        push @required, qw(indexer_k_raw indexer_k indexer_q indexer_score indexer_top_k indexer_sel Qcur Kcur Vcur attn_pregate attn_gated attn_output) if $layer % 4 == 3;
        for my $node (@required) { die "missing $node at length $length layer $layer\n" unless $seen{$length}{$layer}{$node}; }
    }
}
mkdir $out, 0700 or die "create new output directory: $!\n";
my $selected_logits = new_file('candidate-selected-logits.f32');
open my $all_logits, '<:raw', "$cand/logits.f32" or die "open candidate logits: $!\n";
for my $p (@positions) {
    seek($all_logits, $p * $row_bytes, 0) or die "seek candidate logit row: $!\n";
    my $row;
    read($all_logits, $row, $row_bytes) == $row_bytes or die "short candidate logit row\n";
    print {$selected_logits} $row or die "write selected logits: $!\n";
}
close $all_logits or die "close candidate logits: $!\n";
close $selected_logits or die "close selected logits: $!\n";
save('candidate-selected-ids.u32', $ids);
my @command = ("$Bin/qsa_logits_check", "$ref/ids.u32", "$out/candidate-selected-ids.u32", "$ref/logits.f32", "$out/candidate-selected-logits.f32", scalar(@positions), $ri->{vocab_size}, "$out/logits-check.json");
system @command;
my $status = $?;
die "numerical checker failed unexpectedly: $status\n" if $status == -1 || ($status & 127) || (($status >> 8) != 0 && ($status >> 8) != 3);
my $numeric = object("$out/logits-check.json");
# Refuse a completion report if any recorded input changed during this analysis.
for my $path (keys %identities) {
    my $after = hash_file($path);
    die "input changed during comparison: $path\n" unless $after->{bytes} == $identities{$path}{bytes} && $after->{sha256} eq $identities{$path}{sha256};
}
save('reference-layer-diagnostics.json', $json->encode(\@layers));
save('comparison.json', $json->encode({schema => 'flash-sass.qsa-model-comparison.v1',
    scope => 'same-GGUF backend numerical diagnostic with complete-prefix binding',
    candidate_is_historical => $historical ? JSON::PP::true : JSON::PP::false,
    selected_positions => \@positions, input_positions => $ri->{input_positions},
    numerical_limits_pass => $numeric->{numerical_limits_pass},
    per_layer_comparison => 'unmeasured; reference snapshots only',
    independent_checkpoint_evidence => JSON::PP::false, quality_validation => JSON::PP::false,
    issue11_resolved => JSON::PP::false, identities => \%identities, command => \@command,
    complete => JSON::PP::true}));
print "comparison complete; numerical_limits_pass=", ($numeric->{numerical_limits_pass} ? 'true' : 'false'), "; original-checkpoint and per-layer comparison remain unmeasured\n";
exit($status >> 8);
