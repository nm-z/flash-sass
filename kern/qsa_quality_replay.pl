#!/usr/bin/env perl
# Validate frozen layer metadata, execute the numerical control, and bind all
# consumed/produced files by SHA-256. Hashes do not attest checkpoint origin.
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA ();
use File::Basename qw(dirname);
use File::Path qw(make_path);
use FindBin;
use JSON::PP ();
use POSIX qw(isfinite);
use Scalar::Util qw(looks_like_number);
use Fcntl qw(O_WRONLY O_CREAT O_EXCL);

@ARGV == 3 or die "usage: qsa_quality_replay.pl CAPTURE_DIR N NEW_OUTPUT_JSON\n";
my ($capture,$n,$output)=@ARGV;
$n =~ /\A[1-9][0-9]*\z/ && $n <= 4294967295 or die "N must be a positive u32\n";
$capture=abs_path($capture);
defined($capture) && -d $capture or die "capture directory does not exist\n";
my $json=JSON::PP->new->utf8->canonical->pretty;
my %parsed_sha;

sub read_json {
    my ($path)=@_;
    open my $f,'<:raw',$path or die "$path: $!\n";
    local $/;
    my $text=<$f>;
    close $f or die "$path: $!\n";
    $parsed_sha{$path}=Digest::SHA::sha256_hex($text);
    my $v=eval { $json->decode($text) };
    !$@ && ref($v) eq 'HASH' or die "$path: invalid JSON object: $@\n";
    return $v;
}
sub integer {
    my ($v,$name)=@_;
    defined($v) && !ref($v) && "$v" =~ /\A[0-9]+\z/ && $v <= 4294967295
        or die "$name must be an unsigned u32\n";
    return 0+$v;
}
sub sha256 {
    my ($path)=@_;
    -f $path or die "$path: not a regular file\n";
    open my $f,'<:raw',$path or die "$path: $!\n";
    my $sha=Digest::SHA->new(256);
    $sha->addfile($f);
    close $f or die "$path: $!\n";
    return $sha->hexdigest;
}
my $complete_path="$capture/complete.json";
my $snapshot_path="$capture/n$n/snapshot.json";
my $complete=read_json($complete_path);
my $snapshot=read_json($snapshot_path);
($complete->{schema}//'') eq 'flash-sass.qsa-layer-capture.v1'
    or die "unsupported completion schema\n";
($snapshot->{schema}//'') eq 'flash-sass.qsa-layer-snapshot.v1'
    or die "unsupported snapshot schema\n";
my $positions=integer($complete->{positions},'positions');
my $budget=integer($complete->{budget},'budget');
$positions >= $n && $budget > 0 or die "invalid positions/budget\n";
my $epsilon=$complete->{epsilon};
defined($epsilon) && !ref($epsilon) && looks_like_number($epsilon)
    && isfinite($epsilon) && $epsilon >= 0 or die "invalid epsilon\n";
($complete->{byte_order}//'') eq 'little' or die "byte order must be little\n";
ref($complete->{snapshot_lengths}) eq 'ARRAY' or die "snapshot_lengths must be an array\n";
my %lengths;
for my $length (@{$complete->{snapshot_lengths}}) {
    $length=integer($length,'snapshot length');
    $length > 0 && $length <= $positions or die "snapshot outside prefix\n";
    !$lengths{$length}++ or die "duplicate snapshot length\n";
}
$lengths{$n} or die "snapshot is not declared complete\n";
my $blocks=int($n/4);
my $selected=$blocks < $budget ? $blocks : $budget;
my %expected=(
    length=>$n,position=>$n-1,budget=>$budget,complete_blocks=>$blocks,
    tail=>$n%4,attended_keys=>4*$selected+$n%4,
);
for my $key (sort keys %expected) {
    integer($snapshot->{$key},"snapshot.$key")==$expected{$key}
        or die "snapshot.$key contradicts N/budget\n";
}
for my $key (qw(layer die)) {
    integer($snapshot->{$key},"snapshot.$key")==integer($complete->{$key},"complete.$key")
        or die "$key differs between metadata files\n";
}
JSON::PP::is_bool($snapshot->{sparse_exclusion}) or die "sparse_exclusion must be boolean\n";
!!$snapshot->{sparse_exclusion} == !!($blocks>$budget)
    or die "sparse_exclusion contradicts budget\n";
my $synthetic=($complete->{evidence_kind}//'') eq 'synthetic_cpu_fixture';
my $snapshot_synthetic=($snapshot->{evidence_kind}//'') eq 'synthetic_cpu_fixture';
$synthetic == $snapshot_synthetic or die "conflicting evidence kind\n";
if (!$synthetic) {
    ($complete->{input_boundary}//'') eq 'post_projection_before_norm_rope_and_kv_quantization'
        or die "unsupported input boundary\n";
    ($complete->{gamma}//'') eq 'supplied_multiplicative_f32' or die "unsupported gamma convention\n";
}
my $binary=$ENV{QSA_QUALITY_BIN}//"$FindBin::Bin/qsa_quality";
$binary=abs_path($binary);
defined($binary) && -x $binary or die "build kern/qsa_quality first\n";
make_path(dirname($output)) unless -d dirname($output);
-e $output and die "refusing to overwrite $output\n";
my @inputs=($complete_path,$snapshot_path,
    map { "$capture/$_" } qw(raw_ik.f32 rope_cs.f32 rope_sn.f32 iqn.f32 ikn.f32));
push @inputs,map { "$capture/n$n/$_" } qw(iqk.f32 iq.f32 pool.f16 selected_ids.u32);
push @inputs,"$capture/n$n/scores.f32" if $blocks>$budget;
my %before=map { $_=>sha256($_) } @inputs;
for my $path ($complete_path,$snapshot_path) {
    $parsed_sha{$path} eq $before{$path} or die "metadata changed after parsing: $path\n";
}
my $executable_sha=sha256($binary);
my $pending="$output.pending.$$";
system {$binary} $binary,'--replay',$capture,"$n","$positions","$budget","$epsilon",$pending;
$? == 0 or die "replay failed, wait status=$?; no final report created\n";
my %after=map { $_=>sha256($_) } @inputs;
for my $path (@inputs) {
    $before{$path} eq $after{$path} or die "input changed during replay: $path; no final report created\n";
}
sha256($binary) eq $executable_sha or die "replay executable changed\n";
my $report=read_json($pending);
$report->{computation_complete} or die "incomplete C report\n";
my @artifacts=($report->{pool_fp64_artifact},$report->{query_fp64_artifact});
for my $mode (@{$report->{modes}}) {
    push @artifacts,$mode->{scores_artifact},$mode->{ids_artifact};
}
my %hashes=map { $_=>sha256($_) } @artifacts;
$report->{capture_completion_metadata}=$complete;
$report->{capture_snapshot_metadata}=$snapshot;
$report->{input_sha256}=\%after;
$report->{artifact_sha256}=\%hashes;
$report->{input_hashes_match_before_after_replay}=JSON::PP::true;
$report->{replay_executable_sha256}=$executable_sha;
$report->{wrapper_source_sha256}=sha256(__FILE__);
my $origin=$synthetic ? 'synthetic_cpu_fixture' : ($ENV{QSA_CAPTURE_ORIGIN}//'unverified_external_capture');
$origin =~ /\A(?:synthetic_cpu_fixture|gpu_capture_producer_declared|unverified_external_capture)\z/
    or die "unsupported QSA_CAPTURE_ORIGIN\n";
$report->{input_origin}=$origin;
$report->{gpu_execution_performed_by_replay}=JSON::PP::false;
$report->{quality_validation}=JSON::PP::false;
$report->{independent_checkpoint_evidence}=JSON::PP::false;
$report->{complete}=JSON::PP::true;
my $temporary="$output.metadata.$$";
sysopen my $out,$temporary,O_WRONLY|O_CREAT|O_EXCL,0600 or die "$temporary: $!\n";
binmode $out,':raw';
print {$out} $json->encode($report) or die "$temporary: $!\n";
close $out or die "$temporary: $!\n";
# link fails if a concurrent writer created OUTPUT. No stale success report.
link $temporary,$output or die "cannot publish report: $!\n";
unlink $temporary or die "cannot remove metadata temporary: $!\n";
unlink $pending or die "cannot remove numerical temporary: $!\n";
print "Hashed replay report: $output\n";
