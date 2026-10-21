#!/usr/bin/env perl
# Negative controls against retained real captures. Mutations affect temporary
# copies only; the unchanged archived pair exercises the complete comparison.
use strict;
use warnings;
use JSON::PP;
use File::Temp qw(tempdir);
use Cwd qw(abs_path);
use FindBin qw($Bin);
die "usage: qsa_model_compare_test.pl REF_DIR HISTORICAL_CAND_DIR CAND_MANIFEST\n" unless @ARGV == 3;
my ($ref, $cand, $manifest) = map { abs_path($_) // die "missing input $_\n" } @ARGV;
my $root = tempdir('qsa-model-controls-XXXXXX', TMPDIR => 1, CLEANUP => 1);
my $json = JSON::PP->new->canonical;
sub read_bytes {
    open my $f, '<:raw', $_[0] or die "open $_[0]: $!\n";
    local $/; my $data = <$f>;
    close $f or die "close: $!\n";
    return $data;
}
sub replace {
    my ($path, $data) = @_;
    unlink $path if -l $path;
    open my $f, '>:raw', $path or die "write $path: $!\n";
    print {$f} $data or die "write: $!\n";
    close $f or die "close: $!\n";
}
sub clone {
    my ($source, $name) = @_;
    my $dir = "$root/$name";
    mkdir $dir or die "mkdir $dir: $!\n";
    opendir my $dh, $source or die "opendir: $!\n";
    for my $file (readdir $dh) {
        next if $file eq '.' || $file eq '..';
        die "expected flat capture directory\n" unless -f "$source/$file";
        symlink "$source/$file", "$dir/$file" or die "symlink: $!\n";
    }
    closedir $dh;
    return $dir;
}
my ($tests, $failures) = (0, 0);
sub check {
    my ($name, $r, $c, $pattern, $with_history) = @_;
    ++$tests;
    my $output = "$root/out-$tests";
    my @command = ($^X, "$Bin/qsa_model_compare.pl");
    push @command, '--historical-v1' if $with_history;
    push @command, $r, $c, $manifest, $output;
    my $pid = fork();
    die "fork: $!\n" unless defined $pid;
    if (!$pid) {
        open STDOUT, '>', "$root/log-$tests" or die "stdout: $!\n";
        open STDERR, '>&', \*STDOUT or die "stderr: $!\n";
        exec @command;
        die "exec: $!\n";
    }
    waitpid $pid, 0;
    my $status = $?;
    my $log = read_bytes("$root/log-$tests");
    my $ok = $status != 0 && $status != (3 << 8) && $log =~ $pattern && !-e "$output/comparison.json";
    print($ok ? 'ok ' : 'not ok ', "$tests - $name\n");
    if (!$ok) { ++$failures; print "# status=$status log=$log\n"; }
}
check('historical capture requires an explicit diagnostic flag', $ref, $cand, qr/candidate must be completed v2/, 0);
my $other_prefix = clone($cand, 'other-prefix');
my $prefix = read_bytes("$other_prefix/ids.u32");
die "control requires the four-token diagnostic prefix\n" unless length($prefix) == 16;
substr($prefix, 4, 4, pack('L<', 99));
replace("$other_prefix/ids.u32", $prefix);
check('matching selected tokens cannot conceal a different earlier prefix', $ref, $other_prefix, qr/full teacher-forced prefixes differ/, 1);
my $incomplete = clone($ref, 'incomplete');
my $info = $json->decode(read_bytes("$incomplete/capture_info.json"));
$info->{complete} = JSON::PP::false;
replace("$incomplete/capture_info.json", $json->encode($info));
check('incomplete native capture is rejected', $incomplete, $cand, qr/not a completed native QSA capture/, 1);
my $missing = clone($ref, 'missing-layer');
my @records = map { $json->decode($_) } split /\n/, read_bytes("$missing/tensors.jsonl");
my @kept = grep { !($_->{visible_length} == 4 && $_->{layer} == 47 && $_->{node} eq 'attn_output') } @records;
die "required layer fixture absent\n" unless @kept + 1 == @records;
replace("$missing/tensors.jsonl", join('', map { $json->encode($_) . "\n" } @kept));
$info = $json->decode(read_bytes("$missing/capture_info.json"));
$info->{diagnostic_tensors} = scalar @kept;
replace("$missing/capture_info.json", $json->encode($info));
check('a consistent total count cannot conceal a missing QSA layer output', $missing, $cand, qr/missing attn_output at length 4 layer 47/, 1);
my $truncated = clone($ref, 'truncated-tensor');
my $tensor = "$truncated/L000001-B03-attn_pregate.bin";
replace($tensor, substr(read_bytes($tensor), 0, -4));
check('truncated attention output is rejected', $truncated, $cand, qr/tensor byte size differs/, 1);
my $bad_selection = clone($ref, 'future-selection');
replace("$bad_selection/L000001-B03-selected_ids.u32", pack('L<', 1));
check('a future position cannot enter the selection diagnostic', $bad_selection, $cand, qr/invalid selected prefix position/, 1);
my $changed = clone($cand, 'changed-logits');
replace("$changed/logits.f32", read_bytes("$changed/logits.f32") . pack('f<', 0));
check('candidate logits must match their frozen identity', $ref, $changed, qr/candidate logits.f32 differs from frozen manifest/, 1);
print "1..$tests\n";
exit($failures ? 1 : 0);
