#!/usr/bin/env perl
use strict;
use warnings;
use File::Temp qw(tempdir);
use FindBin;
use JSON::PP qw(decode_json);
use Test::More;

my $tmp = tempdir(CLEANUP => 1);
my $binary = "$FindBin::Bin/model_capture_test";
sub slurp {
    open my $f, '<:raw', $_[0] or die "$_[0]: $!";
    local $/; return <$f>;
}
sub run {
    my ($mode) = @_;
    my $directory = "$tmp/$mode";
    my $pid = fork();
    die "fork: $!" unless defined $pid;
    if (!$pid) {
        delete @ENV{qw(MODEL_CAPTURE MODEL_NPH MODEL_INJECT)};
        open STDOUT, '>', "$directory.stdout" or die $!;
        open STDERR, '>', "$directory.stderr" or die $!;
        exec {$binary} $binary, $directory, $mode;
        die "exec: $!";
    }
    waitpid($pid, 0);
    return ($? & 127 ? 128 + ($? & 127) : $? >> 8, $directory);
}
my ($rc, $dir) = run('complete');
is($rc, 0, 'complete two-row mock capture succeeds');
my $metadata = decode_json(slurp("$dir/capture_info.json"));
is($metadata->{schema}, 'flash-sass.raw-model-capture.v2', 'current schema is explicit');
ok(JSON::PP::is_bool($metadata->{complete}) && $metadata->{complete}, 'completion is a true JSON boolean');
is($metadata->{dtype}, '<f4', 'IEEE little-endian FP32 declared');
is($metadata->{positions}, 2, 'both positions completed');
ok(!-e "$dir/capture_info.pending.json", 'temporary marker removed after publication');
is(-s "$dir/logits.f32", 2 * 248320 * 4, 'complete vocabulary rows have exact size');
is_deeply([unpack('V*', slurp("$dir/ids.u32"))], [10,11], 'teacher-forced IDs preserved');
my @logits = unpack('f<*', slurp("$dir/logits.f32"));
is_deeply([@logits[0,124159,124160,248319,248320,496639]],
          [0,124159,124160,248319,1,248320], 'die slices preserve vocabulary and row order');
for my $case (['interrupted',7], ['readback-failure',4], ['duplicate',2],
              ['skip',2], ['overlap',2], ['missing',2], ['out-of-range',2]) {
    ($rc, $dir) = run($case->[0]);
    is($rc, $case->[1], "$case->[0] returns the expected failure");
    ok(!-e "$dir/capture_info.json", "$case->[0] never publishes completion");
}
($rc, $dir) = run('disabled');
is($rc, 0, 'disabled capture remains a no-op');
ok(!-e $dir, 'disabled capture creates no directory');
mkdir "$tmp/existing" or die $!;
open my $keep, '>', "$tmp/existing/sentinel" or die $!;
print {$keep} "preserved\n";
close $keep or die $!;
($rc, $dir) = run('existing');
is($rc, 2, 'existing directory is rejected');
is(slurp("$dir/sentinel"), "preserved\n", 'existing contents remain intact');
ok(!-e "$dir/capture_info.json", 'existing directory never gains a completion marker');
done_testing();
