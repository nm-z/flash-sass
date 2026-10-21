#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Cwd qw(abs_path);
use File::Spec;
use JSON::PP;
use Digest::SHA;
use POSIX qw(strftime);
my ($checker, $out);
Getopt::Long::Configure(qw(no_auto_abbrev no_ignore_case));
GetOptions('checker=s' => \$checker, 'out=s' => \$out) or die "invalid options\n";
die "usage: $0 --checker EXECUTABLE --out NEW_DIRECTORY\n"
    if @ARGV || !defined($checker) || !defined($out);
$checker = abs_path($checker) // die "checker does not exist\n";
die "checker is not executable: $checker\n" unless -f $checker && -x $checker;
die "output already exists: $out\n" if -e $out || -l $out;
mkdir($out, 0700) or die "mkdir $out: $!\n";
$out = abs_path($out) // die "cannot resolve output\n";
my $json = JSON::PP->new->canonical->pretty;
my $started = strftime('%Y-%m-%dT%H:%M:%SZ', gmtime);
sub write_new {
    my ($path, $bytes) = @_;
    open my $fh, '>:raw', $path or die "write $path: $!\n";
    print {$fh} $bytes or die "write $path: $!\n";
    close $fh or die "close $path: $!\n";
}
sub read_all {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "read $path: $!\n";
    local $/;
    my $bytes = <$fh>;
    close $fh or die "close $path: $!\n";
    return defined($bytes) ? $bytes : '';
}
sub sha {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "hash $path: $!\n";
    my $digest = Digest::SHA->new(256)->addfile($fh)->hexdigest;
    close $fh or die "close $path: $!\n";
    return $digest;
}
sub ids { return pack('V*', @_); }
sub floats { return pack('f<*', @_); }
sub bits { return pack('V*', @_); }
sub basic {
    my ($name, %override) = @_;
    return {
        name => $name, rows => '1', vocab => '4', expected_exit => 0,
        ref_ids => ids(0), candidate_ids => ids(0),
        ref_logits => floats(1, 2, 3, 4),
        candidate_logits => floats(1, 2, 3, 4),
        %override,
    };
}
my @cases = (
    basic('identical', rows => '2', ref_ids => ids(0, 2),
          candidate_ids => ids(0, 2),
          ref_logits => floats(1,2,3,4,5,6,7,8),
          candidate_logits => floats(1,2,3,4,5,6,7,8)),
    basic('zero', ref_logits => floats((0)x4), candidate_logits => floats((0)x4)),
    basic('signed_zero', ref_logits => bits(0,0x80000000,0,0x80000000),
          candidate_logits => bits(0x80000000,0,0x80000000,0)),
    basic('below_thresholds', ref_logits => floats(10,20,30,40),
          candidate_logits => floats(10.01,20.01,30.01,40.01)),
    basic('zero_floor_pass', ref_logits => floats((0)x4),
          candidate_logits => floats((5e-11)x4)),
    basic('absolute_only', expected_exit => 3,
          ref_logits => floats((100)x4), candidate_logits => floats(100.125,100,100,100)),
    basic('nrms_only', expected_exit => 3,
          ref_logits => floats((1)x4), candidate_logits => floats((1.02)x4)),
    basic('zero_floor_fail', expected_exit => 3,
          ref_logits => floats((0)x4), candidate_logits => floats((2e-10)x4)),
    basic('largest_score_hides_nrms', vocab => '1024', expected_exit => 3,
          ref_logits => floats(1,(0)x1023), candidate_logits => floats(1,(0.004)x1023)),
    basic('bad_row_hidden_by_aggregate', rows => '2', expected_exit => 3,
          ref_ids => ids(0,1), candidate_ids => ids(0,1),
          ref_logits => floats((1)x4,(100)x4),
          candidate_logits => floats((1.02)x4,(100)x4)),
    basic('different_ids', expected_exit => 2, candidate_ids => ids(1)),
    basic('id_outside_vocabulary', expected_exit => 2, ref_ids => ids(4), candidate_ids => ids(4)),
    basic('short_reference_ids', expected_exit => 2, ref_ids => "\0\0\0"),
    basic('extra_candidate_ids', expected_exit => 2, candidate_ids => ids(0,1)),
    basic('short_candidate_logits', expected_exit => 2,
          candidate_logits => substr(floats(1,2,3,4),0,15)),
    basic('extra_reference_logits', expected_exit => 2,
          ref_logits => floats(1,2,3,4)."\0"),
    basic('empty_logit_file', expected_exit => 2, ref_logits => ''),
    basic('zero_rows', rows => '0', expected_exit => 2),
    basic('zero_vocab', vocab => '0', expected_exit => 2),
    basic('empty_rows_argument', rows => '', expected_exit => 2),
    basic('negative_rows', rows => '-1', expected_exit => 2),
    basic('trailing_junk_rows', rows => '1x', expected_exit => 2),
    basic('decimal_overflow', rows => '18446744073709551616', expected_exit => 2),
    basic('logit_size_overflow', rows => '1152921504606846976', expected_exit => 2),
    basic('vocab_range_overflow', vocab => '4294967296', expected_exit => 2),
    basic('existing_report', expected_exit => 2,
          existing_report => "PREEXISTING REPORT: MUST REMAIN UNCHANGED\n"),
);
for my $side ('ref', 'candidate') {
    for my $nonfinite ([nan => 0x7fc00000], [positive_inf => 0x7f800000], [negative_inf => 0xff800000]) {
        push @cases, basic($side.'_'.$nonfinite->[0], expected_exit => 2,
            $side.'_logits' => bits($nonfinite->[1],0,0,0));
    }
}
my @results;
my %all_files;
my $failures = 0;
$| = 1;
for my $test (@cases) {
    my $name = $test->{name};
    die "unsafe fixture name\n" unless $name =~ /\A[a-z0-9_]+\z/;
    my $dir = File::Spec->catdir($out, $name);
    mkdir($dir,0700) or die "mkdir $dir: $!\n";
    my %path;
    my %before;
    for my $field (qw(ref_ids candidate_ids ref_logits candidate_logits)) {
        my $suffix = $field =~ /ids\z/ ? 'u32' : 'f32';
        $path{$field} = File::Spec->catfile($dir,"$field.$suffix");
        write_new($path{$field}, $test->{$field});
        $before{$field} = sha($path{$field});
    }
    my $result_path = File::Spec->catfile($dir,'result.json');
    write_new($result_path, $test->{existing_report}) if exists $test->{existing_report};
    my @argv = ($checker, @path{qw(ref_ids candidate_ids ref_logits candidate_logits)},
                $test->{rows}, $test->{vocab}, $result_path);
    my $stdout = File::Spec->catfile($dir,'stdout.log');
    my $stderr = File::Spec->catfile($dir,'stderr.log');
    my $pid = fork();
    die "fork: $!\n" unless defined $pid;
    if (!$pid) {
        open STDIN, '<', '/dev/null' or POSIX::_exit(127);
        open STDOUT, '>:raw', $stdout or POSIX::_exit(127);
        open STDERR, '>:raw', $stderr or POSIX::_exit(127);
        exec { $checker } @argv or do {
            print STDERR "exec $checker: $!\n";
            POSIX::_exit(127);
        };
    }
    waitpid($pid,0) == $pid or die "waitpid: $!\n";
    my $status = $?;
    my $signal = $status & 127;
    my $actual = $signal ? undef : $status >> 8;
    my @problems;
    push @problems, "child received signal $signal" if $signal;
    push @problems, 'unexpected exit code'
        if !defined($actual) || $actual != $test->{expected_exit};
    for my $field (sort keys %path) {
        push @problems, "checker changed fixture $field"
            if sha($path{$field}) ne $before{$field};
    }
    my $decoded;
    if (exists $test->{existing_report}) {
        push @problems, 'preexisting report overwritten or removed'
            if !-f $result_path || read_all($result_path) ne $test->{existing_report};
    } elsif ($test->{expected_exit} == 2) {
        push @problems, 'incomplete result was retained' if -e $result_path;
    } else {
        if (!-f $result_path) {
            push @problems, 'numerical result missing';
        } else {
            $decoded = eval { $json->decode(read_all($result_path)) };
            if ($@ || ref($decoded) ne 'HASH') {
                push @problems, 'invalid result JSON';
            } else {
                push @problems, 'result not complete' unless $decoded->{complete};
                push @problems, 'independence incorrectly claimed or omitted'
                    unless exists($decoded->{independent_checkpoint_evidence}) &&
                           !$decoded->{independent_checkpoint_evidence};
                push @problems, 'provenance status not unverified'
                    unless ($decoded->{provenance_status}//'') eq 'not_verified';
                push @problems, 'wrong numerical acceptance result'
                    unless exists($decoded->{numerical_limits_pass}) &&
                           !!$decoded->{numerical_limits_pass} == ($test->{expected_exit} == 0);
                push @problems, 'wrong row count'
                    unless ref($decoded->{per_row}) eq 'ARRAY' &&
                           @{$decoded->{per_row}} == $test->{rows};
                my $a = $decoded->{aggregate};
                if (ref($a) ne 'HASH') {
                    push @problems, 'aggregate metrics missing';
                } else {
                    if ($name eq 'absolute_only') {
                        push @problems, 'absolute-only fixture no longer isolates its gate'
                            unless $a->{max_abs} > 0.10 && $a->{normalized_rms} <= 0.01;
                    }
                    if ($name =~ /\A(?:nrms_only|zero_floor_fail|largest_score_hides_nrms)\z/) {
                        push @problems, 'NRMS-only fixture no longer isolates its gate'
                            unless $a->{max_abs} <= 0.10 && $a->{normalized_rms} > 0.01;
                    }
                    if ($name eq 'bad_row_hidden_by_aggregate') {
                        push @problems, 'bad-row fixture does not conceal failure in aggregate'
                            unless $a->{max_abs} <= 0.10 && $a->{normalized_rms} <= 0.01 &&
                                   $a->{max_row_normalized_rms} > 0.01 && $a->{failing_rows} > 0;
                    }
                    if ($name eq 'signed_zero' || $name eq 'zero') {
                        push @problems, 'zero fixture reported nonzero error'
                            unless $a->{max_abs} == 0 && $a->{normalized_rms} == 0;
                    }
                }
            }
        }
    }
    my %artifacts;
    opendir(my $dh,$dir) or die "opendir $dir: $!\n";
    my @leaves = sort grep { $_ ne '.' && $_ ne '..' } readdir($dh);
    closedir($dh) or die "closedir: $!\n";
    for my $leaf (@leaves) {
        my $file = File::Spec->catfile($dir,$leaf);
        next unless -f $file;
        my $relative = "$name/$leaf";
        my $digest = sha($file);
        $artifacts{$leaf} = { bytes => 0 + (-s $file), sha256 => $digest };
        $all_files{$relative} = $digest;
    }
    my $passed = !@problems;
    ++$failures unless $passed;
    push @results, {
        name => $name, command => \@argv,
        expected_exit => $test->{expected_exit}, actual_exit => $actual,
        signal => $signal, passed => $passed ? JSON::PP::true : JSON::PP::false,
        problems => \@problems, artifacts => \%artifacts,
    };
    print(($passed ? 'PASS ' : 'FAIL '), $name, " expected=$test->{expected_exit} actual=",
          defined($actual) ? $actual : "signal:$signal", "\n");
}
my $summary = {
    schema => 'flash-sass.qsa-logits-checker-regression.v1',
    scope => 'synthetic CPU numerical-checker regression only',
    independent_checkpoint_evidence => JSON::PP::false,
    checker => { path => $checker, sha256 => sha($checker) },
    harness => { path => abs_path($0), sha256 => sha($0), perl_version => "$]" },
    started_utc => $started,
    completed_utc => strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),
    tests => \@results, test_count => scalar(@results), failures => $failures,
    passed => $failures ? JSON::PP::false : JSON::PP::true,
};
my $summary_path = File::Spec->catfile($out,'report.json');
write_new($summary_path,$json->encode($summary));
$all_files{'report.json'} = sha($summary_path);
write_new(File::Spec->catfile($out,'SHA256SUMS'),
          join('',map { "$all_files{$_}  $_\n" } sort keys %all_files));
print scalar(@results), " tests; $failures failures; $summary_path\n";
exit($failures ? 1 : 0);
