#!/usr/bin/env perl
use strict;
use warnings;
use Test::More;
use FindBin;
use File::Temp qw(tempdir);
use lib $FindBin::Bin;
use ResidentConfig qw(parse_config validation_plan file_fnv);
my $sha = 'e0ce409d7f23f8ee924cf312c77eac4fbce89bfca6d180c6328f55b12e771eb2';
my $cubin = '/tmp/issue15/candidate-die.cubin';
my $line = 'config rev=edc66d9 kv_host=0 ple_disk=1 maxt=230000 layers=48/48 nblk=512 dies=6 '
    . 'attn_dies=1,4,5 split=112,97,81,116,97,97 attn_map=1,1,1,1,4,4,4,4,5,5,5,5 '
    . 'img_rows=4096 two_position=0 prefill_chunk=0 cubin=/tmp/baseline/original-die.cubin '
    . 'cubin_fnv=669dbb2e43186b1a';
sub rejects {
    my ($label, $code, $pattern) = @_;
    my $ok = eval { $code->(); 1 };
    ok(!$ok, $label);
    like($@, $pattern, "$label identifies the reason");
}
my $config = parse_config($line);
is_deeply(parse_config("$line\n"), $config, 'one terminal LF is accepted');
my $plan = validation_plan($config, $sha, $cubin);
is_deeply($plan->{baseline}, $config, 'baseline fields are preserved');
is_deeply($plan->{restore}, {
    FLASH_ENGINE_MAXT=>'230000', FLASH_MAXIMG=>'4096',
    FLASH_CUBIN=>'/tmp/baseline/original-die.cubin', FLASH_TRACE=>'0', FLASH_TWO_POSITION=>'0',
}, 'restore plan contains precisely the five changed effective knobs');
like($plan->{trace_restore_basis}, qr/legacy engine .* ignores FLASH_TRACE/,
    'legacy trace exception records its semantic basis');
is_deeply($plan->{candidates}, [
    {FLASH_ENGINE_MAXT=>'512', FLASH_MAXIMG=>'1', FLASH_CUBIN=>$cubin, FLASH_TRACE=>'4', FLASH_TWO_POSITION=>'0'},
    {FLASH_ENGINE_MAXT=>'512', FLASH_MAXIMG=>'1', FLASH_CUBIN=>$cubin, FLASH_TRACE=>'4', FLASH_TWO_POSITION=>'1'},
], 'candidates change only the requested five knobs');
my %known = (%$config, rev=>'abc1234', trace_tokens=>'16', two_position=>'1', cubin=>'relative/baseline.cubin');
my $known_plan = validation_plan(\%known, 'a' x 64, $cubin);
is($known_plan->{restore}{FLASH_TRACE}, '16', 'known effective trace is restored');
is($known_plan->{restore}{FLASH_TWO_POSITION}, '1', 'baseline pair mode is restored');
is($known_plan->{restore}{FLASH_CUBIN}, 'relative/baseline.cubin', 'safe relative baseline path retained');
$known{trace_tokens} = '0';
is(validation_plan(\%known, 'a' x 64, $cubin)->{restore}{FLASH_TRACE}, '0', 'known disabled trace accepted');
rejects('unknown revision without trace',
    sub {validation_plan({%$config, rev=>'abc1234'}, $sha, $cubin)}, qr/lacks effective trace_tokens/);
rejects('legacy revision with different binary',
    sub {validation_plan($config, 'a' x 64, $cubin)}, qr/lacks effective trace_tokens/);
rejects('malformed token', sub {parse_config("$line nonsense")}, qr/Malformed config token/);
rejects('missing response prefix', sub {my $s=$line; $s=~s/^config //; parse_config($s)}, qr/response prefix/);
rejects('duplicate key', sub {parse_config("$line maxt=512")}, qr/Duplicate config key maxt/);
rejects('missing required key', sub {my $s=$line; $s=~s/ nblk=512//; parse_config($s)}, qr/Missing required config key nblk/);
rejects('embedded control', sub {parse_config("$line\ttrace_tokens=4")}, qr/control/);
rejects('embedded equals in value', sub {parse_config("$line extra=/tmp/a=b")}, qr/Malformed config token/);
rejects('relative candidate path', sub {validation_plan($config, $sha, 'candidate.cubin')}, qr/absolute path/);
rejects('unsafe candidate path', sub {validation_plan($config, $sha, "/tmp/a\nb")}, qr/safe protocol token/);
rejects('invalid known trace', sub {validation_plan({%$config, trace_tokens=>'-1'}, $sha, $cubin)}, qr/trace_tokens must be/);
rejects('excessive known trace', sub {validation_plan({%$config, trace_tokens=>'257'}, $sha, $cubin)}, qr/exceeds/);
for my $case (
    ['layers','47/48',qr/Unsupported resident layout/], ['nblk','256',qr/Unsupported resident layout/],
    ['dies','5',qr/Unsupported resident layout/], ['kv_host','1',qr/Unsupported resident layout/],
    ['ple_disk','0',qr/Unsupported resident layout/], ['prefill_chunk','1',qr/Unsupported resident layout/],
    ['maxt','511',qr/at least 512/], ['maxt','2147483648',qr/fit an int/],
    ['img_rows','0',qr/must be positive/], ['two_position','2',qr/must be 0 or 1/],
    ['split','112,97',qr/six entries/], ['attn_dies','1,1,5',qr/Invalid attn_dies/],
    ['attn_map','0,1,1,1,4,4,4,4,5,5,5,5',qr/unlisted die/],
) {
    my ($key,$value,$pattern)=@$case;
    rejects("unsupported $key", sub {validation_plan({%$config, $key=>$value},$sha,$cubin)}, $pattern);
}
my $dir = tempdir(CLEANUP=>1);
for my $vector (['','cbf29ce484222325'], ['a','af63dc4c8601ec8c'], ['hello','a430d84680aabd0b']) {
    open my $f,'>:raw',"$dir/fnv" or die $!;
    print {$f} $vector->[0]; close $f or die $!;
    is(file_fnv("$dir/fnv"),$vector->[1],"FNV-1a known vector '$vector->[0]'");
}
done_testing();
