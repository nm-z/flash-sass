#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use BenchManifest;
use Test::More;
use File::Temp qw(tempdir);
use JSON::PP;
my $tmp=tempdir(CLEANUP=>1);
sub write_text {open my $f,'>:raw',$_[0] or die $!;print {$f} $_[1] or die $!;close $f or die $!;}
sub clone {JSON::PP->new->decode(JSON::PP->new->encode($_[0]))}
sub rejects {my($code,$pattern,$name)=@_;my $ok=eval{$code->();1};ok(!$ok,$name);like($@,$pattern,"$name reason") unless $ok;}
write_text("$tmp/file","isolated CPU fixture\n");
my $id=BenchManifest::identity("$tmp/file");
sub conversion_fixture {
    {argv=>['/fixture/converter','--input','/fixture/source'],tool=>clone($id),
        exit=>0,elapsed_ns=>1,input_sha256=>$id->{sha256},output_sha256=>$id->{sha256}};
}
sub weights_fixture {
    {model=>{name=>'isolated fixture',sha256=>$id->{sha256}},files=>[clone($id)],
        tensors=>[{name=>'tensor.fixture',file_sha256=>$id->{sha256},conversion_sha256=>$id->{sha256}}],
        conversion=>conversion_fixture()};
}
sub launch_fixture {
    my %env=map {$_=>undef} @BenchManifest::ENV_KEYS;
    @env{qw(FLASH_MAXT FLASH_ENGINE_MAXT FLASH_NBLK FLASH_MAXIMG FLASH_PLE_DISK FLASH_PLE_OFF FLASH_CUBIN
        FLASH_KV_HOST CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER SPLIT_SHARES ATTN_DIE ATTN_MAP FLASH_SOCK)}
        =('8192','8192','512','1','1','192','/fixture/die.cubin','1','0,1,2,3,4,5','PCI_BUS_ID','1,1,1,1,1,1','1','1,1,1,1','/fixture/host.sock');
    my $host={pid=>101,argv=>['/fixture/flashcore','/fixture/model'],launch_environment=>\%env,
        engine_reload_commands=>[],limitations=>[],ple_cache_conditions=>{mode=>'disk',
            disk_identity=>'fixture-device-and-file-hash',page_cache_policy=>'retained warm cache after fixed preparation',
            engine_cache_policy=>'fixed staging allocation',staging_allocation_bytes=>4096,preparation_argv=>[]}};
    my $vram=clone($host);$vram->{pid}=102;$vram->{launch_environment}{FLASH_KV_HOST}='0';
    $vram->{launch_environment}{FLASH_SOCK}='/fixture/vram.sock';return {host=>$host,vram=>$vram};
}
subtest 'independent file identities and exact FNV' => sub {
    is($id->{bytes},length("isolated CPU fixture\n"),'size');
    like($id->{sha256},qr/\A[0-9a-f]{64}\z/,'SHA256');
    like("$id->{mtime_ns}",qr/\A[0-9]+\z/,'integer nanoseconds');
    is_deeply(BenchManifest::verified_identity($id,'fixture'),$id,'independent verification');
    my $wrong=clone($id);$wrong->{sha256}='0'x64;
    rejects(sub{BenchManifest::verified_identity($wrong,'fixture')},qr/hash mismatch/,'bad hash rejected');
    write_text("$tmp/fnv",'');is(BenchManifest::fnv64("$tmp/fnv"),'cbf29ce484222325','FNV empty vector');
    write_text("$tmp/fnv",'a');is(BenchManifest::fnv64("$tmp/fnv"),'af63dc4c8601ec8c','FNV one byte');
    write_text("$tmp/fnv",'foobar');is(BenchManifest::fnv64("$tmp/fnv"),'85944171f73967e8','FNV multiple bytes');
};
subtest 'environment preserves unset and ignores no unknown setting' => sub {
    is_deeply(BenchManifest::environment({FLASH_NBLK=>'512'}),{FLASH_NBLK=>'512'},'known setting');
    rejects(sub{BenchManifest::environment({AWS_SECRET_ACCESS_KEY=>'x'})},qr/unrecognized launch environment/,'unrelated credential rejected');
    rejects(sub{BenchManifest::environment({FLASH_API_KEY=>'x'})},qr/unrecognized launch environment/,'secret-like prefix extension rejected');
    my $r=launch_fixture();$r->{host}{launch_environment}{PROG_DUP}='';
    rejects(sub{BenchManifest::matched_launch($r)},qr/unexplained launch/,'empty and unset are different');
};
subtest 'complete weight and conversion identities' => sub {
    my $w=weights_fixture();ok(BenchManifest::validate_weights($w),'complete CPU schema');
    my $bad=clone($w);$bad->{files}=[];
    rejects(sub{BenchManifest::validate_weights($bad)},qr/nonempty array/,'empty files rejected');
    $bad=clone($w);$bad->{tensors}[0]{file_sha256}='2'x64;
    rejects(sub{BenchManifest::validate_weights($bad)},qr/tensor file absent/,'unidentified tensor file rejected');
    $bad=clone($w);$bad->{conversion}{exit}=1;
    rejects(sub{BenchManifest::validate_weights($bad)},qr/conversion failed/,'failed conversion rejected');
    $bad=clone($w);$bad->{conversion}{input_sha256}='3'x64;
    rejects(sub{BenchManifest::validate_weights($bad)},qr/conversion input absent/,'unverified conversion input rejected');
};
subtest 'matched launch and explicit PLE conditions' => sub {
    my $r=launch_fixture();ok(BenchManifest::matched_launch($r),'only placement/socket vary');
    my $bad=clone($r);$bad->{vram}{launch_environment}{FLASH_MAXT}='262144';
    rejects(sub{BenchManifest::matched_launch($bad)},qr/unexplained launch/,'capacity mismatch rejected');
    $bad=clone($r);$bad->{vram}{ple_cache_conditions}{staging_allocation_bytes}=8192;
    rejects(sub{BenchManifest::matched_launch($bad)},qr/unexplained launch/,'staging mismatch rejected');
    $bad=clone($r);delete $bad->{host}{ple_cache_conditions};
    rejects(sub{BenchManifest::matched_launch($bad)},qr/ple_cache_conditions must be an object/,'missing cache record');
    $bad=clone($r);$bad->{host}{ple_cache_conditions}{page_cache_policy}='unknown';
    rejects(sub{BenchManifest::matched_launch($bad)},qr/unknown PLE/,'unknown cache policy');
    $bad=clone($r);$bad->{host}{engine_reload_commands}=['reload path FLASH_NBLK=512'];
    rejects(sub{BenchManifest::matched_launch($bad)},qr/nonempty array/,'ambiguous shell-string reload');
};
subtest 'build records bind source flags and every output role' => sub {
    my %art=map {$_=>clone($id)}qw(executable library cubin);my $sha='a'x40;
    my $b={source_sha=>$sha,source_dirty_sha256=>'d'x64,conversion=>conversion_fixture(),steps=>[
        {kind=>'compiler',argv=>['/fixture/cc','-O3','fixture.c'],flags=>['-O3'],tool=>clone($id),exit=>0,elapsed_ns=>1,
            outputs=>[{role=>'executable',sha256=>$id->{sha256}},{role=>'library',sha256=>$id->{sha256}}]},
        {kind=>'assembler',argv=>['/fixture/as','fixture.s'],flags=>[],tool=>clone($id),exit=>0,elapsed_ns=>1,
            outputs=>[{role=>'cubin',sha256=>$id->{sha256}}]}]};
    ok(BenchManifest::validate_build($b,$sha,\%art,'d'x64),'complete CPU build schema');
    my $bad=clone($b);$bad->{source_sha}='b'x40;
    rejects(sub{BenchManifest::validate_build($bad,$sha,\%art)},qr/build source mismatch/,'wrong source');
    rejects(sub{BenchManifest::validate_build($b,$sha,\%art,'e'x64)},qr/build dirty source mismatch/,'different source patch');
    $bad=clone($b);$bad->{steps}[0]{outputs}=[{role=>'executable',sha256=>$id->{sha256}}];
    rejects(sub{BenchManifest::validate_build($bad,$sha,\%art)},qr/build outputs must identify/,'missing library output');
    $bad=clone($b);$bad->{steps}[0]{flags}=['-march=unrecorded'];
    rejects(sub{BenchManifest::validate_build($bad,$sha,\%art)},qr/flag absent/,'flag absent from argv');
};
subtest 'runtime revisions capabilities and loaded cubin' => sub {
    my $p={schema=>'issue14.provenance.v1',source=>{head=>'a'x40,dirty=>JSON::PP::false},
        artifacts=>{cubin=>{%$id,fnv64=>'abc123'}},processes=>{host=>{cwd=>$tmp},vram=>{cwd=>$tmp}}};
    my $host={kv_host=>'1',layers=>'48/48',nblk=>'512',maxt=>'8192',dies=>'6',
        requestmeta_schema=>'1',tensor_schema=>'1',rev=>'aaaaaaa',cubin_fnv=>'abc123',
        additional_future_capability=>'2'};
    my $vram=clone($host);$vram->{kv_host}='0';my $c={host=>$host,vram=>$vram};
    ok(BenchManifest::validate_runtime($p,$c),'equal future capability keys tolerated');
    my $bad=clone($c);$bad->{vram}{additional_future_capability}='3';
    rejects(sub{BenchManifest::validate_runtime($p,$bad)},qr/unexplained runtime/,'unknown differing key');
    $bad=clone($c);delete $bad->{host}{requestmeta_schema};
    rejects(sub{BenchManifest::validate_runtime($p,$bad)},qr/missing runtime field/,'missing lifecycle capability');
    $bad=clone($c);$bad->{host}{cubin_fnv}='different';
    rejects(sub{BenchManifest::validate_runtime($p,$bad)},qr/loaded cubin identity differs/,'wrong loaded cubin');
    $bad=clone($c);$bad->{host}{rev}='bbbbbbb';
    rejects(sub{BenchManifest::validate_runtime($p,$bad)},qr/runtime revision/,'wrong loaded revision');
};
subtest 'actual tensor inventory cannot omit or invent loaded entries' => sub {
    my $w=weights_fixture();my $m={schema=>'1',dies=>'1'};
    my @rows=('arena die=0 physical_die=2 bytes=64 tensors=1',
        'tensor die=0 name=tensor.fixture offset=0 bytes=64 type=2 rows=1 K=32 stride=64 experts=0');
    my $r=BenchManifest::tensor_inventory(\@rows,$m,$w);
    is($r->{arenas}[0]{physical_die},2,'real physical die retained');
    is(scalar(@{$r->{tensors}}),1,'complete actual inventory');
    rejects(sub{BenchManifest::tensor_inventory([$rows[0]],$m,$w)},qr/incomplete loaded/,'missing live tensor');
    my $bad=$rows[1];$bad=~s/tensor.fixture/undeclared/;
    rejects(sub{BenchManifest::tensor_inventory([$rows[0],$bad],$m,$w)},qr/undeclared loaded tensor/,'model manifest omits live tensor');
    $bad=$rows[1];$bad=~s/bytes=64/bytes=65/;
    rejects(sub{BenchManifest::tensor_inventory([$rows[0],$bad],$m,$w)},qr/outside loaded arena/,'invalid bounds');
};
subtest 'process evidence cannot be asserted when absent' => sub {
    my $r=launch_fixture();$r->{host}{pid}=999999999;
    my %art=map {$_=>clone($id)}qw(executable library cubin);
    rejects(sub{BenchManifest::process_identity($r,\%art)},qr/unresolvable identity/,'missing process rejected');
};
done_testing;
