#!/usr/bin/env perl
use strict;
use POSIX qw(strftime);
use Time::HiRes qw(time);
use Sys::Hostname qw(hostname);
use Text::ParseWords qw(shellwords);
use warnings;
use Digest::SHA;
use JSON::PP;
use File::Basename qw(dirname basename);
use File::Spec;
use File::Path qw(make_path);
use File::Copy qw(copy);
use File::Find qw(find);
use Getopt::Long qw(GetOptions);
use IO::Uncompress::Unzip qw($UnzipError);
my $root=dirname(dirname(File::Spec->rel2abs(__FILE__)));
my $old_report_sha='d213b972c0e20552c45e6b2fd880036618090430d4893dbdb4e686d7a46e3b41';
my $old_catalog_sha='e6588cc3686becef542f02a8f42de7483ae8e11fe160a5b08b482ec3eaf86280';
my $hf_commit='8498ed425fa01bbee7d086efdf1a6634c2962aed';
my $hf_blob='946f5fd6e9ba21347654b1b2ca196e86b0f751b3';
my $hf_path='src/transformers/models/qwen4_exp/modeling_qwen4_exp.py';
my $json=JSON::PP->new->canonical->pretty;
sub readfile {my($p)=@_;open my$f,'<:raw',$p or die "$p: $!\n";local$/;my$s=<$f>;close$f or die$!;return $s;}
sub writefile {my($p,$s)=@_;open my$f,'>:raw',$p or die "$p: $!\n";print $f $s or die$!;close$f or die$!;}
sub hashfile {my($p)=@_;open my$f,'<:raw',$p or die "$p: $!\n";my$h=Digest::SHA->new(256);$h->addfile($f);close$f or die$!;return$h->hexdigest;}
sub pinned_origin {my($p)=@_;hashfile($p) eq $old_report_sha or die "original report identity changed\n";my$r=decode_json(readfile($p));my$o=$r->{external_oracle};ref($o) eq'HASH' && $o->{repository} eq'huggingface/transformers' && $o->{commit} eq$hf_commit && $o->{blob} eq$hf_blob && $o->{path} eq$hf_path or die "pinned external origin mismatch\n";return$o;}
sub npy {
 my($bytes)=@_;length($bytes)>=10 && substr($bytes,0,6) eq"\x93NUMPY" or die"not NPY\n";
 my$ver=ord(substr($bytes,6,1));my($off,$len);if($ver==1){$off=10;$len=unpack('v',substr($bytes,8,2));}elsif($ver==2||$ver==3){length($bytes)>=12 or die"short NPY\n";$off=12;$len=unpack('V',substr($bytes,8,4));}else{die"unsupported NPY version\n";}
 length($bytes)>=$off+$len or die"short NPY header\n";my$header=substr($bytes,$off,$len);
 $header=~/'fortran_order'\s*:\s*False/ or die"Fortran-order fixture unsupported\n";
 my($dtype)=$header=~/'descr'\s*:\s*'([^']+)'/;my($shape)=$header=~/'shape'\s*:\s*\(([^)]*)\)/;defined($dtype)&&defined($shape) or die"invalid NPY header\n";
 my@dims=grep{length}map{s/^\s+|\s+$//gr}split(/,/,$shape);@dims or die"scalar NPY unsupported\n";
 my$count=1;for(@dims){/\A\d+\z/ or die"invalid NPY dimension\n";$count*=$_;$count<=134_217_728 or die"oversized fixture\n";}
 my%width=('<u2'=>2,'<f4'=>4,'<u4'=>4,'<f8'=>8,'<i8'=>8,'|i1'=>1,'|u1'=>1);
 exists$width{$dtype} or die"unsupported dtype $dtype\n";my$payload=substr($bytes,$off+$len);
 length($payload)==$count*$width{$dtype} or die"NPY byte/shape mismatch\n";return{dtype=>$dtype,dims=>\@dims,payload=>$payload};
}
sub arrays {
 my($p,$wanted)=@_;my%out;my$zip=IO::Uncompress::Unzip->new($p) or die"unzip: $UnzipError\n";
 for(;;){my$name=$zip->getHeaderInfo->{Name};if($name=~/\A([a-z0-9_]+)\.npy\z/&&$wanted->{$1}){my$key=$1;!exists$out{$key} or die"duplicate NPY member\n";my$bytes='';for(;;){my$n=$zip->read(my$chunk,65536);defined($n)&&$n>=0 or die"unzip read: $UnzipError\n";last unless$n;$bytes.=$chunk;length($bytes)<=536_870_912 or die"oversized member\n";}$out{$key}=npy($bytes);}my$s=$zip->nextStream;die"unzip stream: $UnzipError\n" if$s<0;last unless$s;}
 return\%out;
}
sub export_array {
 my($a,$key,$shape,$type,$dir,$dest)=@_;my$v=$a->{$key} or die"missing $key\n";
 join(',',@{$v->{dims}}) eq$shape or die"shape mismatch $key\n";my$p=$v->{payload};
 if($type eq'f32'){$v->{dtype} eq'<f4' or die"expected FP32 $key\n";}
 elsif($type eq'f64'){$v->{dtype} eq'<f8'||$v->{dtype} eq'<f4' or die"expected floating $key\n";$p=pack('d<*',unpack('f<*',$p)) if$v->{dtype} eq'<f4';}
 elsif($type eq'u16'){$v->{dtype} eq'<u2' or die"expected uint16 $key\n";}
 elsif($type eq'u32'){if($v->{dtype} eq'<i8'){my@ids=unpack('q<*',$p);for(@ids){$_>=0&&$_<=0xffffffff or die"invalid ID\n";}$p=pack('V*',@ids);}else{$v->{dtype} eq'<u4' or die"expected IDs $key\n";}}
 else{die"unknown export dtype\n";}
 writefile("$dir/".($dest//$key).".$type",$p);
}
sub fixture_mode {
 my($path,$tmp)=@_;my$name=basename($path);my($mode,$n);my%spec;
 if($name=~/\Aattention-(\d+)\.npz\z/){($mode,$n)=('--attention',0+$1);%spec=(query=>['24,256','f32'],gate=>['24,256','f32'],key=>["$n,2,256",'f32'],value=>["$n,2,256",'f32'],reference=>['24,256','f64'],quant_reference=>['24,256','f64']);}
 elsif($name=~/\Aindex-(\d+)\.npz\z/){($mode,$n)=('--index',0+$1);my$nv=int($n/4);my$k=$nv<512?$nv:512;%spec=(local_iq=>['4,128','f32'],hf_iq=>['4,128','f32'],local_pool_bits=>["$nv,128",'u16'],hf_pool=>["$nv,128",'f32'],hf_score=>["$nv",'f32'],raw_score=>["$nv",'f32'],hf_ids=>["$k",'u32']);}
 elsif($name=~/\Asparse-output-(\d+)\.npz\z/){($mode,$n)=('--sparse',0+$1);my$nv=int($n/4);my$nk=4*($nv<512?$nv:512)+$n%4;%spec=(local_token_ids=>["$nk",'u32'],hf_token_ids=>["$nk",'u32'],query=>['24,256','f32'],gate=>['24,256','f32'],external_selection_output=>['24,256','f64']);}
 elsif($name eq'hf-bf16-boundary.npz'){($mode,$n)=('--bf16',undef);%spec=(mean_before_cast=>['5,128','f32'],mean_after_cast=>['5,128','f32'],normalized=>['5,128','f32']);}
 elsif($name=~/\Acutoff-(?:adjacent-low-id-[ab]|equal|zero)\.npz\z/){($mode,$n)=('--cutoff',517);%spec=(raw=>['517','f32'],scaled=>['517','f32'],hf_ids=>['512','u32']);}
 elsif($name=~/\Afull-chain-cutoff-[01]\.npz\z/){($mode,$n)=('--full-chain',513);%spec=(raw_score=>['513','f32','raw'],scaled_score=>['513','f32','scaled'],hf_score=>['513','f32'],hf_ids=>['512','u32'],raw_keys=>['2052,128','f32'],index_query=>['4,128','f32'],q_gamma=>['128','f32'],k_gamma=>['128','f32'],local_pool_bits=>['513,128','u16'],hf_iq=>['4,128','f32'],hf_pool=>['513,128','f32']);}
 else{die"unrecognized fixture $name\n";}
 my%want=map{$_=>1}keys%spec;my$a=arrays($path,\%want);for my$key(sort keys%spec){export_array($a,$key,$spec{$key}[0],$spec{$key}[1],$tmp,$spec{$key}[2]);}
 if($mode eq'--cutoff'){copy("$tmp/scaled.f32","$tmp/hf_score.f32") or die$!;}
 return defined($n)?($mode,$tmp,$n):($mode,$tmp);
}
my($full,$out,$archive,$goldens,$binary,$pack,$gpu,$device)=(0,undef,undef,"$root/validation/issue11/goldens","$root/kern/qsa_oracle",undef,undef,0);
Getopt::Long::Configure(qw(no_auto_abbrev no_ignore_case));
GetOptions('full'=>\$full,'out=s'=>\$out,'archive=s'=>\$archive,'goldens=s'=>\$goldens,'binary=s'=>\$binary,'pack-goldens=s'=>\$pack,'gpu=s'=>\$gpu,'device=i'=>\$device) or die"invalid options\n";
@ARGV==0 or die"unexpected arguments\n";
my@small=(qw(cutoff-adjacent-low-id-a.npz cutoff-adjacent-low-id-b.npz cutoff-equal.npz cutoff-zero.npz full-chain-cutoff-0.npz full-chain-cutoff-1.npz hf-bf16-boundary.npz index-2052.npz),map{"attention-$_.npz"}(1,2,3,4,1025));
sub catalog {
 my($p)=@_;hashfile($p) eq $old_catalog_sha or die"original artifact catalog pin changed\n";my$a=decode_json(readfile($p));ref($a) eq'ARRAY' or die"artifact catalog must be an array\n";my%m;
 for my$r(@$a){ref($r) eq'HASH'&&defined$r->{name}&&defined$r->{sha256} or die"invalid artifact entry\n";my$name=$r->{name};$name=~/\A[a-zA-Z0-9_.-]+\z/ && $r->{sha256}=~/\A[0-9a-f]{64}\z/ or die"unsafe artifact declaration\n";!exists$m{$name} or die"duplicate artifact name\n";$m{$name}=$r;}
 return\%m;
}
sub verify_artifact {my($dir,$row)=@_;my$p="$dir/$row->{name}";-f$p or die"missing external artifact $p\n";hashfile($p) eq$row->{sha256} or die"external artifact hash changed: $p\n";(!defined$row->{bytes}||-s$p==$row->{bytes}) or die"external artifact size changed: $p\n";return$p;}
defined$out or die"usage: perl kern/qsa_oracle.pl [--full] --out NEWDIR [--archive ORIGINALDIR] [--goldens DIR] [--binary FILE]\n";
$out=File::Spec->rel2abs($out);!-e$out && !-l$out or die"output directory must be new\n";make_path(dirname($out));mkdir$out or die"$out: $!\n";
if(defined$pack){
 my$catfile="$root/validation/issue11/artifacts.json";my$c=catalog($catfile);my$origin=pinned_origin("$pack/report.json");my@rows;
 for my$name(@small){my$r=$c->{$name} or die"original catalog lacks $name\n";my$p=verify_artifact($pack,$r);copy($p,"$out/$name") or die$!;push@rows,{name=>$name,bytes=>0+(-s$p),sha256=>$r->{sha256}};}
 copy("$pack/report.json","$out/original-report.json") or die$!;copy($catfile,"$out/original-artifacts.json") or die$!;
 writefile("$out/manifest.json",$json->encode({schema=>'flash-sass.qsa-goldens.v1',original_report_sha256=>$old_report_sha,original_artifacts_sha256=>hashfile($catfile),external_oracle=>$origin,artifacts=>\@rows}));print"Packed pinned external goldens: $out\n";exit 0;
}
$binary=File::Spec->rel2abs($binary);$goldens=File::Spec->rel2abs($goldens);
sub command_metadata {
 my(@cmd)=@_;open my$f,'-|',@cmd or die "cannot run $cmd[0]: $!\n";
 my$s=do{local$/;<$f>}//'';my$closed=close$f;my$raw=$?;
 return {command=>\@cmd,exit_code=>($raw&127)?128+($raw&127):$raw>>8,ok=>$closed?JSON::PP::true:JSON::PP::false,output=>$s};
}
sub host_metadata {
 my@cc=shellwords($ENV{CC}//'cc');@cc or die "empty compiler command\n";
 my$boot=readfile('/proc/sys/kernel/random/boot_id');$boot=~s/\s+\z//;
 return {utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),unix_time=>0+time(),hostname=>hostname(),boot_id=>$boot,pid=>0+$$,
 kernel=>command_metadata('uname','-srmo'),compiler=>command_metadata(@cc,'--version'),
 git_head=>command_metadata('git','-C',$root,'rev-parse','HEAD'),
 git_status=>command_metadata('git','-C',$root,'status','--porcelain=v1','--untracked-files=no')};
}

sub system_logged {my($log,@cmd)=@_;my$pid=fork();defined$pid or die"fork\n";if(!$pid){open STDOUT,'>',$log or die$!;open STDERR,'>&',\*STDOUT or die$!;exec{$cmd[0]}@cmd;die"exec $cmd[0]: $!\n";}waitpid($pid,0);return $?;}
my$default_binary=File::Spec->rel2abs("$root/kern/qsa_oracle");
my$build_mode=$binary eq$default_binary?'rebuilt_default':'provided_binary';
my@build_command;
if($build_mode eq'rebuilt_default'){
 @build_command=('make','-B','-C',"$root/kern",'qsa_oracle');
 my$rc=system_logged("$out/build.log",@build_command);
 $rc==0&&-x$binary or die "native oracle build failed (see $out/build.log)\n";
}else{-x$binary or die "provided oracle binary is not executable\n";}
my$started=host_metadata();
$started->{native_binary_sha256}=hashfile($binary);

if(defined$gpu){($ENV{QSA_ORACLE_GPU_GUARDED}//'') eq'1' or die"GPU invocation must run through qsa_oracle_gpu_gate.sh\n";-f$gpu or die"missing cubin\n";$gpu=File::Spec->rel2abs($gpu);$device>=0 or die"invalid GPU ordinal\n";}
my$manifest=decode_json(readfile("$goldens/manifest.json"));ref($manifest) eq'HASH'&&$manifest->{schema} eq'flash-sass.qsa-goldens.v1' or die"wrong golden manifest schema\n";
$manifest->{original_report_sha256} eq$old_report_sha or die"golden report pin changed\n";my$origin=pinned_origin("$goldens/original-report.json");hashfile("$goldens/original-artifacts.json") eq$manifest->{original_artifacts_sha256} or die"golden catalog identity changed\n";
my$cat=catalog("$goldens/original-artifacts.json");my%want=map{$_=>1}@small;my%seen;my@frozen;
for my$r(@{$manifest->{artifacts}}){$want{$r->{name}}&&!$seen{$r->{name}}++ or die"unexpected/duplicate golden\n";my$original=$cat->{$r->{name}} or die"golden not in original catalog\n";$r->{sha256} eq$original->{sha256} or die"golden hash differs from original catalog\n";push@frozen,{path=>verify_artifact($goldens,$r),sha256=>$r->{sha256},name=>$r->{name}};}
keys(%seen)==@small or die"incomplete external golden suite\n";
if(defined$archive){$archive=File::Spec->rel2abs($archive);pinned_origin("$archive/report.json");@frozen=();for my$name(sort keys%$cat){next unless$name=~/\A(?:attention-\d+|index-\d+|sparse-output-\d+|cutoff-(?:adjacent-low-id-[ab]|equal|zero)|full-chain-cutoff-[01]|hf-bf16-boundary)\.npz\z/;push@frozen,{path=>verify_artifact($archive,$cat->{$name}),sha256=>$cat->{$name}{sha256},name=>$name};}}
make_path("$out/checks","$out/generated","$out/frozen");my@results;my$failed=0;my$seq=0;
my$report={schema=>'flash-sass.native-qsa-oracle.v1',status=>'RUNNING',scope=>'Independent native mathematical transcription plus frozen pinned external probes; local candidate scalar routines are never the external oracle',checkpoint_conformance=>JSON::PP::false,external_oracle_capture=>$origin,original_report_sha256=>$old_report_sha,goldens_manifest_sha256=>hashfile("$goldens/manifest.json"),native_binary_sha256=>hashfile($binary),full_matrix=>$full?JSON::PP::true:JSON::PP::false,limits=>{score_normalized_rms=>.005,attention_abs=>3e-5,attention_normalized_rms=>3e-5,kv_attention_abs=>.025,kv_attention_normalized_rms=>.025,external_score_math_abs=>3e-5,external_score_math_normalized_rms=>3e-5,teacher_forced_logit_abs=>.10,teacher_forced_logit_normalized_rms=>.01},generator=>{algorithm=>'xorshift64star: shifts12,25,27; multiplier2685821657736338717; upper24bits mapped to uniform[-scale,scale)',index_seed=>'100000+visible_length',sparse_seed=>'200000+visible_length',attention_seed=>'300000+key_count'},sources_sha256=>{},results=>\@results};
for my$p(qw(kern/qsa_oracle.c kern/qsa_oracle.pl kern/qsa_oracle_bridge.c kern/qsa_oracle_gpu.c kern/qsa_oracle_gpu.h kern/qsa_oracle_gpu_gate.sh kern/Makefile ref/qsa.c ref/qsa.h ref/attn.c ref/attn.h ref/fmath.c ref/fmath.h)){ $report->{sources_sha256}{$p}=hashfile("$root/$p"); }
$report->{build_mode}=$build_mode;$report->{build_command}=\@build_command;
$report->{start}=$started;
$report->{build_log_sha256}=hashfile("$out/build.log") if -f"$out/build.log";
if(defined$gpu){$report->{gpu_request}={cubin=>$gpu,cubin_sha256=>hashfile($gpu),ordinal=>$device};}
writefile("$out/report.json",$json->encode($report));
sub execute {
 my($expected,$label,@args)=@_;my$stem=sprintf('%03d',$seq++);
 my$stdout="$out/checks/$stem.json";my$stderr="$out/checks/$stem.stderr";my@cmd=($binary,@args);
 my$cpu_only=$args[0] eq'--self-test'||$args[0] eq'--negative-score'||$args[0] eq'--bf16';
 my$target=defined($gpu)&&!$cpu_only;
 my$image_before=$target?hashfile($gpu):undef;
 my$pid=fork();defined$pid or die "fork\n";
 if(!$pid){
  open STDOUT,'>',$stdout or die$!;open STDERR,'>',$stderr or die$!;
  if($target){$ENV{GPU_CUBIN}=$gpu;$ENV{GPU_DEVICE}=$device;}else{delete$ENV{GPU_CUBIN};delete$ENV{GPU_DEVICE};}
  if(!$cpu_only && $args[0]!~/\A--generate-/){$ENV{QSA_ORACLE_CAPTURE_DIR}="$args[1]/candidate";}
  else{delete$ENV{QSA_ORACLE_CAPTURE_DIR};}
  exec{$binary}@cmd;die "exec: $!\n";
 }
 waitpid($pid,0)==$pid or die"waitpid: $!\n";
 my$raw=$?;my$code=($raw&127)?128+($raw&127):$raw>>8;
 my($r,$parse_error);my$parsed=eval{$r=decode_json(readfile($stdout));1};$parse_error="$@" unless$parsed;
 my$ok=$parsed&&ref($r) eq'HASH'&&$code==$expected;
 $ok&&=defined($r->{status})&&$r->{status} eq($expected?'FAIL_NATIVE':'PASS_NATIVE')&&defined($r->{failures});
 $ok&&=($expected?$r->{failures}>0:$r->{failures}==0);
 if($expected==3){
  my@bad=ref($r) eq'HASH'&&ref($r->{metrics}) eq'ARRAY'?(grep{ref($_) eq'HASH'&&defined($_->{name})&&$_->{name} eq'score_gate_must_reject'}@{$r->{metrics}}):();
  $ok&&=@bad==1&&$bad[0]{enforced}&&!$bad[0]{pass}&&$bad[0]{normalized_rms}>.005;
 }
 if(!$parsed||ref($r) ne'HASH'){$r={status=>'INVALID_CHILD_REPORT',error=>$parse_error//'JSON result is not an object'};}
 my$image_after=$target?hashfile($gpu):undef;
 if($target){
  my$empty_index=$args[0] eq'--index'&&$args[2]==1;
  $ok&&=($r->{candidate_backend}//'') eq'SM52'&&defined($r->{gpu_launches});
  $ok&&=$empty_index?($r->{gpu_launches}==0):($r->{gpu_launches}>0);
  $r->{gpu_applicable}=JSON::PP::false,$r->{gpu_skip_reason}='frozen index has zero complete blocks; generated index1 covers target normalization' if$empty_index;
  $ok&&=$image_before eq$image_after&&$image_before eq$report->{gpu_request}{cubin_sha256};
 }
 my$row={label=>$label,command=>\@cmd,exit_code=>$code,expected_exit_code=>$expected,
  passes=>$ok?JSON::PP::true:JSON::PP::false,result=>$r,
  stdout_sha256=>hashfile($stdout),stderr_sha256=>hashfile($stderr)};
 if($target){$row->{cubin_sha256_before}=$image_before;$row->{cubin_sha256_after}=$image_after;}
 push@results,$row;$failed||=!$ok;writefile("$out/report.json",$json->encode($report));
 print "$label: ".($ok?'PASS':'FAIL')." (exit $code)\n";return$row;
}
execute(0,'host arithmetic/domain gates','--self-test');execute(3,'negative control rejected by actual score NRMS gate','--negative-score');
for my$f(@frozen){my$tmp="$out/frozen/$f->{name}";mkdir($tmp) or die$!;hashfile($f->{path}) eq$f->{sha256} or die"fixture changed before execution\n";my@args=fixture_mode($f->{path},$tmp);my$row=execute(0,"frozen/$f->{name}",@args);$row->{fixture_sha256}=$f->{sha256};}
my@lengths=(1,2047..2052,3100..3103,8000..8003);push@lengths,(65000..65003,262000..262003) if$full;
for my$n(@lengths){execute(0,"generated/index-and-sparse-$n",'--generate-index',"$out/generated/index-$n",$n);}
my@attention=(1..4,1023..1027,2047..2052);push@attention,(8191..8193) if$full;
for my$n(@attention){execute(0,"generated/attention-$n",'--generate-attention',"$out/generated/attention-$n",$n);}
$report->{status}='VERIFYING_ARTIFACTS';
$report->{coverage}={frozen_fixtures=>scalar@frozen,index_and_sparse_lengths=>\@lengths,same_ID_attention_lengths=>\@attention,negative_control_exit_expected=>3};
writefile("$out/report.json",$json->encode($report));
my@artifacts;
find({no_chdir=>1,wanted=>sub{
 return unless-f$_;my$p=$File::Find::name;
 my$name=File::Spec->abs2rel($p,$out);
 return if$name eq'report.json'||$name eq'artifacts.json';
 push@artifacts,{name=>$name,bytes=>0+(-s$p),sha256=>hashfile($p)};
}},$out);
@artifacts=sort{$a->{name}cmp$b->{name}}@artifacts;
writefile("$out/artifacts.json",$json->encode(\@artifacts));
$report->{artifacts_manifest_sha256}=hashfile("$out/artifacts.json");
$report->{artifacts_count}=scalar@artifacts;
my$finished=host_metadata();
$finished->{native_binary_sha256}=-f$binary?hashfile($binary):undef;
my$consistent=defined($finished->{native_binary_sha256})&&$finished->{native_binary_sha256} eq$started->{native_binary_sha256};
$consistent&&=$finished->{hostname} eq$started->{hostname}&&$finished->{boot_id} eq$started->{boot_id};
for my$key(qw(git_head compiler kernel)){
 $consistent&&=$started->{$key}{ok}&&$finished->{$key}{ok}&&$started->{$key}{output} eq$finished->{$key}{output};
}
my(%after,@changed);
for my$p(sort keys%{$report->{sources_sha256}}){
 $after{$p}=-f"$root/$p"?hashfile("$root/$p"):undef;
 push@changed,$p unless defined($after{$p})&&$after{$p} eq$report->{sources_sha256}{$p};
}
$consistent&&=!@changed;
$report->{sources_sha256_after}=\%after;$report->{changed_sources}=\@changed;
if(defined$gpu){
 my$end_hash=-f$gpu?hashfile($gpu):undef;
 $report->{gpu_request}{cubin_sha256_after}=$end_hash;
 $consistent&&=defined($end_hash)&&$end_hash eq$report->{gpu_request}{cubin_sha256};
}
$report->{end}=$finished;
$report->{elapsed_seconds}=$finished->{unix_time}-$started->{unix_time};
$report->{provenance_consistent}=$consistent?JSON::PP::true:JSON::PP::false;
$report->{gpu_executed}=scalar(grep{($_->{result}{gpu_launches}//0)>0}@results)?JSON::PP::true:JSON::PP::false;
$failed||=!$consistent;
$report->{status}=$failed?'FAIL_NATIVE_SUITE':defined($gpu)?'PASS_TARGET_SYNTHETIC':'PASS_NATIVE_SYNTHETIC';
writefile("$out/report.json",$json->encode($report));
print "$report->{status}: $out/report.json\n";exit($failed?3:0);
