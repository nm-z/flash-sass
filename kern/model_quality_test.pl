#!/usr/bin/env perl
use strict;use warnings;use File::Temp qw(tempdir);use File::Path qw(make_path);use JSON::PP;use Test::More;use FindBin;
my $tool=$ARGV[0]//"$FindBin::Bin/model_quality";my $tmp=tempdir(CLEANUP=>1);my $seq=0;
sub put {my($p,$s)=@_;open my $f,'>:raw',$p or die "$p: $!";print $f $s;close $f or die $!;}
sub get {my($p)=@_;open my $f,'<:raw',$p or die "$p: $!";local $/;return <$f>;}
sub run {my @args=@_;my $pid=fork();die 'fork' unless defined $pid;if(!$pid){open STDOUT,'>',"$tmp/stdout" or die $!;open STDERR,'>',"$tmp/stderr" or die $!;exec {$tool} $tool,@args;die "exec: $!";}waitpid($pid,0);die 'child signaled' if $?&127;return $?>>8;}
sub capture {my($kind,$rows,$ids,$vocab)=@_;$ids//=[0,1];my $d="$tmp/raw".++$seq;make_path($d);put("$d/ids",pack('V*',@$ids));put("$d/logits",pack('f<*',map {@$_} @$rows));my @args=('freeze','--kind',$kind,'--ids',"$d/ids",'--logits',"$d/logits",'--vocab-size',scalar @{$rows->[0]},'--out',"$d/frozen");if($vocab){put("$d/vocab",pack('V*',@$vocab));push @args,'--vocab-ids',"$d/vocab";}is(run(@args),0,'freeze succeeds') or diag get("$tmp/stderr");return "$d/frozen";}
sub compare {my($c,$r,@extra)=@_;unlink "$tmp/report.json";my $rc=run('compare','--candidate',$c,'--reference',$r,'--out',"$tmp/report.json",@extra);my $j=-e "$tmp/report.json"?decode_json(get("$tmp/report.json")):undef;return($rc,$j);}
sub near {my($a,$b,$msg)=@_;ok(defined($a)&&abs($a-$b)<1e-6,$msg) or diag "got=".(defined($a)?$a:'null')." expected=$b";}
my $c=capture('candidate',[[log(.6),log(.4)],[log(.5),log(.5)]]);my $r=capture('reference',[[log(.8),log(.2)],[log(.3),log(.7)]]);
my($rc,$j)=compare($c,$r,'--diagnostic');is($rc,0,'diagnostic exit');my $m=$j->{metrics};near($m->{mean_reference_to_candidate_kl},(.8*log(.8/.6)+.2*log(.2/.4)+.3*log(.3/.5)+.7*log(.7/.5))/2,'analytic KL');near($m->{reference_mean_nll},-log(.2),'reference next-token NLL');near($m->{candidate_mean_nll},-log(.4),'candidate next-token NLL');near($m->{top1_agreement},.5,'top1 tie');ok(!defined $j->{quality_pass},'no quality attestation');ok(!$j->{independent_checkpoint_evidence},'no independent evidence');
($rc,$j)=compare($c,$r);is($rc,2,'missing declarations rejected');($rc,$j)=compare($c,$r,'--diagnostic','--max-mean-kl','0');is($rc,3,'limit failure exit');ok(!$j->{declared_limits_pass},'limit fail report');
my $big=capture('candidate',[[1e30,1e30],[1e30,1e30]]);my $zero=capture('reference',[[0,0],[0,0]]);($rc,$j)=compare($big,$zero,'--diagnostic');is($rc,0,'large offset accepted');near($j->{metrics}{candidate_mean_nll},log(2),'large offset normalization');near($j->{metrics}{maximum_absolute_probability_error},0,'large offset probabilities');near($j->{metrics}{centered_logit_rmse},0,'constant offset centering');
my $aligned=capture('candidate',[[2,1],[1,3]],[0,1],[1,0]);my $base=capture('reference',[[1,2],[3,1]]);($rc,$j)=compare($aligned,$base,'--diagnostic');near($j->{metrics}{logit_rmse},0,'vocabulary reordering');
my $tc=capture('candidate',[[3,2,2,0],[0,0,0,0]],[0,1],[0,1,2,3]);my $tr=capture('reference',[[3,2,0,2],[0,0,0,0]],[0,1],[0,1,2,3]);($rc,$j)=compare($tc,$tr,'--diagnostic','--top-k','2');near($j->{metrics}{mean_topk_overlap},1,'cutoff ties by token ID');
my $ic=capture('candidate',[[0,1],[1,0]],[99,1]);my $ir=capture('reference',[[0,1],[1,0]],[99,1]);($rc,$j)=compare($ic,$ir,'--diagnostic');is($rc,2,'invalid first input rejected');
my $different=capture('candidate',[[0,1],[1,0]],[1,0]);my $samevocab=capture('reference',[[0,1],[1,0]],[0,1]);($rc,$j)=compare($different,$samevocab,'--diagnostic');is($rc,2,'different teacher-forced input IDs rejected');
my $nan=unpack('f<',pack('V',0x7fc00000));my $nc=capture('candidate',[[$nan,0],[0,1]]);($rc,$j)=compare($nc,$r,'--diagnostic');is($rc,3,'NaN rejected');ok(!$j->{quality_pass},'NaN never quality pass');
for my $nll (700,710){my $pc=capture('candidate',[[0,-$nll],[0,0]]);my $pr=capture('reference',[[0,-$nll],[0,0]]);($rc,$j)=compare($pc,$pr,'--diagnostic');is($rc,0,"NLL $nll metric exit");near($j->{metrics}{candidate_mean_nll},$nll,"NLL $nll");if($nll==700){ok(defined $j->{metrics}{candidate_perplexity},'representable perplexity retained');}else{ok(!defined $j->{metrics}{candidate_perplexity}&&$j->{metrics}{perplexity_overflow},'overflow explicit');}}
my $fc=capture('candidate',[[0,1],[1,0]]);my $fr=capture('reference',[[0,1],[1,0]]);
for my $d($fc,$fr){my $p="$d/manifest.json";my $v=decode_json(get($p));$v->{metadata}={teacher_forced=>JSON::PP::true,base_model=>{id=>'declared',revision=>'a'x40},tokenizer=>{sha256=>'b'x64},checkpoint=>{files=>[{source_uri=>'declared:model',sha256=>'c'x64}]},implementation=>{name=>'declared',revision=>'d'x40,independent_reference=>JSON::PP::true},command=>['declared-command']};put($p,encode_json($v));}
for my $extra([],['--diagnostic']){($rc,$j)=compare($fc,$fr,@$extra,'--max-rmse','0');is($rc,0,'complete declarations numerical exit');ok($j->{declared_limits_pass}&&!defined($j->{quality_pass})&&!$j->{independent_checkpoint_evidence},'fake producer declarations cannot attest');}
for my $args(['--top-k','0'],['--max-rmse','nan'],['--min-top1-agreement','1.01'],['--max-mean-kl','-1']){($rc,$j)=compare($fc,$fr,@$args);is($rc,2,'invalid CLI rejected');}
my $bytes=get("$fc/logits.f32");substr($bytes,0,1)=chr(ord(substr($bytes,0,1))^1);put("$fc/logits.f32",$bytes);($rc,$j)=compare($fc,$fr,'--diagnostic');is($rc,2,'changed frozen bytes rejected');
my $mixed_c=capture('candidate',[[1e30,1]],[0]);
my $mixed_r=capture('reference',[[1e30,0]],[0]);
($rc,$j)=compare($mixed_c,$mixed_r,'--diagnostic');
is($rc,0,'shared huge-logit regression exits successfully');
is($j->{metrics}{centered_logit_rmse},0.5,'centered RMSE preserves small difference beside a shared huge logit');
near($j->{metrics}{logit_rmse},sqrt(0.5),'shared huge logit ordinary RMSE');
is($j->{metrics}{maximum_absolute_logit_error},1,'shared huge logit maximum error');
my $mixed_nll_c=capture('candidate',[[1e30,1],[0,0]],[0,1]);
my $mixed_nll_r=capture('reference',[[1e30,0],[0,0]],[0,1]);
($rc,$j)=compare($mixed_nll_c,$mixed_nll_r,'--diagnostic');
is($rc,0,'large NLL difference regression exits successfully');
is($j->{metrics}{mean_nll_increase},-1,'difference form preserves small NLL change');
ok($j->{consumed_artifact_hashes_verified},'hashes bind bytes consumed by the comparator');
ok($j->{input_hashes_match_before_after_comparison},'frozen files remain unchanged through comparison');
my $rawinfo={schema=>'flash-sass.raw-model-capture.v1',teacher_forced=>JSON::PP::true,
    positions=>2,vocab_size=>2,byte_order=>'little'};
sub freeze_info {
    my ($info)=@_;
    my $d="$tmp/info".++$seq;
    put("$tmp/raw-info.json",encode_json($info));
    return run('freeze','--kind','reference','--ids',"$fr/ids.u32",'--logits',"$fr/logits.f32",
        '--vocab-size',2,'--raw-info',"$tmp/raw-info.json",'--out',$d);
}
is(freeze_info($rawinfo),0,'legacy GPU metadata without dtype/completion remains replayable');
for my $bad ({%$rawinfo,dtype=>'>f4'},{%$rawinfo,dtype=>'<f2'},
             {%$rawinfo,byte_order=>'big'},{%$rawinfo,complete=>JSON::PP::false},
             {%$rawinfo,complete=>'true'}) {
    is(freeze_info($bad),2,'contradictory raw interpretation/completion is rejected');
}
my $reference_info={%$rawinfo,schema=>'flash-sass.independent-gguf-capture.v1'};
is(freeze_info($reference_info),2,'reference adapter schema requires dtype and completion');
$reference_info->{dtype}='<f4';$reference_info->{complete}=JSON::PP::true;
is(freeze_info($reference_info),0,'complete little-endian FP32 adapter capture accepted');
my $candidate_v2={%$rawinfo,schema=>'flash-sass.raw-model-capture.v2'};
is(freeze_info($candidate_v2),2,'current GPU capture requires its final completion marker');
$candidate_v2->{dtype}='<f4';$candidate_v2->{complete}=JSON::PP::true;
is(freeze_info($candidate_v2),0,'complete current GPU capture is accepted');
done_testing();
