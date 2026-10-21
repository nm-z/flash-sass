#!/usr/bin/env perl
# Validate and bind frozen attention inputs, then run the independent C control.
# Matching endpoint hashes do not attest the history or origin of a capture.
use v5.16;
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA ();
use Errno qw(ENOENT);
use Fcntl qw(O_RDONLY O_WRONLY O_CREAT O_EXCL O_NOFOLLOW S_ISREG S_ISDIR);
use File::Basename qw(dirname basename);
use File::Spec ();
use File::Temp qw(tempdir);
use FindBin qw($RealBin);
use Getopt::Long qw(GetOptionsFromArray);
use JSON::PP ();
use POSIX qw(isfinite strftime);
use Time::HiRes qw(time);

my $json=JSON::PP->new->utf8->canonical;
my $scalar=JSON::PP->new->utf8->canonical->allow_nonref;
my @limit_fields=qw(arithmetic_max_abs arithmetic_rmse arithmetic_relative_l2 query_max_abs);
sub jb { $_[0] ? JSON::PP::true : JSON::PP::false }
sub field {
    my ($v,$key,$where)=@_;
    ref($v) eq 'HASH' && exists($v->{$key}) or die "$where.$key is missing\n";
    return $v->{$key};
}
sub string {
    my ($v,$where)=@_;
    defined($v) && !ref($v) && $scalar->encode($v)=~/\A"/ && length($v)
        or die "$where must be a nonempty JSON string\n";
    return $v;
}
sub number {
    my ($v,$where)=@_;
    defined($v) && !ref($v) or die "$where must be a JSON number\n";
    $scalar->encode($v)=~/\A-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\z/
        or die "$where must be a JSON number\n";
    isfinite($v) or die "$where must be finite\n";
    return 0+$v;
}
sub nonnegative {
    my ($v,$where)=@_;
    $v=number($v,$where); $v>=0 or die "$where must be nonnegative\n"; return $v;
}
sub uint {
    my ($v,$lo,$hi,$where)=@_;
    $v=number($v,$where);
    $v==int($v) && $v>=$lo && $v<=$hi or die "$where must be an integer in [$lo,$hi]\n";
    return int($v);
}
sub boolean {
    my ($v,$where)=@_;
    JSON::PP::is_bool($v) or die "$where must be a JSON boolean\n";
    return $v ? 1 : 0;
}
sub expect_string {
    my ($v,$key,$wanted,$where)=@_;
    string(field($v,$key,$where),"$where.$key") eq $wanted
        or die "$where.$key must equal $wanted\n";
}
sub expect_bool {
    my ($v,$key,$wanted,$where)=@_;
    boolean(field($v,$key,$where),"$where.$key")==$wanted
        or die "$where.$key has the wrong boolean value\n";
}
sub expect_uint {
    my ($v,$key,$wanted,$where)=@_;
    uint(field($v,$key,$where),0,4294967295,"$where.$key")==$wanted
        or die "$where.$key must equal $wanted\n";
}
# Reject duplicate object keys, including equivalent escaped spellings.
# JSON::PP validates the grammar first; this pass tracks keys at each depth.
sub decode_object {
    my ($bytes,$where)=@_;
    my $v=eval { $json->decode($bytes) };
    !$@ && ref($v) eq 'HASH' or die "$where is not a valid JSON object: $@\n";
    my @stack; pos($bytes)=0;
    while ($bytes=~/\G\s*("(?:[^"\\]|\\.)*"|[{}\[\],:]|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null)/gc) {
        my $t=$1;
        if ($t eq '{') { push @stack,{kind=>'object',key=>1,seen=>{}}; }
        elsif ($t eq '[') { push @stack,{kind=>'array'}; }
        elsif ($t eq '}' || $t eq ']') { pop @stack; }
        elsif ($t eq ',') { $stack[-1]{key}=1 if @stack && $stack[-1]{kind} eq 'object'; }
        elsif ($t=~/\A"/ && @stack && $stack[-1]{kind} eq 'object' && $stack[-1]{key}) {
            my $key=$scalar->decode($t);
            !$stack[-1]{seen}{$key}++ or die "$where contains a duplicate object key\n";
            $stack[-1]{key}=0;
        }
    }
    substr($bytes,pos($bytes)//0)=~/\A\s*\z/ or die "$where could not be scanned consistently\n";
    return $v;
}
sub exists_path {
    my ($p)=@_; my @st=lstat($p);
    return 1 if @st;
    return 0 if $! == ENOENT;
    die "cannot inspect $p: $!\n";
}
sub fingerprint {
    my ($p,$expected,$retain,$limit)=@_;
    sysopen(my $f,$p,O_RDONLY|O_NOFOLLOW) or die "cannot open regular input $p: $!\n";
    binmode($f) or die "$p: $!\n";
    my @before=stat($f);
    @before && S_ISREG($before[2]) or die "$p is not a regular file\n";
    $expected=$before[7] unless defined($expected);
    $before[7]==$expected or die "$p has $before[7] bytes; expected $expected\n";
    !$retain || $expected<=$limit or die "$p exceeds retained-byte limit\n";
    my $sha=Digest::SHA->new(256); my $count=0; my $bytes='';
    while (1) {
        my $n=sysread($f,my $chunk,1024*1024);
        defined($n) or die "cannot read $p: $!\n";
        last unless $n;
        $count+=$n; $count<=$expected or die "$p grew during read\n";
        $sha->add($chunk); $bytes.=$chunk if $retain;
    }
    my @after=stat($f);
    @after && $count==$expected &&
        join(':',@before[0,1,7,9,10]) eq join(':',@after[0,1,7,9,10])
        or die "$p changed during read\n";
    close($f) or die "$p: $!\n";
    return ({bytes=>$count,sha256=>$sha->hexdigest},$retain ? $bytes : undef);
}
sub validate_metrics {
    my ($m,$where)=@_;
    for my $key (qw(max_abs mean_abs rmse error_l2 reference_l2 observed_l2)) {
        nonnegative(field($m,$key,$where),"$where.$key");
    }
    my $finite=boolean(field($m,'relative_l2_finite',$where),"$where.relative_l2_finite");
    my $relative=field($m,'relative_l2',$where);
    if ($finite) { nonnegative($relative,"$where.relative_l2"); }
    else { !defined($relative) or die "$where nonfinite relative L2 must be null\n"; }
    for my $key (qw(reference observed)) {
        boolean(field($m,"$key\_zero",$where),"$where.$key\_zero")==($m->{"$key\_l2"}==0 ? 1 : 0)
            or die "$where zero flag contradicts norm\n";
    }
    my $defined=boolean(field($m,'cosine_defined',$where),"$where.cosine_defined");
    my $cosine=field($m,'cosine',$where);
    if ($defined) {
        $cosine=number($cosine,"$where.cosine");
        $cosine>=-1 && $cosine<=1 or die "$where cosine outside [-1,1]\n";
    } else { !defined($cosine) or die "$where undefined cosine must be null\n"; }
}
sub main {
    my @args=@ARGV; my $limits_arg;
    Getopt::Long::Configure(qw(no_auto_abbrev no_ignore_case));
    GetOptionsFromArray(\@args,'limits=s'=>\$limits_arg) && @args==3
        or die "usage: $0 [--limits LIMITS_JSON] CAPTURE_DIR N NEW_OUTPUT_JSON\n";
    my ($capture_arg,$n,$output_arg)=@args;
    $n=~/\A[1-9][0-9]{0,5}\z/ && $n<=262144 or die "N must be in [1,262144]\n";
    $n=0+$n;
    my $capture=abs_path($capture_arg);
    defined($capture) && -d $capture or die "capture directory unavailable\n";
    $output_arg!~m{/\z} or die "output must name a new file\n";
    my $parent=abs_path(dirname(File::Spec->rel2abs($output_arg)));
    defined($parent) && -d $parent or die "output parent unavailable\n";
    my $leaf=basename($output_arg);
    length($leaf) && $leaf ne '.' && $leaf ne '..' or die "invalid output file\n";
    my $output=File::Spec->catfile($parent,$leaf);
    !exists_path($output) or die "refusing to replace existing output $output\n";
    my $snap="$capture/n$n"; my @st=lstat($snap);
    @st && S_ISDIR($st[2]) or die "$snap must be an ordinary directory\n";
    my $binary=abs_path($ENV{QSA_ATTENTION_QUALITY_BIN}//"$RealBin/qsa_attention_quality");
    defined($binary) && -f $binary && -x $binary or die "build qsa_attention_quality first\n";
    my (@bindings,@absent);
    my $bind=sub {
        my ($role,$label,$path,$expected,$retain,$limit)=@_;
        my ($identity,$bytes)=fingerprint($path,$expected,$retain,$limit//4*1024*1024);
        push @bindings,{role=>$role,file=>$label,path=>$path,
            bytes_before=>$identity->{bytes},sha256_before=>$identity->{sha256}};
        return $bytes;
    };
    my $complete=decode_object($bind->('metadata','complete.json',"$capture/complete.json",undef,1),'complete.json');
    my $snapshot=decode_object($bind->('metadata',"n$n/snapshot.json","$snap/snapshot.json",undef,1),'snapshot');
    my $schema=string(field($complete,'schema','complete'),'complete.schema');
    my $synthetic;
    if ($schema eq 'flash-sass.qsa-layer-capture.v1') {
        $synthetic=0; expect_string($snapshot,'schema','flash-sass.qsa-layer-snapshot.v1','snapshot');
    } elsif ($schema eq 'flash-sass.qsa-attention-capture.v1') {
        $synthetic=1; expect_string($snapshot,'schema','flash-sass.qsa-attention-snapshot.v1','snapshot');
    } else { die "unsupported capture schema\n"; }
    my $total=uint(field($complete,'positions','complete'),$n,262144,'complete.positions');
    my $budget=uint(field($complete,'budget','complete'),1,4096,'complete.budget');
    my $epsilon=nonnegative(field($complete,'epsilon','complete'),'complete.epsilon');
    my %contract=(byte_order=>'little',
        input_boundary=>'post_projection_before_norm_rope_and_kv_quantization',
        gamma=>'supplied_multiplicative_f32',rope_tables=>'supplied_f32');
    for my $key (sort keys %contract) {
        expect_string($complete,$key,$contract{$key},'complete');
        expect_string($snapshot,$key,$contract{$key},'snapshot') if exists $snapshot->{$key};
    }
    my $build=string(field($complete,'build_revision','complete'),'complete.build_revision');
    expect_bool($complete,'teacher_forced',$synthetic ? 0 : 1,'complete');
    expect_bool($snapshot,'teacher_forced',$synthetic ? 0 : 1,'snapshot')
        if exists $snapshot->{teacher_forced};
    my $lengths=field($complete,'snapshot_lengths','complete');
    ref($lengths) eq 'ARRAY' && @$lengths or die "snapshot_lengths must be a nonempty array\n";
    my %lengths;
    for my $value (@$lengths) {
        my $length=uint($value,1,$total,'snapshot_lengths[]');
        !$lengths{$length}++ or die "duplicate snapshot length\n";
    }
    $lengths{$n} or die "N not declared complete\n";
    my $blocks=int($n/4); my $tail=$n%4;
    my $selected=$blocks<$budget ? $blocks : $budget;
    my $attended=4*$selected+$tail;
    my $sparse=$blocks>$selected ? 1 : 0;
    my %expected=(length=>$n,position=>$n-1,budget=>$budget,complete_blocks=>$blocks,
        tail=>$tail,attended_keys=>$attended);
    for my $key (sort keys %expected) { expect_uint($snapshot,$key,$expected{$key},'snapshot'); }
    expect_bool($snapshot,'sparse_exclusion',$sparse,'snapshot');
    my %origin=(capture_schema=>$schema,
        evidence_kind=>$synthetic ? 'synthetic_gpu_fixture' : 'model_layer_capture',
        build_revision_declared=>$build,teacher_forced=>jb(!$synthetic),
        historical_producer_identity_attested=>jb(0),%contract);
    if ($synthetic) {
        for my $item ([$complete,'complete'],[$snapshot,'snapshot']) {
            my ($m,$where)=@$item;
            expect_string($m,'evidence_kind','synthetic_gpu_fixture',$where);
            expect_string($m,'selection_source','supplied_fixture_ids',$where);
            expect_bool($m,'production_sparse_selector_executed',0,$where);
            expect_bool($m,'full_model',0,$where);
        }
        $origin{seed_declared}=uint(field($complete,'seed','complete'),0,4294967295,'complete.seed');
        expect_string($complete,'rope_table_generator','fp64_libm_cast_f32','complete');
        $origin{selection_source}='supplied_fixture_ids';
        $origin{production_sparse_selector_executed}=jb(0);
        $origin{full_model}=jb(0);
        $origin{sparse_exclusion_scope}='Supplied block IDs; production model/indexer dispatch is not claimed.';
    } else {
        for my $key (qw(layer die)) {
            $origin{$key}=uint(field($complete,$key,'complete'),0,2147483647,"complete.$key");
            expect_uint($snapshot,$key,$origin{$key},'snapshot');
        }
        expect_string($complete,'pool_dtype','fp16','complete');
        expect_uint($complete,'rope_base',10000000,'complete');
        $origin{FLASH_KV_HOST_declared}=uint(field($complete,'FLASH_KV_HOST','complete'),0,1,'complete.FLASH_KV_HOST');
        expect_string($snapshot,'scores_source',$sparse?'gpu':'not_dispatched','snapshot');
        expect_string($snapshot,'selected_ids_source',$sparse?'gpu':'implicit_all_blocks','snapshot');
        $origin{selected_ids_source_declared}=$snapshot->{selected_ids_source};
        $origin{sparse_exclusion_scope}='Captured IDs and producer declarations; historical execution is not independently attested.';
    }
    if (exists_path("$capture/ids.u32")) {
        $bind->('token_ids','ids.u32',"$capture/ids.u32",$total*4,0);
    } elsif ($synthetic) { push @absent,"$capture/ids.u32"; }
    else { die "model-layer capture requires ids.u32\n"; }
    for my $spec (
        ['raw_k.f32',$total*512*4],['raw_v.f32',$total*512*4],
        ['kn.f32',256*4],['qn.f32',256*4],
        ['rope_cs.f32',$total*32*4],['rope_sn.f32',$total*32*4]) {
        $bind->('replay_input',$spec->[0],"$capture/$spec->[0]",$spec->[1],0);
    }
    for my $spec (
        ['qg.f32',12288*4],['qo.f32',6144*4],['aout.f32',6144*4],
        ['kq.i8',$n*512],['vq.i8',$n*512],['ks.f16',$n*32],['vs.f16',$n*32]) {
        $bind->('replay_input',"n$n/$spec->[0]","$snap/$spec->[0]",$spec->[1],0);
    }
    my $id_bytes=$bind->('replay_input',"n$n/selected_ids.u32","$snap/selected_ids.u32",$selected*4,1);
    my @ids=unpack('V*',$id_bytes); my %seen;
    for my $i (0..$#ids) {
        my $id=$ids[$i];
        $id<$blocks && !$seen{$id}++ or die "selected ID out of range or duplicate\n";
        $sparse || $id==$i or die "implicit dense IDs must be ascending complete blocks\n";
    }
    my $normalized=exists_path("$capture/normalized_k.f32");
    my $declared=exists($complete->{normalized_k_available})
        ? boolean($complete->{normalized_k_available},'normalized_k_available') : 0;
    $declared==$normalized or die "normalized K presence contradicts metadata\n";
    if ($normalized) {
        expect_string($complete,'prequant_k_source','separate_gpu_fp32_append','complete');
        $bind->('replay_input','normalized_k.f32',"$capture/normalized_k.f32",$total*512*4,0);
    } else {
        expect_string($complete,'prequant_k_source','unavailable','complete')
            if exists $complete->{prequant_k_source};
        push @absent,"$capture/normalized_k.f32";
    }
    $origin{normalized_k_available}=jb($normalized);
    my ($limits,$limits_identity); my %limit_values;
    if (defined $limits_arg) {
        my $p=abs_path($limits_arg); defined($p) or die "limits profile unavailable\n";
        $limits=decode_object($bind->('limits','limits_profile',$p,undef,1),'limits profile');
        $limits_identity={path=>$p,sha256=>$bindings[-1]{sha256_before},bytes=>$bindings[-1]{bytes_before}};
        expect_string($limits,'schema','flash-sass.qsa-attention-limits.v1','limits');
        my %allowed=map { $_=>1 } ('schema',@limit_fields);
        for my $key (keys %$limits) { $allowed{$key} or die "unknown limit: $key\n"; }
        for my $key (@limit_fields) { $limit_values{$key}=nonnegative(field($limits,$key,'limits'),"limits.$key"); }
    }
    $bind->('reference_binary','qsa_attention_quality',$binary,undef,0);
    my $wrapper=abs_path($0); defined($wrapper) or die "cannot resolve wrapper source\n";
    $bind->('wrapper_source','qsa_attention_replay.pl',$wrapper,undef,0);
    my $temp=tempdir('.qsa-attention-replay-XXXXXXXX',DIR=>$parent,CLEANUP=>1);
    my $raw="$temp/reference.json"; my $epsilon_arg=sprintf('%.17g',$epsilon);
    my @command=($binary,'--replay',$capture,"$n","$total","$budget",$epsilon_arg,$raw);
    my $started_utc=strftime('%Y-%m-%dT%H:%M:%SZ',gmtime());
    my $start=time(); my $wait=system {$binary} @command; my $elapsed=time()-$start;
    $wait!=-1 or die "could not execute C replay: $!\n";
    !($wait&127) && ($wait>>8)==0 or die "C replay failed with wait status $wait\n";
    my ($raw_identity,$raw_bytes)=fingerprint($raw,undef,1,32*1024*1024);
    my $r=decode_object($raw_bytes,'reference result');
    expect_string($r,'schema','flash-sass.qsa-attention-reference.v1','reference');
    for my $key (qw(complete all_modes_same_selected_ids all_modes_same_attended_token_order)) {
        expect_bool($r,$key,1,'reference');
    }
    for my $key (qw(quality gpu framework_dtype_equivalence pure_k_storage_isolated)) {
        expect_bool($r,$key,0,'reference');
    }
    expect_bool($r,'normalized_k_available',$normalized,'reference');
    my %r_expected=(length=>$n,total_positions=>$total,budget=>$budget,attended_keys=>$attended,
        complete_blocks=>$blocks,selected_blocks=>$selected,excluded_complete_blocks=>$blocks-$selected,tail=>$tail);
    for my $key (keys %r_expected) { expect_uint($r,$key,$r_expected{$key},'reference'); }
    number(field($r,'epsilon','reference'),'reference.epsilon')==$epsilon or die "C epsilon changed\n";
    my $reported=field($r,'selected_ids','reference');
    ref($reported) eq 'ARRAY' && @$reported==@ids or die "C selection shape changed\n";
    for my $i (0..$#ids) { uint($reported->[$i],0,4294967295,'selected_ids[]')==$ids[$i] or die "C selection changed\n"; }
    my @tokens;
    for my $id (@ids) { push @tokens,map { $id*4+$_ } 0..3; }
    push @tokens,4*$blocks..$n-1 if $tail;
    my $rt=field($r,'attended_token_ids','reference');
    ref($rt) eq 'ARRAY' && @$rt==@tokens or die "C attended-token shape changed\n";
    for my $i (0..$#tokens) { uint($rt->[$i],0,$n-1,'attended_token_ids[]')==$tokens[$i] or die "C attended-token order changed\n"; }
    validate_metrics(field($r,'arithmetic_vs_gpu','reference'),'arithmetic_vs_gpu');
    validate_metrics(field($r,'query_vs_gpu','reference'),'query_vs_gpu');
    my @names=qw(exported_q_dequantized_kv exported_q_raw_k_normalized_rope_dequantized_v
        exported_q_dequantized_k_raw_v exported_q_raw_k_normalized_rope_raw_v reconstructed_q_dequantized_kv);
    push @names,qw(exported_q_gpu_f32_k_dequantized_v exported_q_gpu_f32_k_raw_v) if $normalized;
    my $modes=field($r,'modes','reference');
    ref($modes) eq 'ARRAY' && @$modes==@names or die "wrong C mode count\n";
    for my $i (0..$#names) {
        my $m=$modes->[$i]; my $where="mode[$i]";
        expect_string($m,'name',$names[$i],$where);
        string(field($m,'interpretation',$where),"$where.interpretation");
        expect_string($m,'reference',$i?$names[0]:'supplied_aout',$where);
        expect_bool($m,'same_selected_ids',1,$where);
        validate_metrics(field($m,'metrics',$where),"$where.metrics");
        my $shape=field($m,'output_shape',$where);
        ref($shape) eq 'ARRAY' && @$shape==2
            && uint($shape->[0],24,24,'output_shape[0]')==24
            && uint($shape->[1],256,256,'output_shape[1]')==256 or die "wrong C output shape\n";
        my $v=field($m,'output_f64_row_major',$where);
        ref($v) eq 'ARRAY' && @$v==6144 or die "wrong C output array length\n";
        number($_,"$where.output[]") for @$v;
    }
    for my $b (@bindings) {
        my ($after)=fingerprint($b->{path},$b->{bytes_before},0,0);
        $b->{bytes_after}=$after->{bytes}; $b->{sha256_after}=$after->{sha256};
        $b->{sha256_before} eq $b->{sha256_after} or die "$b->{path} changed across replay\n";
    }
    for my $p (@absent) { !exists_path($p) or die "previously absent input appeared: $p\n"; }
    my %observed=(arithmetic_max_abs=>$r->{arithmetic_vs_gpu}{max_abs},
        arithmetic_rmse=>$r->{arithmetic_vs_gpu}{rmse},
        arithmetic_relative_l2=>$r->{arithmetic_vs_gpu}{relative_l2},query_max_abs=>$r->{query_vs_gpu}{max_abs});
    my @failed;
    if (defined $limits) {
        for my $key (@limit_fields) {
            push @failed,{field=>$key,observed=>$observed{$key},limit=>$limit_values{$key},
                reason=>defined($observed{$key})?'exceeds limit':'undefined or nonfinite metric'}
                if !defined($observed{$key}) || $observed{$key}>$limit_values{$key};
        }
    }
    my $status=!defined($limits)?'diagnostic_only':@failed?'arithmetic_limits_exceeded':'arithmetic_limits_met';
    my $report={
        schema=>'flash-sass.qsa-attention-replay.v1',complete=>jb(1),status=>$status,exit_status=>@failed?3:0,
        quality=>jb(0),qsa_quality_validated=>jb(0),full_model_quality_validated=>jb(0),
        independent_checkpoint_reference=>jb(0),framework_dtype_equivalence=>jb(0),
        producer_origin=>\%origin,
        replay=>{backend=>'CPU FP64 reconstruction from supplied capture inputs',gpu=>jb(0),
            argv=>\@command,started_utc=>$started_utc,elapsed_seconds=>$elapsed,process_exit_status=>0,
            epsilon=>$epsilon,epsilon_argv=>$epsilon_arg,raw_output_bytes=>$raw_identity->{bytes},
            raw_output_sha256=>$raw_identity->{sha256}},
        input_binding=>{before_after_hashes_match=>jb(1),immutable_during_execution_attested=>jb(0),
            files=>\@bindings,absent_before_and_after=>\@absent},
        limits=>defined($limits)?{schema=>'flash-sass.qsa-attention-limits.v1',profile=>$limits_identity,
            values=>\%limit_values,enforced_fields=>\@limit_fields,observed=>\%observed,failed=>\@failed,
            file_observed_before_replay=>jb(1),historical_predeclaration_attested=>jb(0),
            quantization_effects_gated=>jb(0)}:undef,
        quantization_effects=>{gated=>jb(0),k_storage_only_measured=>jb(0),
            gpu_fp32_k_append_comparison_measured=>jb($normalized),
            k_append_basis=>$normalized
                ? 'Separate GPU FP32 append; differences may include K normalization/RoPE between append kernels.'
                : 'Unavailable: no separately supplied normalized/RoPE FP32 K control.',
            interpretation=>'All arms hold selected IDs and order fixed; combined effects are not additive.'},
        reference_result=>$r};
    my $pending="$temp/report.pending.json";
    sysopen(my $out,$pending,O_WRONLY|O_CREAT|O_EXCL,0600) or die "cannot create pending report: $!\n";
    binmode($out) or die "$pending: $!\n";
    print {$out} $json->encode($report),"\n" or die "cannot write report: $!\n";
    close($out) or die "cannot close report: $!\n";
    link($pending,$output) or die "cannot publish new report without replacement: $!\n";
    print "$output\n";
    return @failed ? 3 : 0;
}
my $rc=eval { main() };
if ($@) { print STDERR "qsa_attention_replay: $@"; exit 2; }
exit $rc;
