package BenchAcceptance;
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;
use File::Basename qw(basename);
use File::Copy qw(copy);
use File::Path qw(make_path);
use POSIX qw(isfinite);
our @DEPTHS = (2048,2051,2052,3100,8000,65000,262000);
my $json = JSON::PP->new->canonical;
sub uint { defined($_[0]) && !ref($_[0]) && $_[0] =~ /\A[0-9]+\z/ }
sub canonical_hash { sha256_hex($json->encode($_[0])) }

sub policy {
    my ($a,$depths) = @_;
    die "outputs must be 2..256; pairs >=5; warmup outputs >=2\n"
        unless uint($a->{outputs}) && $a->{outputs} >= 2 && $a->{outputs} <= 256
        && uint($a->{pairs}) && $a->{pairs} >= 5
        && uint($a->{warmup_outputs}) && $a->{warmup_outputs} >= 2
        && $a->{warmup_outputs} <= 256;
    my $standard = $a->{outputs} == 129 && $json->encode($depths) eq $json->encode(\@DEPTHS);
    return {standard_matrix => $standard ? JSON::PP::true : JSON::PP::false,
            required_depths => [@DEPTHS], requested_depths => $depths,
            outputs => $a->{outputs}, intervals => $a->{outputs}-1,
            pairs => $a->{pairs}, warmup_outputs => $a->{warmup_outputs},
            order => 'alternating host/vram and vram/host per repetition'};
}

sub capabilities {
    my ($configs,$trace,$outputs) = @_;
    for my $name (qw(host vram)) {
        my $c = $configs->{$name};
        die "$name lacks requestmeta_schema=1; update both engines before measuring\n"
            unless ($c->{requestmeta_schema}//'') eq '1';
        die "$name lacks tensor_schema=1; actual loaded tensor inventory required\n"
            unless ($c->{tensor_schema}//'') eq '1';
        for my $key (qw(rev maxt layers nblk dies kv_host ple_disk attn_dies split attn_map cubin_fnv)) {
            die "$name missing runtime config $key\n" unless defined($c->{$key}) && length($c->{$key});
        }
        die "$name requires full 48-layer, default 512-block QSA configuration\n"
            unless $c->{layers} eq '48/48' && $c->{nblk} eq '512';
        die "$name invalid capacity or die count\n"
            unless uint($c->{maxt}) && $c->{maxt}>0 && uint($c->{dies}) && $c->{dies}>0;
        if ($trace) {
            die "$name lacks the integrated schema-2 phase tracing runtime\n"
                unless ($c->{trace_schema}//'') eq '2';
            die "$name insufficient trace capacity\n"
                unless uint($c->{trace_tokens}) && $c->{trace_tokens} >= $outputs;
            die "$name trace_ctas must be 1 or 16\n"
                unless ($c->{trace_ctas}//'') =~ /\A(?:1|16)\z/;
            die "$name exceeds schema-2 trace allocation capacity\n"
                unless $c->{trace_tokens}*$c->{trace_ctas}<=256;
        } else {
            die "$name authoritative throughput requires disabled phase instrumentation\n"
                if ($c->{trace_tokens}//0) != 0;
        }
    }
}

sub requestmeta {
    my ($m,$prompt,$outputs,$previous) = @_;
    die "request metadata schema/clock missing\n"
        unless ($m->{schema}//'') eq '1' && ($m->{clock}//'') eq 'CLOCK_MONOTONIC';
    for my $k (qw(request aborted ended completed requested prompt resumed snapshot ple_short_reads
                  request_start_ns request_end_ns build_start_ns build_end_ns run_start_ns
                  engine_first_observed_ns engine_last_observed_ns engine_prefix_observed_ns)) {
        die "invalid request metadata $k\n" unless uint($m->{$k});
    }
    die "stale request identity\n" unless $m->{request}>0
        && (!defined($previous) || $m->{request}>$previous);
    die "unsuccessful request lifecycle\n"
        unless ($m->{status}//'') eq 'completed' && $m->{aborted}==0 && $m->{ended}==0
        && $m->{requested}==$prompt+$outputs-1 && $m->{completed}==$m->{requested}
        && $m->{prompt}==$prompt && $m->{ple_short_reads}==0
        && $m->{resumed}<$prompt && $m->{snapshot}<$prompt;
    my @order = map {0+$m->{$_}} qw(request_start_ns build_start_ns build_end_ns run_start_ns
                                    engine_first_observed_ns engine_last_observed_ns request_end_ns);
    die "invalid engine timing boundaries\n" unless $order[0]>0;
    for my $i (1..$#order) { die "invalid engine timing order\n" if $order[$i]<$order[$i-1]; }
    my $prefix=0+$m->{engine_prefix_observed_ns};
    die "invalid host-observed prefix boundary\n" if $prefix
        && ($prefix<$m->{run_start_ns} || $prefix>$m->{engine_first_observed_ns});
    return {
        build_ms => ($m->{build_end_ns}-$m->{build_start_ns})/1e6,
        engine_first_observed_ms => ($m->{engine_first_observed_ns}-$m->{run_start_ns})/1e6,
        engine_observed_decode_ms => ($m->{engine_last_observed_ns}-$m->{engine_first_observed_ns})/1e6,
        engine_request_ms => ($m->{request_end_ns}-$m->{request_start_ns})/1e6,
        engine_prefix_observed_ms => $prefix ? ($prefix-$m->{run_start_ns})/1e6
            : $m->{resumed}==$prompt-1 ? 0 : undef,
        prefill_device_ms => undef,
        note => 'Prefix completion is a host observation, not a device prefill duration; zero/absent observation stays null. Build measures request program construction/setup, not the offline compiler.',
    };
}

sub answer_reference {
    my ($r,$ids,$outputs) = @_;
    die "answer reference must be schema 1\n" unless ref($r) eq 'HASH' && ($r->{schema}//0)==1;
    die "answer reference does not identify these post-template IDs\n"
        unless ($r->{prompt_ids_sha256}//'') eq canonical_hash($ids);
    my $p=$r->{provenance};
    die "answer reference provenance required\n"
        unless ref($p) eq 'HASH' && ($p->{method}//'') eq 'independent_reference_tokens'
        && ($p->{source_sha}//'') =~ /\A[0-9a-f]{40}\z/
        && ($p->{artifact_sha256}//'') =~ /\A[0-9a-f]{64}\z/
        && defined($p->{description}) && length($p->{description});
    die "answer reference depths required\n" unless ref($r->{depths}) eq 'HASH' && keys %{$r->{depths}};
    for my $depth (keys %{$r->{depths}}) {
        my $v=$r->{depths}{$depth};
        die "invalid answer reference depth\n" unless uint($depth) && $depth>0 && $depth<=@$ids;
        die "answer reference must explicitly validate each answer\n"
            unless ref($v) eq 'HASH' && JSON::PP::is_bool($v->{valid}) && $v->{valid};
        die "invalid reference outputs\n" unless ref($v->{generated_ids}) eq 'ARRAY'
            && @{$v->{generated_ids}}==$outputs
            && !(grep {!uint($_) || $_>=248320} @{$v->{generated_ids}});
        my @prompt=@$ids[0..$depth-1];
        die "answer reference prefix identity mismatch\n"
            unless ($v->{prompt_sha256}//'') eq canonical_hash(\@prompt);
    }
    return $r;
}

sub check_answer {
    my ($reference,$depth,$row) = @_;
    return undef unless $reference;
    my $r=$reference->{depths}{$depth};
    die "no validated answer reference for supported depth $depth\n" unless $r;
    die "generated answer differs from validated independent reference at depth $depth\n"
        unless $json->encode($row->{generated_ids}) eq $json->encode($r->{generated_ids});
    return JSON::PP::true;
}

sub summary {
    my ($rows,$depths,$pairs,$require_valid) = @_;
    my @out;
    for my $depth (@$depths) {
        for my $placement (qw(host vram)) {
            my @r=grep {($_->{depth}//-1)==$depth && ($_->{placement}//'') eq $placement
                && ($_->{status}//'') eq 'complete'} @$rows;
            next unless @r;
            die "incomplete paired matrix at $depth/$placement\n" unless @r==$pairs;
            die "unvalidated answer cannot enter successful aggregate\n"
                if $require_valid && grep {!$_->{answer_valid}} @r;
            my @raw=map {map {$_/1e6} @{$_->{intervals_ns}}} @r;
            my @means=map {$_->{client_decode_ms}/$_->{decode_intervals}} @r;
            my $median=PairedDepthBench::median(\@means);
            push @out,{depth=>$depth,placement=>$placement,runs=>scalar(@r),intervals=>scalar(@raw),
                median_run_ms_per_token=>$median,p95_interval_ms=>PairedDepthBench::p95(\@raw),
                median_tok_s=>1000/$median};
        }
    }
    return \@out;
}

sub attention_work {
    my($traces,$depth,$outputs)=@_;my %seen;
    for my $s(@$traces) {
        die "attribution block budget differs from default QSA\n" unless ($s->{metadata}{nblk}//0)==512;
        for my $a(@{$s->{selected_keys}}) {
            die "dense attention cannot be labeled default capped QSA\n" unless $a->{mode} eq 'qsa';
            my($position,$layer)=@$a{qw(position layer)};
            die "attention execution outside the requested layer/position matrix\n"
                unless $position>=$depth-1 && $position<$depth+$outputs-1
                && $layer>=3 && $layer<48 && $layer%4==3;
            die "duplicate attention execution across dies\n" if $seen{"$position:$layer"}++;
        }
    }
    for my $position($depth-1..$depth+$outputs-2) {
        for my $layer(map {4*$_+3}0..11) {
            die "missing executed attention layer at $position:$layer\n" unless $seen{"$position:$layer"};
        }
    }
    return {mode=>'default_512_block_qsa',layer_positions=>scalar(keys %seen),
        first_visible=>$depth,last_visible=>$depth+$outputs-1,
        note=>'Before the cap, default QSA may attend every visible key; capped is recorded separately for each execution.'};
}

sub validate_collected_row {
    my($r,$outputs)=@_;
    die "invalid collected request identity\n" unless uint($r->{depth}) && uint($r->{pair})
        && ($r->{placement}//'')=~/\A(?:host|vram)\z/ && ($r->{status}//'') eq 'complete';
    my $timing=requestmeta($r->{requestmeta},$r->{depth},$outputs,undef);
    die "collected engine interval disagrees with request metadata\n"
        unless $r->{engine_observed_decode_ms}==$timing->{engine_observed_decode_ms};
    die "incomplete collected output arrays\n" unless ref($r->{generated_ids}) eq 'ARRAY'
        && ref($r->{client_observed_ns}) eq 'ARRAY' && @{$r->{generated_ids}}==$outputs
        && @{$r->{client_observed_ns}}==$outputs && !(grep {!uint($_)||$_>=248320} @{$r->{generated_ids}})
        && !(grep {!uint($_)} @{$r->{client_observed_ns}});
    my @intervals=map {$r->{client_observed_ns}[$_]-$r->{client_observed_ns}[$_-1]}1..$outputs-1;
    die "collected client interval differs from raw timestamps\n"
        if (grep {$_<=0} @intervals) || $json->encode(\@intervals) ne $json->encode($r->{intervals_ns})
        || ($r->{decode_intervals}//0)!=$outputs-1
        || $r->{client_decode_ms}!=PairedDepthBench::sum(\@intervals)/1e6;
    return 1;
}

sub matching_runtime {
    my ($manifest,$other,$context) = @_;
    for my $field (qw(source_sha source_dirty_sha256 weights sampling policy prompt_ids_sha256 tensors)) {
        die "unexplained $context difference: $field\n"
            unless exists($manifest->{$field}) && exists($other->{$field})
            && $json->encode($manifest->{$field}) eq $json->encode($other->{$field});
    }
    # Instrumentation changes the cubin, but must not change the host program,
    # converted weights, placement-specific launch settings or cache protocol.
    for my $role (qw(executable library)) {
        my $a=$manifest->{provenance}{artifacts}{$role}{sha256};
        my $b=$other->{provenance}{artifacts}{$role}{sha256};
        die "$context $role identity differs or is absent\n"
            unless defined($a) && defined($b) && $a eq $b;
    }
    for my $name(qw(host vram)) {
        my $a=$manifest->{provenance}{core_launch_records}{$name};
        my $b=$other->{provenance}{core_launch_records}{$name};
        die "$context launch records absent\n" unless ref($a) eq 'HASH' && ref($b) eq 'HASH';
        for my $field(qw(launch_environment engine_reload_commands ple_cache_conditions)) {
            my $av=$a->{$field};my $bv=$b->{$field};
            if($field eq 'launch_environment') {
                die "$context launch environment absent\n" unless ref($av) eq 'HASH' && ref($bv) eq 'HASH';
                $av={%$av};$bv={%$bv};
                delete @$av{qw(FLASH_TRACE FLASH_TRACE_CTAS FLASH_CUBIN FLASH_SOCK)};
                delete @$bv{qw(FLASH_TRACE FLASH_TRACE_CTAS FLASH_CUBIN FLASH_SOCK)};
            }
            die "unexplained $context launch difference: $name.$field\n"
                unless defined($av) && defined($bv) && $json->encode($av) eq $json->encode($bv);
        }
    }
}

sub compare_attribution {
    my ($out,$manifest,$rows,$directory) = @_;
    my $other=PairedDepthBench::load_json("$directory/manifest.json");
    my $state=PairedDepthBench::load_json("$directory/status.json");
    die "attribution record was not collected successfully\n"
        unless ($state->{collection_complete}//0) && ($other->{mode}//'') eq 'attribution';
    matching_runtime($manifest,$other,'throughput/attribution');
    my $tr=PairedDepthBench::load_json("$directory/results.json");
    my (%trace,%raw_files);
    for my $r (@$tr) {
        next if ($r->{status}//'') eq 'capacity_skip';
        validate_collected_row($r,$other->{policy}{outputs});
        die "invalid attribution request\n" unless ($r->{status}//'') eq 'complete'
            && ref($r->{device_publication_decode_ms_by_die}) eq 'ARRAY'
            && @{$r->{device_publication_decode_ms_by_die}};
        die "attribution raw exports absent\n" unless ref($r->{trace_files}) eq 'ARRAY'
            && @{$r->{trace_files}}==$other->{configs}{$r->{placement}}{dies}+1;
        my %dies;my $hosts=0;my @parsed_traces;
        for my $file (@{$r->{trace_files}}) {
            my $raw=$file->{raw};die "raw trace identity absent\n" unless ref($raw) eq 'HASH';
            my $path="$directory/".basename($raw->{path}//'');
            my $actual=PairedDepthBench::identity($path);
            die "attribution raw file changed\n"
                unless $actual->{sha256} eq ($raw->{sha256}//'') && $actual->{bytes}==$raw->{bytes};
            die "attribution raw basename refers to different files\n"
                if exists($raw_files{basename($path)}) && $raw_files{basename($path)} ne $path;
            $raw_files{basename($path)}=$path;
            my $m=$file->{metadata};
            die "attribution metadata does not cover this request\n"
                unless ref($m) eq 'HASH' && ($m->{schema}//0)==2 && ($m->{count}//0)==$other->{policy}{outputs}
                && ($m->{first}//-1)==$r->{depth}-1 && ($m->{request}//0)==($r->{requestmeta}{request}//-1);
            PairedDepthBench::trace_lifecycle($m,require_current=>1,
                expected_requested=>$r->{depth}+$other->{policy}{outputs}-1,allow_ended=>0);
            if(($file->{kind}//'') eq 'trace') {
                my $d=$file->{die};
                die "invalid attribution die identity\n" unless uint($d) && $d<$other->{configs}{$r->{placement}}{dies}
                    && ($m->{die}//-1)==$d;
                die "duplicate attribution die\n" if $dies{$d}++;
                my $s=PairedDepthBench::phase_summary(PairedDepthBench::read_csv_gzip($path),$file->{metadata});
                $parsed_traces[$d]=$s;
                die "attribution summary disagrees with raw intervals\n"
                    unless $s->{decode_ms}==$r->{device_publication_decode_ms_by_die}[$d];
                $r->{trace_op_mean_us_by_die}[$d]=$s->{op_mean_us};
            } elsif(($file->{kind}//'') eq 'hosttrace') {
                $hosts++;PairedDepthBench::host_summary(PairedDepthBench::read_csv_gzip($path),$file->{metadata});
            } else {die "unexpected attribution export\n";}
        }
        die "attribution requires each participating die and one host export\n"
            unless $hosts==1 && keys(%dies)==$other->{configs}{$r->{placement}}{dies};
        $r->{attention_execution}=attention_work(\@parsed_traces,$r->{depth},$other->{policy}{outputs});
        my $key=join(':',@$r{qw(depth pair placement)});
        die "duplicate attribution request $key\n" if exists($trace{$key});
        $trace{$key}=$r;
    }
    for my $name (qw(host vram)) {
        my %a=%{$manifest->{configs}{$name}};
        my %b=%{$other->{configs}{$name}};
        delete @a{qw(trace_tokens trace_ctas trace_schema cubin cubin_fnv)};
        delete @b{qw(trace_tokens trace_ctas trace_schema cubin cubin_fnv)};
        PairedDepthBench::matched(\%a,\%b,{});
    }
    my @comparison;
    for my $r (@$rows) {
        next if ($r->{status}//'') eq 'capacity_skip';
        my $key=join(':',@$r{qw(depth pair placement)});
        my $t=delete($trace{$key}) or die "missing matched attribution request $key\n";
        die "attribution used different outputs at $key\n"
            unless $json->encode($r->{generated_ids}) eq $json->encode($t->{generated_ids});
        die "attribution used different resume state at $key\n"
            unless $r->{done}{resumed}==$t->{done}{resumed};
        my $delta=$t->{client_decode_ms}-$r->{client_decode_ms};
        push @comparison,{depth=>$r->{depth},pair=>$r->{pair},placement=>$r->{placement},
            throughput_client_decode_ms=>$r->{client_decode_ms},
            instrumented_client_decode_ms=>$t->{client_decode_ms},
            trace_overhead_ms=>$delta,trace_overhead_fraction=>$delta/$r->{client_decode_ms},
            instrumented_device_publication_decode_ms_by_die=>$t->{device_publication_decode_ms_by_die},
            instrumented_engine_observed_decode_ms=>$t->{engine_observed_decode_ms},
            note=>'Separately collected matched request durations; a negative difference is retained, not clamped.'};
    }
    die "unmatched extra attribution requests\n" if keys %trace;
    make_path("$out/attribution");
    for my $name(sort keys %raw_files) {
        copy($raw_files{$name},"$out/attribution/$name") or die "copy attribution raw file: $!\n";
        die "attribution file changed while copying\n"
            unless PairedDepthBench::sha($raw_files{$name}) eq PairedDepthBench::sha("$out/attribution/$name");
    }
    PairedDepthBench::save("$out/attribution/manifest.json",$other);
    PairedDepthBench::save("$out/attribution/status.json",$state);
    PairedDepthBench::save("$out/attribution/results.json",$tr);
    $manifest->{attribution_bundle}='attribution/';
    PairedDepthBench::save("$out/attribution-comparison.json",\@comparison);
    context_increments($out,$tr);
    return \@comparison;
}

sub context_increments {
    my($out,$rows)=@_;
    my @report;
    for my $placement(qw(host vram)) {
        my %group;
        for my $r(@$rows) {push @{$group{$r->{depth}}},$r
            if ($r->{status}//'') eq 'complete' && ($r->{placement}//'') eq $placement;}
        my @depths=sort {$a<=>$b} keys %group;next unless @depths>1;
        my @comparisons=map {[$depths[0],$_]} @depths[1..$#depths];
        push @comparisons,[3100,8000] if exists($group{3100}) && exists($group{8000}) && $depths[0]!=3100;
        for my $comparison(@comparisons) {
            my($base,$depth)=@$comparison;
            my @die;
            for my $d(0..$#{$group{$base}[0]{device_publication_decode_ms_by_die}}) {
                my(%means,%device);
                for my $n($base,$depth) {
                    my $rr=$group{$n};
                    $device{$n}=PairedDepthBench::sum([map {$_->{device_publication_decode_ms_by_die}[$d]/$_->{decode_intervals}} @$rr])/@$rr;
                    for my $r(@$rr) {
                        my $ops=$r->{trace_op_mean_us_by_die}[$d];
                        die "missing phase partition for context reconciliation\n" unless ref($ops) eq 'HASH';
                        for my $op(keys %$ops) {
                            for my $category(keys %{$ops->{$op}}) {
                                $means{$n}{$op}{$category}+=$ops->{$op}{$category}/1000/@$rr;
                            }
                        }
                    }
                }
                my %ops=map {$_=>1}(keys %{$means{$base}//{}},keys %{$means{$depth}//{}});
                my(%delta,$partition_delta);
                $partition_delta=0;
                for my $op(sort keys %ops) {
                    my %categories=map {$_=>1}(keys %{$means{$base}{$op}//{}},keys %{$means{$depth}{$op}//{}});
                    for my $c(sort keys %categories) {
                        my $v=($means{$depth}{$op}{$c}//0)-($means{$base}{$op}{$c}//0);
                        $delta{$op}{$c}=$v;$partition_delta+=$v;
                    }
                }
                my $span=$device{$depth}-$device{$base};
                push @die,{die=>$d,publication_interval_delta_ms=>$span,
                    opcode_partition_delta_ms=>\%delta,partition_delta_ms=>$partition_delta,
                    between_record_and_publication_delta_ms=>$span-$partition_delta};
            }
            my %host;
            for my $metric(qw(client_decode_ms engine_observed_decode_ms)) {
                my %mean;
                for my $n($base,$depth) {my $rr=$group{$n};
                    $mean{$n}=PairedDepthBench::sum([map {$_->{$metric}/$_->{decode_intervals}} @$rr])/@$rr;}
                $host{$metric}=$mean{$depth}-$mean{$base};
            }
            push @report,{placement=>$placement,baseline_depth=>0+$base,depth=>0+$depth,
                host_interval_delta_ms=>\%host,dies=>\@die,
                interpretation=>'Instrumented intervals in each local die clock. Per-die partitions are parallel and are never summed. The between-record term is an explicit remainder, not a causal critical-path claim.'};
        }
    }
    PairedDepthBench::save("$out/context-increments.json",\@report);
}

sub counter_pointer {
    my ($node,$pointer) = @_;
    die "invalid counter JSON pointer\n" unless defined($pointer) && !ref($pointer)
        && ($pointer eq '' || $pointer =~ m{\A/});
    return $node if $pointer eq '';
    my @parts=$pointer eq '/' ? ('') : split(m{/},substr($pointer,1),-1);
    for my $part (@parts) {
        die "invalid counter JSON pointer escape\n" if $part =~ /~(?![01])/;
        $part =~ s/~1/\//g; $part =~ s/~0/~/g;
        if (ref($node) eq 'HASH') {
            die "counter JSON pointer member absent\n" unless exists($node->{$part});
            $node=$node->{$part};
        } elsif (ref($node) eq 'ARRAY') {
            die "invalid counter JSON pointer index\n" unless $part =~ /\A(?:0|[1-9][0-9]*)\z/
                && $part < @$node;
            $node=$node->[$part];
        } else { die "counter JSON pointer traverses a scalar\n"; }
    }
    return $node;
}

sub profiler_record {
    my($record,$manifest,$out)=@_;
    BenchManifest::safe($record,'profiler record');
    die "profiler record must identify a separate collection\n"
        unless ref($record) eq 'HASH' && ($record->{schema}//0)==2
        && ($record->{scope}//'') eq 'separate_profiler';
    matching_runtime($manifest,$record,'throughput/profiler');
    for my $name(qw(host vram)) {
        my $c=$record->{configs}{$name};
        die "profiler runtime config absent: $name\n" unless ref($c) eq 'HASH';
        PairedDepthBench::matched($manifest->{configs}{$name},$c,
            {map {$_=>1}qw(trace_tokens trace_ctas trace_schema cubin cubin_fnv)});
    }
    die "profiler instrumentation settings missing\n" unless ref($record->{instrumentation}) eq 'HASH'
        && keys %{$record->{instrumentation}};
    die "profiler command records missing\n" unless ref($record->{commands}) eq 'ARRAY' && @{$record->{commands}};
    for my $c(@{$record->{commands}}) {
        die "failed or incomplete profiler command\n" unless ref($c) eq 'HASH' && ref($c->{argv}) eq 'ARRAY'
            && @{$c->{argv}} && uint($c->{exit}) && $c->{exit}==0
            && !(grep {!defined($_)||ref($_)||!length($_)} @{$c->{argv}});
    }
    die "profiler raw files missing\n" unless ref($record->{raw_files}) eq 'ARRAY' && @{$record->{raw_files}};
    my $index=0; my %verified;
    make_path("$out/profiler") if defined($out);
    for my $raw(@{$record->{raw_files}}) {
        die "invalid profiler raw identity\n" unless ref($raw) eq 'HASH';
        my $actual=PairedDepthBench::identity($raw->{path});
        die "profiler raw file identity mismatch\n" unless $actual->{sha256} eq ($raw->{sha256}//'')
            && defined($raw->{bytes}) && $actual->{bytes}==$raw->{bytes};
        my $verified_path=$raw->{path};
        if(defined($out)) {
            my $path='profiler/'.(++$index).'-'.basename($raw->{path});
            copy($raw->{path},"$out/$path") or die "copy profiler raw file: $!\n";
            die "profiler raw file changed while copying\n" unless PairedDepthBench::sha("$out/$path") eq $raw->{sha256};
            $raw->{archive_path}=$path;
            $verified_path="$out/$path";
        }
        $verified{$raw->{sha256}}=$verified_path;
    }
    die "profiler counters missing\n" unless ref($record->{counters}) eq 'ARRAY' && @{$record->{counters}};
    my (%decoded,%seen,@accepted);
    for my $c(@{$record->{counters}}) {
        die "invalid profiler counter\n" unless ref($c) eq 'HASH'
            && defined($c->{name}) && !ref($c->{name}) && length($c->{name})
            && defined($c->{unit}) && !ref($c->{unit}) && length($c->{unit});
        die "profiler counter value must come from raw evidence\n" if exists($c->{value});
        my $e=$c->{evidence};
        die "profiler counter evidence missing\n" unless ref($e) eq 'HASH'
            && defined($e->{raw_sha256}) && !ref($e->{raw_sha256})
            && exists($verified{$e->{raw_sha256}})
            && defined($e->{json_pointer}) && !ref($e->{json_pointer});
        my $sha=$e->{raw_sha256};
        die "duplicate profiler counter evidence\n" if $seen{"$sha\0$e->{json_pointer}"}++;
        if (!exists($decoded{$sha})) {
            my $bytes=PairedDepthBench::read_text($verified{$sha});
            die "profiler counter bytes changed\n" unless sha256_hex($bytes) eq $sha;
            $decoded{$sha}=$json->decode($bytes);
        }
        my $raw=counter_pointer($decoded{$sha},$e->{json_pointer});
        die "profiler counter pointer must select a record\n" unless ref($raw) eq 'HASH';
        for my $field(qw(name unit)) {
            die "profiler counter $field disagrees with raw evidence\n"
                unless defined($raw->{$field}) && !ref($raw->{$field}) && $raw->{$field} eq $c->{$field};
        }
        my $v=$raw->{value};
        die "invalid raw profiler counter value\n" unless defined($v) && !ref($v)
            && $v=~/\A-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\z/
            && isfinite(0+$v);
        push @accepted,{name=>$raw->{name},unit=>$raw->{unit},value=>$v,evidence=>{%$e}};
    }
    return {%$record,counters=>\@accepted};
}
1;
