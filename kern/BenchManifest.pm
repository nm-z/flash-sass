package BenchManifest;
use strict;
use warnings;
use Cwd qw(abs_path getcwd);
use Digest::SHA;
use File::Spec;
use IO::Select;
use IPC::Open3;
use JSON::PP;
use Symbol qw(gensym);
our $JSON = JSON::PP->new->canonical->allow_nonref;
# These are the runtime getenv keys in engine/prog/flashcore/holdc/split,
# plus the separately built schema-2 trace runtime's documented settings.
our @ENV_KEYS = qw(
 FLASH_MAXT FLASH_ENGINE_MAXT FLASH_NBLK FLASH_MAXIMG FLASH_PLE_DISK
 FLASH_PLE_OFF FLASH_PLE_CONSUME FLASH_CUBIN FLASH_KV_HOST FLASH_LAYERS
 FLASH_PREFILL_CHUNK FLASH_TWO_POSITION FLASH_WATCHDOG_S FLASH_ANCHOR_DIE
 FLASH_HOLD_WAIT_S FLASH_TRACE FLASH_TRACE_CTAS CUDA_VISIBLE_DEVICES
 CUDA_DEVICE_ORDER SPLIT_SHARES ATTN_DIE ATTN_MAP FAKE_Q6K PROG_DUP
 PROG_NOHCUP PROG_NONOBAR PROG_NORMSFUSE PROG_NORMSW FLASH_SOCK
);
my %ENV_OK = map {$_=>1} @ENV_KEYS;
sub fail { die "provenance: $_[0]\n" }
sub run {
    my($argv)=@_;my $err=gensym;
    my $pid=open3(my $in,my $out,$err,@$argv);close $in;
    my $select=IO::Select->new($out,$err);
    my %which=(fileno($out)=>'stdout',fileno($err)=>'stderr');
    my %r=(argv=>[@$argv],stdout=>'',stderr=>'');
    while($select->count) {
        for my $fh($select->can_read) {
            my $n=sysread($fh,my $buf,65536);
            if(!defined($n)){next if $!{EINTR};fail("read child: $!");}
            if(!$n){$select->remove($fh);close $fh;}
            else{$r{$which{fileno($fh)}}.=$buf;}
        }
    }
    waitpid($pid,0)==$pid or fail("wait child: $!");
    my $status=$?;$r{exit}=($status&127)?-($status&127):$status>>8;return \%r;
}
sub text {
    open my $f,'<:raw',$_[0] or fail("open $_[0]: $!");
    local $/;my $v=<$f>;close $f or fail("close $_[0]: $!");return $v//'';
}
sub json { $JSON->decode(text($_[0])) }
sub hashfile {
    open my $f,'<:raw',$_[0] or fail("open $_[0]: $!");
    my $h=Digest::SHA->new(256);$h->addfile($f);close $f or fail("close $_[0]: $!");return $h->hexdigest;
}
sub stamp {
    my($p)=@_;my $r=run(['stat','-L','-c',"%s\n%Y\n%y",'--',$p]);
    fail("stat failed $p") if $r->{exit};
    my @v=split /\n/,$r->{stdout};
    fail("invalid stat $p") unless @v>=3 && $v[0]=~/\A[0-9]+\z/
        && $v[1]=~/\A-?[0-9]+\z/ && $v[2]=~/\.([0-9]{9}) [+-][0-9]{4}\z/;
    my $fraction=$1;return {bytes=>0+$v[0],mtime_ns=>$v[1]*1_000_000_000+$fraction};
}
sub identity {
    my($p)=@_;my $path=abs_path($p);fail("unresolvable identity $p") unless defined($path)&&-f $path;
    my $a=stamp($path);my $hash=hashfile($path);my $b=stamp($path);
    fail("file changed during hash $p") unless $JSON->encode($a) eq $JSON->encode($b);
    return {path=>$path,%$b,sha256=>$hash};
}
# FNV-1a modulo 2^64 as two 32-bit halves, without floating-point overflow.
sub fnv64 {
    open my $f,'<:raw',$_[0] or fail("open $_[0]: $!");
    my($hi,$lo)=(0xcbf29ce4,0x84222325);
    while(1) {
        my $n=read($f,my $buf,65536);fail("read cubin: $!") unless defined $n;last unless $n;
        for my $byte(unpack('C*',$buf)) {
            $lo^=$byte;my $prod=$lo*435;
            $hi=($hi*435+int($prod/4294967296)+($lo<<8))&0xffffffff;
            $lo=$prod&0xffffffff;
        }
    }
    close $f or fail("close cubin: $!");return sprintf('%08x%08x',$hi,$lo);
}
sub obj { fail("$_[1] must be an object") unless ref($_[0]) eq 'HASH';$_[0] }
sub arr { fail("$_[1] must be a nonempty array") unless ref($_[0]) eq 'ARRAY' && @{$_[0]};$_[0] }
sub scalar_text { fail("$_[1] must be nonempty text") unless defined($_[0])&&!ref($_[0])&&length($_[0]);$_[0] }
sub sha256 { fail("invalid $_[1] SHA256") unless defined($_[0])&&!ref($_[0])&&$_[0]=~/\A[0-9a-f]{64}\z/ }
sub uint { fail("invalid $_[1] integer") unless defined($_[0])&&!ref($_[0])&&$_[0]=~/\A[0-9]+\z/ }
sub safe {
    my($x,$label)=@_;
    if(ref($x) eq 'HASH') {
        for my $k(keys %$x) {
            fail("secret-like key in $label") if $k=~/(?:password|credential|secret|api.?key|access.?token|auth.?token)/i;
            safe($x->{$k},$label);
        }
    } elsif(ref($x) eq 'ARRAY'){safe($_,$label) for @$x;}
    elsif(defined($x)) {
        fail("secret-like argument in $label") if $x=~/(?:--(?:password|secret|api[-_]?key|access[-_]?token)|(?:PASSWORD|SECRET|API_KEY|ACCESS_TOKEN)=)/i;
    }
}
sub argv { arr($_[0],$_[1]);scalar_text($_,$_[1]) for @{$_[0]};safe($_[0],$_[1]); }
sub verified_identity {
    my($v,$label)=@_;obj($v,$label);scalar_text($v->{path},"$label.path");sha256($v->{sha256},$label);
    my $now=identity($v->{path});fail("$label hash mismatch") unless $now->{sha256} eq $v->{sha256};
    for my $k(qw(bytes mtime_ns)) {
        if(exists $v->{$k}) {uint($v->{$k},"$label.$k");fail("$label $k mismatch") unless $v->{$k}==$now->{$k};}
    }
    return $now;
}
sub conversion {
    my($v,$label)=@_;obj($v,$label);argv($v->{argv},"$label.argv");verified_identity($v->{tool},"$label.tool");
    uint($v->{exit},"$label.exit");fail("$label failed") unless $v->{exit}==0;
    uint($v->{elapsed_ns},"$label.elapsed_ns");sha256($v->{input_sha256},"$label.input");sha256($v->{output_sha256},"$label.output");return $v;
}
sub validate_build {
    my($b,$sha,$artifacts,$dirty_hash)=@_;obj($b,'build');safe($b,'build');
    fail("build source mismatch") unless ($b->{source_sha}//'') eq $sha;
    fail("build dirty source mismatch") if defined($dirty_hash) && ($b->{source_dirty_sha256}//'') ne $dirty_hash;
    arr($b->{steps},'build.steps');my(%seen,%output_roles);
    for my $s(@{$b->{steps}}) {
        obj($s,'build step');fail("unknown build step kind") unless ($s->{kind}//'')=~/\A(?:compiler|assembler)\z/;
        $seen{$s->{kind}}++;argv($s->{argv},'build argv');
        fail("build flags must be explicit array") unless ref($s->{flags}) eq 'ARRAY';
        for my $flag(@{$s->{flags}}) {
            scalar_text($flag,'build flag');fail("recorded flag absent from argv") unless grep {$_ eq $flag} @{$s->{argv}};
        }
        verified_identity($s->{tool},'build tool');uint($s->{exit},'build exit');fail("unsuccessful build") unless $s->{exit}==0;
        uint($s->{elapsed_ns},'build elapsed_ns');arr($s->{outputs},'build outputs');
        for my $output(@{$s->{outputs}}) {
            obj($output,'build output');my $role=$output->{role}//'';
            fail("unknown build output role") unless exists $artifacts->{$role};sha256($output->{sha256},'build output');
            fail("build output identity differs from artifact") unless $output->{sha256} eq $artifacts->{$role}{sha256};
            $output_roles{$role}=1;
        }
    }
    fail("compiler and assembler steps required") unless $seen{compiler}&&$seen{assembler};
    fail("build outputs must identify executable/library/cubin") unless $output_roles{executable}&&$output_roles{library}&&$output_roles{cubin};
    conversion($b->{conversion},'build conversion');return $b;
}
sub validate_weights {
    my($w)=@_;obj($w,'weights');safe($w,'weights');obj($w->{model},'model');
    scalar_text($w->{model}{name},'model name');sha256($w->{model}{sha256},'model');arr($w->{files},'weight files');
    my(%hash,%path);
    for my $f(@{$w->{files}}) {
        obj($f,'weight file');uint($f->{bytes},'weight bytes');uint($f->{mtime_ns},'weight mtime_ns');
        my $id=verified_identity($f,'weight file');fail("duplicate weight path") if $path{$id->{path}}++;$hash{$id->{sha256}}=1;
    }
    fail("model hash must identify a verified file") unless $hash{$w->{model}{sha256}};
    my $c=conversion($w->{conversion},'weight conversion');
    fail("conversion input absent from verified files") unless $hash{$c->{input_sha256}};
    fail("converted output absent from weight files") unless $hash{$c->{output_sha256}};
    arr($w->{tensors},'tensors');my %names;
    for my $t(@{$w->{tensors}}) {
        obj($t,'tensor');scalar_text($t->{name},'tensor name');fail("duplicate tensor name") if $names{$t->{name}}++;
        sha256($t->{file_sha256},'tensor file');fail("tensor file absent") unless $hash{$t->{file_sha256}};
        sha256($t->{conversion_sha256},'tensor conversion');fail("tensor conversion mismatch") unless $t->{conversion_sha256} eq $c->{output_sha256};
    }
    return $w;
}
sub environment {
    my($env)=@_;obj($env,'launch environment');my %r;
    for my $k(keys %$env) {
        fail("unrecognized launch environment $k; add an explicit relevant allowlist entry") unless $ENV_OK{$k};
        # JSON null means unset. An empty string means set to the empty string.
        fail("invalid environment value $k") if ref($env->{$k});$r{$k}=$env->{$k};
    }
    safe(\%r,'launch environment');return \%r;
}
sub ple_conditions {
    my($v)=@_;obj($v,'ple_cache_conditions');
    for my $k(qw(mode disk_identity page_cache_policy engine_cache_policy staging_allocation_bytes preparation_argv)) {
        fail("missing PLE/cache condition $k") unless exists $v->{$k};
    }
    for my $k(qw(mode disk_identity page_cache_policy engine_cache_policy)) {
        scalar_text($v->{$k},"PLE/cache $k");fail("unknown PLE/cache condition $k") if $v->{$k}=~/\A(?:unknown|unreported|null)\z/i;
    }
    uint($v->{staging_allocation_bytes},'PLE staging allocation');
    fail("PLE preparation_argv must be explicit array") unless ref($v->{preparation_argv}) eq 'ARRAY';
    argv($_,'PLE preparation argv') for @{$v->{preparation_argv}};safe($v,'PLE/cache conditions');return $v;
}
sub matched_launch {
    my($r)=@_;obj($r,'core records');
    fail("core records must name exactly host and vram") unless join(',',sort keys %$r) eq 'host,vram';
    my %canonical;
    for my $name(qw(host vram)) {
        my $v=obj($r->{$name},"core $name");safe($v,"core $name");uint($v->{pid},'core pid');fail("invalid core pid") unless $v->{pid}>0;
        argv($v->{argv},'core argv');my $env=environment($v->{launch_environment});
        for my $k(@ENV_KEYS){fail("missing launch setting $k (use null for unset)") unless exists $env->{$k};}
        fail("wrong launch KV placement") unless ($env->{FLASH_KV_HOST}//'') eq ($name eq 'host'?'1':'0');
        fail("reload commands must be explicit array") unless ref($v->{engine_reload_commands}) eq 'ARRAY';
        argv($_,'reload argv') for @{$v->{engine_reload_commands}};
        fail("limitations must be explicit array") unless ref($v->{limitations}) eq 'ARRAY';
        scalar_text($_,'limitation') for @{$v->{limitations}};ple_conditions($v->{ple_cache_conditions});
        delete @$env{qw(FLASH_KV_HOST FLASH_SOCK)};
        my @args=@{$v->{argv}};shift @args;
        $canonical{$name}={argv=>\@args,environment=>$env,reload=>$v->{engine_reload_commands},ple_cache_conditions=>$v->{ple_cache_conditions}};
    }
    fail("unexplained launch/build/model-affecting differences") unless $JSON->encode($canonical{host}) eq $JSON->encode($canonical{vram});
    return 1;
}
sub process_identity {
    my($records,$artifacts)=@_;my %out;
    for my $name(qw(host vram)) {
        my $pid=$records->{$name}{pid};my $root="/proc/$pid";
        my $exe=identity("$root/exe");
        fail("$name running executable differs from artifact") unless $exe->{sha256} eq $artifacts->{executable}{sha256};
        my @cmd=split /\0/,text("$root/cmdline");
        fail("$name live argv differs from launch record") unless $JSON->encode(\@cmd) eq $JSON->encode($records->{$name}{argv});
        my $cwd=readlink("$root/cwd");fail("$name process working directory unavailable") unless defined $cwd;
        my %env=map {$_=>undef} @ENV_KEYS;
        # Persist only the allowlist; unrelated process credentials never leave
        # this temporary parsing buffer.
        for my $entry(split /\0/,text("$root/environ")) {
            my($k,$v)=split /=/,$entry,2;$env{$k}=$v if $ENV_OK{$k};
        }
        fail("$name launch environment differs from initial /proc environment")
            unless $JSON->encode(\%env) eq $JSON->encode($records->{$name}{launch_environment});
        my %mapped;
        for my $line(split /\n/,text("$root/maps")) {
            next unless $line=~/^\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\/.*)$/;
            my $p=$1;
            next unless $p=~m{/(?:libengine[^/]*|flashcore-\Q$pid\E-gen[0-9]+)\.so(?: \(deleted\))?\z};
            fail("$name loaded engine copy was deleted") if $p=~/ \(deleted\)\z/;
            $mapped{$p}=1;
        }
        my @engine;
        for my $path(sort keys %mapped) {
            my $id=identity($path);
            fail("$name mapped engine differs from artifact") unless $id->{sha256} eq $artifacts->{library}{sha256};
            push @engine,$id;
        }
        fail("$name engine library not independently located in process maps") unless @engine==1;
        $out{$name}={pid=>0+$pid,cwd=>$cwd,executable=>$exe,argv=>\@cmd,
            launch_environment=>\%env,mapped_engine_libraries=>\@engine,
            verification=>'Actual /proc executable, original argv/environment and private engine copy; later reload history remains operator supplied.'};
    }
    return \%out;
}
sub capture {
    my($a)=@_;obj($a,'options');my $dir=abs_path($a->{source_dir}//getcwd());fail("invalid source directory") unless defined $dir;
    my $head=run(['git','-C',$dir,'rev-parse','HEAD']);fail("cannot read source HEAD") if $head->{exit};
    my $sha=$head->{stdout};$sha=~s/\s+\z//;
    fail("source SHA does not match actual HEAD") unless $sha=~/\A[0-9a-f]{40}\z/ && ($a->{source_sha}//'') eq $sha;
    my @paths=('.');
    if(defined($a->{out})) {
        my $output=abs_path($a->{out});
        if(defined($output) && index($output,"$dir/")==0) {
            push @paths,':(exclude,literal)'.substr($output,length($dir)+1);
        }
    }
    my $status=run(['git','-C',$dir,'status','--porcelain=v1','--untracked-files=all','--',@paths]);
    fail("cannot read source status") if $status->{exit};
    fail("untracked source files must be committed before capture") if $status->{stdout}=~/^\?\?/m;
    my $diff=run(['git','-C',$dir,'diff','HEAD','--binary','--no-ext-diff','--',@paths]);
    fail("cannot read source diff") if $diff->{exit};
    my $dirty_hash=Digest::SHA::sha256_hex($diff->{stdout});
    if(defined($a->{dirty_patch})) {
        fail("provided dirty patch does not match actual HEAD diff") unless text($a->{dirty_patch}) eq $diff->{stdout};
    }
    my %art;arr($a->{artifact},'artifacts');
    for my $s(@{$a->{artifact}}) {
        fail("artifact must be ROLE=PATH") unless $s=~/\A(executable|library|cubin)=(.+)\z/s;
        my($role,$path)=($1,$2);fail("duplicate artifact role $role") if $art{$role};$art{$role}=identity($path);
    }
    fail("executable, library and cubin artifacts required") unless keys(%art)==3;
    $art{cubin}{fnv64}=fnv64($art{cubin}{path});
    my $build=validate_build(json($a->{build_record}),$sha,\%art,$dirty_hash);
    my $weights=validate_weights(json($a->{weight_manifest}));
    fail("build and weight conversion mismatch") unless $JSON->encode($build->{conversion}) eq $JSON->encode($weights->{conversion});
    my $launch=json($a->{core_records});matched_launch($launch);my $process=process_identity($launch,\%art);
    my %env=map {$_=>exists($ENV{$_})?$ENV{$_}:undef} @ENV_KEYS;
    return {schema=>'issue14.provenance.v1',
        source=>{directory=>$dir,head=>$sha,head_command=>$head,status=>$status,diff=>$diff,
            diff_sha256=>$dirty_hash,dirty=>length($diff->{stdout})?JSON::PP::true:JSON::PP::false},
        artifacts=>\%art,build=>$build,weights=>$weights,core_launch_records=>$launch,processes=>$process,environment=>\%env,
        claims=>{source_and_file_identities=>'independently captured',
            build_conversion_and_launch_history=>'operator supplied and schema checked; tool/file hashes verified',
            cubin_loading=>'runtime reported loaded FNV compared with SHA256-identified file; not GPU memory attestation'}};
}
sub validate_runtime {
    my($p,$cfg)=@_;obj($p,'provenance');obj($cfg,'runtime configs');
    fail("wrong provenance schema") unless ($p->{schema}//'') eq 'issue14.provenance.v1';
    for my $name(qw(host vram)) {
        my $c=obj($cfg->{$name},"$name config");
        for my $k(qw(rev kv_host layers nblk maxt dies requestmeta_schema tensor_schema cubin_fnv)) {
            fail("missing runtime field $name.$k") unless defined $c->{$k};
        }
        fail("wrong runtime placement") unless $c->{kv_host} eq ($name eq 'host'?'1':'0');
        fail("metadata capabilities required") unless $c->{requestmeta_schema} eq '1' && $c->{tensor_schema} eq '1';
        my($rev,$dirty)=$c->{rev}=~/\A([0-9a-f]{7,40})(-dirty)?\z/;
        fail("runtime revision does not identify captured source") unless defined($rev)
            && index($p->{source}{head},$rev)==0 && (!!$dirty)==(!!$p->{source}{dirty});
        fail("runtime loaded cubin identity differs") unless $c->{cubin_fnv} eq $p->{artifacts}{cubin}{fnv64};
        if(defined($c->{cubin}) && length($c->{cubin})) {
            my $path=File::Spec->rel2abs($c->{cubin},$p->{processes}{$name}{cwd});
            my $id=identity($path);fail("runtime cubin file differs") unless $id->{sha256} eq $p->{artifacts}{cubin}{sha256};
        }
    }
    my %a=%{$cfg->{host}};my %b=%{$cfg->{vram}};delete $a{kv_host};delete $b{kv_host};
    fail("unexplained runtime config differences") unless $JSON->encode(\%a) eq $JSON->encode(\%b);return 1;
}
# Validate the engine's in-memory tensor table, rather than inferring complete
# tensor coverage from a nonempty operator-supplied list.
sub tensor_inventory {
    my($lines,$meta,$weights)=@_;obj($meta,'tensor metadata');arr($lines,'tensor rows');
    fail("unsupported tensor schema") unless ($meta->{schema}//'') eq '1';
    uint($meta->{dies},'tensor dies');fail("empty tensor die list") unless $meta->{dies}>0;
    my %declared=map {$_->{name}=>$_} @{$weights->{tensors}};
    my(%arenas,%seen,@tensors);
    for my $line(@$lines) {
        my($kind,@words)=split /\s+/,$line;my %f;
        for my $word(@words) {
            my($k,$v)=split /=/,$word,2;fail("malformed tensor field") unless defined($v)&&!exists($f{$k});$f{$k}=$v;
        }
        uint($f{die},'tensor die');fail("tensor die outside inventory") unless $f{die}<$meta->{dies};
        if($kind eq 'arena') {
            fail("duplicate tensor arena") if exists($arenas{$f{die}});
            uint($f{$_},"arena $_") for qw(physical_die bytes tensors);
            fail("empty tensor arena") unless $f{bytes}>0 && $f{tensors}>0;
            $arenas{$f{die}}=\%f;
        } elsif($kind eq 'tensor') {
            my $arena=$arenas{$f{die}};fail("tensor precedes arena") unless $arena;
            scalar_text($f{name},'tensor name');fail("undeclared loaded tensor $f{name}") unless $declared{$f{name}};
            fail("duplicate loaded tensor") if $seen{"$f{die}:$f{name}"}++;
            uint($f{$_},"tensor $_") for qw(offset bytes type rows K stride experts);
            fail("tensor outside loaded arena") unless $f{offset}<=$arena->{bytes} && $f{bytes}<=$arena->{bytes}-$f{offset};
            push @tensors,\%f;
        } else {fail("unexpected tensor row");}
    }
    fail("missing loaded arena") unless keys(%arenas)==$meta->{dies};
    for my $d(0..$meta->{dies}-1) {
        fail("incomplete loaded tensor inventory") unless scalar(grep {$_->{die}==$d} @tensors)==$arenas{$d}{tensors};
    }
    return {metadata=>$meta,arenas=>[map {$arenas{$_}}0..$meta->{dies}-1],tensors=>\@tensors};
}
1;
