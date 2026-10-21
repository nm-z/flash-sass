#!/usr/bin/env perl
use strict;
use warnings;
use B ();
use JSON::PP ();
use Digest::SHA qw(sha256_hex);
use File::Copy qw(copy);
use File::Spec ();
use POSIX qw(isfinite strftime);
use Scalar::Util qw(looks_like_number);

my $JSON = JSON::PP->new->utf8->canonical;
my $TRUE = JSON::PP::true;
my $FALSE = JSON::PP::false;
my $NOTE =
    'Metadata and independence are producer declared. File checksums bind the '
  . 'supplied artifacts but do not attest their origin, independent execution, '
  . 'or original-checkpoint quality.';

sub fail { die $_[0] . "\n" }
sub boolean { return $_[0] ? $TRUE : $FALSE }
sub is_true { my ($v) = @_; return JSON::PP::is_bool($v) && $v }
sub is_string {
    my ($v) = @_;
    return 0 if !defined($v) || ref($v);
    return (B::svref_2object(\$v)->FLAGS & B::SVp_POK()) != 0;
}
sub nonempty_string {
    my ($v) = @_;
    return is_string($v) && length($v) != 0;
}
sub json_uint {
    my ($v, $max) = @_;
    return 0 if !defined($v) || ref($v);
    my $flags = B::svref_2object(\$v)->FLAGS;
    return 0 unless $flags & (B::SVp_IOK() | B::SVp_NOK());
    return isfinite($v) && $v >= 0 && $v <= $max && int($v) == $v;
}
sub value_at {
    my ($v, @keys) = @_;
    for my $key (@keys) {
        return undef unless ref($v) eq 'HASH';
        $v = $v->{$key};
    }
    return $v;
}
sub good_hex {
    my ($v, $pattern) = @_;
    return is_string($v) && $v =~ $pattern;
}
sub positive_argument {
    my ($raw, $name, $max) = @_;
    fail("$name must be a positive integer")
        unless defined($raw) && $raw =~ /\A[0-9]+\z/;
    my $s = $raw;
    $s =~ s/\A0+//;
    fail("$name must be a positive integer") unless length($s);
    fail("$name is too large")
        if length($s) > length($max)
        || (length($s) == length($max) && $s gt $max);
    return 0 + $s;
}
sub bound_argument {
    my ($raw, $name) = @_;
    fail("$name must be finite and nonnegative")
        unless defined($raw) && looks_like_number($raw);
    my $v = 0 + $raw;
    fail("$name must be finite and nonnegative")
        unless isfinite($v) && $v >= 0;
    return $v;
}
sub options {
    my ($mode, $args) = @_;
    my %allowed = map { $_ => 1 } (
        $mode eq 'freeze'
        ? qw(kind ids logits vocab-size vocab-ids metadata raw-info out)
        : qw(candidate reference out top-k diagnostic max-mean-kl max-rmse
             max-abs-logit min-top1-agreement max-mean-nll-increase)
    );
    my %o;
    while (@$args) {
        my $a = shift @$args;
        fail("unknown option: $a")
            unless $a =~ /\A--(.+)\z/ && $allowed{$1};
        my $key = $1;
        fail("duplicate option: --$key") if exists $o{$key};
        if ($key eq 'diagnostic') {
            $o{$key} = 1;
            next;
        }
        fail("missing value for --$key")
            unless @$args && $args->[0] !~ /\A--/;
        $o{$key} = shift @$args;
    }
    my @required = $mode eq 'freeze'
        ? qw(kind ids logits vocab-size out)
        : qw(candidate reference);
    for my $key (@required) {
        fail("missing required option: --$key")
            unless exists $o{$key} && length($o{$key});
    }
    return \%o;
}
sub file_size {
    my ($path) = @_;
    fail("not a regular file: $path") unless -f $path;
    my @st = stat($path);
    fail("cannot stat $path: $!") unless @st;
    return $st[7];
}
sub slurp {
    my ($path) = @_;
    open(my $fh, '<:raw', $path) or fail("cannot open $path: $!");
    local $/;
    my $s = <$fh>;
    $s = '' unless defined $s;
    close($fh) or fail("cannot close $path: $!");
    return $s;
}
sub write_bytes {
    my ($path, $bytes) = @_;
    open(my $fh, '>:raw', $path) or fail("cannot write $path: $!");
    print {$fh} $bytes or fail("cannot write $path: $!");
    close($fh) or fail("cannot close $path: $!");
}
sub read_json {
    my ($path) = @_;
    my $v = eval { $JSON->decode(slurp($path)) };
    fail("invalid JSON in $path: $@") if $@;
    return $v;
}
sub write_json {
    my ($path, $v) = @_;
    write_bytes($path, $JSON->encode($v) . "\n");
}
sub file_sha256 {
    my ($path) = @_;
    file_size($path);
    open(my $fh, '<:raw', $path) or fail("cannot open $path: $!");
    my $sha = Digest::SHA->new(256);
    $sha->addfile($fh);
    close($fh) or fail("cannot close $path: $!");
    return $sha->hexdigest;
}
sub path_in { return File::Spec->catfile($_[0], $_[1]) }
sub artifact {
    my ($path) = @_;
    return { bytes => file_size($path), sha256 => file_sha256($path) };
}
sub read_u32 {
    my ($path, $label) = @_;
    my $n = file_size($path);
    fail("$label must be nonempty and divisible by 4")
        unless $n > 0 && $n % 4 == 0;
    return [unpack('V*', slurp($path))];
}
sub validate_vocab {
    my ($vocab, $v) = @_;
    fail("vocabulary length does not match vocab_size") unless @$vocab == $v;
    my %seen;
    for my $id (@$vocab) {
        fail("duplicate vocabulary token ID: $id") if $seen{$id}++;
    }
}
sub validate_shape {
    my ($t, $v, $bytes) = @_;
    my $row_bytes = 4 * $v;
    my $max_positions;
    { use integer; $max_positions = 9223372036854775807 / $row_bytes; }
    fail("logits shape overflows the supported file size") if $t > $max_positions;
    fail("logits shape does not match positions * vocab_size * 4")
        unless $bytes == $t * $row_bytes;
}
sub missing_provenance {
    my ($kind, $meta) = @_;
    my @missing;
    push @missing, 'teacher_forced'
        unless is_true(value_at($meta, 'teacher_forced'));
    push @missing, 'base_model.id'
        unless nonempty_string(value_at($meta, 'base_model', 'id'));
    push @missing, 'base_model.revision'
        unless good_hex(value_at($meta, 'base_model', 'revision'),
                        qr/\A(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\z/);
    push @missing, 'tokenizer.sha256'
        unless good_hex(value_at($meta, 'tokenizer', 'sha256'),
                        qr/\A[0-9a-fA-F]{64}\z/);
    my $files = value_at($meta, 'checkpoint', 'files');
    if (ref($files) ne 'ARRAY' || !@$files) {
        push @missing, 'checkpoint.files';
    } else {
        for my $i (0 .. $#$files) {
            my $f = $files->[$i];
            if (ref($f) ne 'HASH') {
                push @missing, "checkpoint.files[$i]";
                next;
            }
            push @missing, "checkpoint.files[$i].sha256"
                unless good_hex($f->{sha256}, qr/\A[0-9a-fA-F]{64}\z/);
            push @missing, "checkpoint.files[$i].path/source_uri"
                unless nonempty_string($f->{path}) || nonempty_string($f->{source_uri});
        }
    }
    push @missing, 'implementation.name'
        unless nonempty_string(value_at($meta, 'implementation', 'name'));
    push @missing, 'implementation.revision'
        unless good_hex(value_at($meta, 'implementation', 'revision'),
                        qr/\A(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\z/);
    my $command = value_at($meta, 'command');
    push @missing, 'command'
        unless nonempty_string($command)
        || (ref($command) eq 'ARRAY' && @$command);
    push @missing, 'implementation.independent_reference=true'
        if $kind eq 'reference'
        && !is_true(value_at($meta, 'implementation', 'independent_reference'));
    return \@missing;
}
sub verify_local_checkpoint_files {
    my ($meta) = @_;
    my $files = value_at($meta, 'checkpoint', 'files');
    return unless defined $files;
    fail("metadata.checkpoint.files must be an array") unless ref($files) eq 'ARRAY';
    for my $f (@$files) {
        fail("metadata.checkpoint.files entries must be objects")
            unless ref($f) eq 'HASH';
        my $p = $f->{path};
        if (nonempty_string($p) && -f $p) {
            my $sha = file_sha256($p);
            if (exists($f->{sha256}) && defined($f->{sha256})) {
                fail("checkpoint sha256 mismatch: $p")
                    unless is_string($f->{sha256})
                    && lc($f->{sha256}) eq $sha;
            }
            $f->{bytes} = file_size($p);
            $f->{sha256} = $sha;
            $f->{verified_local} = $TRUE;
        } else {
            $f->{verified_local} = $FALSE;
        }
    }
}
sub freeze_capture {
    my ($o) = @_;
    my $kind = $o->{kind};
    fail("--kind must be candidate or reference")
        unless $kind eq 'candidate' || $kind eq 'reference';
    my $v = positive_argument($o->{'vocab-size'}, '--vocab-size', '4294967295');
    my $ids = read_u32($o->{ids}, 'IDs');
    my $t = @$ids;
    validate_shape($t, $v, file_size($o->{logits}));
    my $vocab = exists($o->{'vocab-ids'})
        ? read_u32($o->{'vocab-ids'}, 'vocabulary IDs')
        : [0 .. $v - 1];
    validate_vocab($vocab, $v);
    my $meta = exists($o->{metadata}) ? read_json($o->{metadata}) : {};
    fail("metadata must be a JSON object") unless ref($meta) eq 'HASH';
    if (defined($meta->{kind})) {
        fail("metadata kind does not match capture kind")
            unless is_string($meta->{kind}) && $meta->{kind} eq $kind;
    }
    verify_local_checkpoint_files($meta);
    my $raw;
    if (exists($o->{'raw-info'})) {
        $raw = read_json($o->{'raw-info'});
        fail("raw-info must be a JSON object") unless ref($raw) eq 'HASH';
        fail("raw-info teacher_forced must be true") unless is_true($raw->{teacher_forced});
        fail("raw-info positions do not match IDs")
            unless json_uint($raw->{positions}, 9223372036854775807)
            && $raw->{positions} == $t;
        fail("raw-info vocab_size does not match")
            unless json_uint($raw->{vocab_size}, 4294967295)
            && $raw->{vocab_size} == $v;
        fail("raw-info byte_order must be little")
            unless nonempty_string($raw->{byte_order}) && $raw->{byte_order} eq 'little';
        # Historical raw-model-capture.v1 files predate dtype/completion fields.
        # Keep those replayable, but never reinterpret an explicit contradiction.
        fail("raw-info dtype must be <f4")
            if exists($raw->{dtype})
            && !(nonempty_string($raw->{dtype}) && $raw->{dtype} eq '<f4');
        fail("raw-info complete must be true when supplied")
            if exists($raw->{complete}) && !is_true($raw->{complete});
        if (($raw->{schema} // '') eq 'flash-sass.independent-gguf-capture.v1'
            || ($raw->{schema} // '') eq 'flash-sass.raw-model-capture.v2') {
            fail("current raw-info schema requires dtype and completion marker")
                unless exists($raw->{dtype}) && is_true($raw->{complete});
        }
    }
    my $out = $o->{out};
    mkdir($out) or fail("cannot create new capture directory $out: $!");
    copy($o->{ids}, path_in($out, 'ids.u32')) or fail("cannot copy IDs: $!");
    copy($o->{logits}, path_in($out, 'logits.f32')) or fail("cannot copy logits: $!");
    write_bytes(path_in($out, 'vocab_ids.u32'), pack('V*', @$vocab));
    my %artifacts = map { $_ => artifact(path_in($out, $_)) }
        qw(ids.u32 logits.f32 vocab_ids.u32);
    if (defined $raw) {
        write_json(path_in($out, 'raw_capture_info.json'), $raw);
        $artifacts{'raw_capture_info.json'} =
            artifact(path_in($out, 'raw_capture_info.json'));
    }
    my $missing = missing_provenance($kind, $meta);
    my $manifest = {
        schema => 'flash-sass.frozen-logits.v1', kind => $kind,
        positions => $t, vocab_size => $v, dtype => '<f4', byte_order => 'little',
        metadata => $meta, created_utc => strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()),
        missing_provenance => $missing, provenance_note => $NOTE,
        artifacts => \%artifacts,
    };
    write_json(path_in($out, 'manifest.json'), $manifest);
    print $JSON->encode({
        capture => $out,
        manifest_sha256 => file_sha256(path_in($out, 'manifest.json')),
        missing_provenance => $missing, quality_validation => $FALSE,
    }), "\n";
    return 0;
}
sub load_capture {
    my ($dir, $expected_kind) = @_;
    my $manifest_path = path_in($dir, 'manifest.json');
    my $manifest_bytes = slurp($manifest_path);
    my $m = eval { $JSON->decode($manifest_bytes) };
    fail("invalid manifest JSON: $@") if $@;
    fail("manifest must be a JSON object") unless ref($m) eq 'HASH';
    fail("unsupported capture schema")
        unless is_string($m->{schema}) && $m->{schema} eq 'flash-sass.frozen-logits.v1';
    fail("capture kind must be $expected_kind")
        unless is_string($m->{kind}) && $m->{kind} eq $expected_kind;
    fail("capture dtype must be <f4")
        unless is_string($m->{dtype}) && $m->{dtype} eq '<f4';
    fail("capture byte_order must be little")
        unless is_string($m->{byte_order}) && $m->{byte_order} eq 'little';
    fail("manifest positions must be a positive integer")
        unless json_uint($m->{positions}, 9223372036854775807) && $m->{positions} > 0;
    fail("manifest vocab_size must be a positive u32 integer")
        unless json_uint($m->{vocab_size}, 4294967295) && $m->{vocab_size} > 0;
    fail("manifest metadata must be an object") unless ref($m->{metadata}) eq 'HASH';
    fail("manifest artifacts must be an object") unless ref($m->{artifacts}) eq 'HASH';
    my $artifacts = $m->{artifacts};
    for my $name (keys %$artifacts) {
        fail("artifact name must be a plain filename")
            if !length($name) || $name eq '.' || $name eq '..' || $name =~ m{[\\/\0]};
        my $a = $artifacts->{$name};
        fail("artifact declaration must be an object: $name") unless ref($a) eq 'HASH';
        fail("artifact bytes must be a nonnegative integer: $name")
            unless json_uint($a->{bytes}, 9223372036854775807);
        fail("artifact sha256 must contain 64 hexadecimal digits: $name")
            unless good_hex($a->{sha256}, qr/\A[0-9a-fA-F]{64}\z/);
        my $p = path_in($dir, $name);
        fail("artifact size mismatch: $name") unless file_size($p) == $a->{bytes};
        fail("artifact sha256 mismatch: $name") unless file_sha256($p) eq lc($a->{sha256});
    }
    for my $name (qw(ids.u32 logits.f32 vocab_ids.u32)) {
        fail("missing required artifact: $name") unless exists $artifacts->{$name};
    }
    my $ids = read_u32(path_in($dir, 'ids.u32'), 'IDs');
    my $vocab = read_u32(path_in($dir, 'vocab_ids.u32'), 'vocabulary IDs');
    fail("IDs length does not match positions") unless @$ids == $m->{positions};
    validate_vocab($vocab, $m->{vocab_size});
    validate_shape($m->{positions}, $m->{vocab_size}, file_size(path_in($dir, 'logits.f32')));
    my @order = sort { $vocab->[$a] <=> $vocab->[$b] } 0 .. $#$vocab;
    my @sorted_vocab = map { $vocab->[$_] } @order;
    return {
        dir => $dir, manifest => $m,
        manifest_sha256 => sha256_hex($manifest_bytes),
        ids => $ids, order => \@order, vocab => \@sorted_vocab,
        missing => missing_provenance($expected_kind, $m->{metadata}),
    };
}
sub ksum {
    my ($values) = @_;
    my ($sum, $correction) = (0.0, 0.0);
    for my $v (@$values) {
        my $y = $v - $correction;
        my $next = $sum + $y;
        $correction = ($next - $sum) - $y;
        $sum = $next;
    }
    return $sum;
}
sub read_row {
    my ($fh, $v, $label, $sha) = @_;
    my $want = 4 * $v;
    my $buf = '';
    while (length($buf) < $want) {
        my $n = read($fh, my $part, $want - length($buf));
        fail("cannot read $label logits: $!") unless defined $n;
        fail("unexpected EOF in $label logits") unless $n;
        $buf .= $part;
    }
    $sha->add($buf);
    return [unpack('f<*', $buf)];
}
sub verify_consumed_capture {
    my ($capture, $logit_sha) = @_;
    my $artifacts = $capture->{manifest}{artifacts};
    fail("consumed logits sha256 mismatch")
        unless $logit_sha->hexdigest eq lc($artifacts->{'logits.f32'}{sha256});
    fail("consumed input IDs sha256 mismatch")
        unless sha256_hex(pack('V*', @{$capture->{ids}}))
            eq lc($artifacts->{'ids.u32'}{sha256});
    # order maps each sorted vocabulary column back to its original position.
    my @original_vocab;
    for my $i (0 .. $#{$capture->{order}}) {
        $original_vocab[$capture->{order}[$i]] = $capture->{vocab}[$i];
    }
    fail("consumed vocabulary IDs sha256 mismatch")
        unless sha256_hex(pack('V*', @original_vocab))
            eq lc($artifacts->{'vocab_ids.u32'}{sha256});
    fail("manifest changed during comparison")
        unless file_sha256(path_in($capture->{dir}, 'manifest.json'))
            eq $capture->{manifest_sha256};
    for my $name (keys %$artifacts) {
        my $path = path_in($capture->{dir}, $name);
        fail("artifact changed during comparison: $name")
            unless file_size($path) == $artifacts->{$name}{bytes}
            && file_sha256($path) eq lc($artifacts->{$name}{sha256});
    }
}
sub log_softmax {
    my ($row) = @_;
    my $max = $row->[0];
    for my $v (@$row) { $max = $v if $v > $max }
    my @terms = map { exp($_ - $max) } @$row;
    my $normalizer = log(ksum(\@terms));
    return [map { ($_ - $max) - $normalizer } @$row];
}
sub finite_exp {
    my ($v) = @_;
    return undef if $v > log(1.7976931348623157e308);
    my $x = exp($v);
    return isfinite($x) ? $x : undef;
}
sub emit_report {
    my ($report, $out) = @_;
    my $bytes = $JSON->encode($report) . "\n";
    write_bytes($out, $bytes) if defined $out;
    print $bytes;
}
sub compare_capture {
    my ($o) = @_;
    my $requested_k = exists($o->{'top-k'})
        ? positive_argument($o->{'top-k'}, '--top-k', '9223372036854775807') : 10;
    my @rules = (
        ['max-mean-kl', 'mean_reference_to_candidate_kl', 'max'],
        ['max-rmse', 'logit_rmse', 'max'],
        ['max-abs-logit', 'maximum_absolute_logit_error', 'max'],
        ['min-top1-agreement', 'top1_agreement', 'min'],
        ['max-mean-nll-increase', 'mean_nll_increase', 'max'],
    );
    my %limits;
    for my $rule (@rules) {
        my ($option) = @$rule;
        $limits{$option} = bound_argument($o->{$option}, "--$option") if exists $o->{$option};
    }
    fail("--min-top1-agreement must not exceed 1")
        if exists($limits{'min-top1-agreement'}) && $limits{'min-top1-agreement'} > 1;
    my $c = load_capture($o->{candidate}, 'candidate');
    my $r = load_capture($o->{reference}, 'reference');
    my $t = $c->{manifest}{positions};
    my $v = $c->{manifest}{vocab_size};
    fail("candidate and reference positions differ") unless $t == $r->{manifest}{positions};
    fail("candidate and reference vocabulary sizes differ") unless $v == $r->{manifest}{vocab_size};
    my $k = $requested_k < $v ? $requested_k : $v;
    for my $i (0 .. $v - 1) {
        fail("candidate and reference vocabulary sets differ")
            unless $c->{vocab}[$i] == $r->{vocab}[$i];
    }
    my %column = map { $c->{vocab}[$_] => $_ } 0 .. $v - 1;
    for my $i (0 .. $t - 1) {
        fail("candidate and reference input IDs differ at position $i")
            unless $c->{ids}[$i] == $r->{ids}[$i];
        fail("input token ID is absent from vocabulary at position $i")
            unless exists $column{$c->{ids}[$i]};
    }
    for my $path (['base_model', 'id'], ['base_model', 'revision'], ['tokenizer', 'sha256']) {
        my $cv = value_at($c->{manifest}{metadata}, @$path);
        my $rv = value_at($r->{manifest}{metadata}, @$path);
        fail("candidate and reference provenance mismatch: " . join('.', @$path))
            if nonempty_string($cv) && nonempty_string($rv) && $cv ne $rv;
    }
    my %common = (
        schema => 'flash-sass.logit-comparison.v1',
        independent_checkpoint_evidence => $FALSE,
        provenance_status => 'producer_declared_unattested',
        scope => 'Numerical comparison of supplied teacher-forced rows only.',
        provenance_note => $NOTE,
        missing_provenance => { candidate => $c->{missing}, reference => $r->{missing} },
        candidate_manifest_sha256 => $c->{manifest_sha256},
        reference_manifest_sha256 => $r->{manifest_sha256},
    );
    my $incomplete = @{$c->{missing}} || @{$r->{missing}};
    open(my $cf, '<:raw', path_in($c->{dir}, 'logits.f32')) or fail("cannot open candidate logits: $!");
    open(my $rf, '<:raw', path_in($r->{dir}, 'logits.f32')) or fail("cannot open reference logits: $!");
    my $csha = Digest::SHA->new(256);
    my $rsha = Digest::SHA->new(256);
    my %a = map { $_ => 0.0 } qw(
        sq absolute maximum ref_sq centered_sq cosine cosine_count kl
        max_probability top1 topk cnll rnll dnll crank rrank
    );
    my @positions;
    for my $position (0 .. $t - 1) {
        my $cr = read_row($cf, $v, 'candidate', $csha);
        my $rr = read_row($rf, $v, 'reference', $rsha);
        my $cnon = grep { !isfinite($_) } @$cr;
        my $rnon = grep { !isfinite($_) } @$rr;
        if ($cnon || $rnon) {
            emit_report({
                %common, status => 'failed_numerical_contract', quality_pass => $FALSE,
                position => $position, candidate_nonfinite => $cnon, reference_nonfinite => $rnon,
            }, $o->{out});
            return 3;
        }
        my @x = map { $cr->[$_] + 0.0 } @{$c->{order}};
        my @y = map { $rr->[$_] + 0.0 } @{$r->{order}};
        my @difference;
        my ($xx, $yy, $xy) = (0.0, 0.0, 0.0);
        for my $i (0 .. $v - 1) {
            # Subtract before centering: a shared huge logit must not erase a
            # small candidate/reference difference in another column.
            my $d = $x[$i] - $y[$i];
            push @difference, $d;
            $a{sq} += $d * $d;
            $a{absolute} += abs($d);
            $a{maximum} = abs($d) if abs($d) > $a{maximum};
            $xx += $x[$i] * $x[$i];
            $yy += $y[$i] * $y[$i];
            $xy += $x[$i] * $y[$i];
        }
        my $mean_difference = ksum(\@difference) / $v;
        for my $d (@difference) {
            my $centered = $d - $mean_difference;
            $a{centered_sq} += $centered * $centered;
        }
        $a{ref_sq} += $yy;
        if ($xx > 0 && $yy > 0) {
            $a{cosine} += $xy / sqrt($xx * $yy);
            ++$a{cosine_count};
        }
        my $lx = log_softmax(\@x);
        my $ly = log_softmax(\@y);
        my @kl_terms;
        for my $i (0 .. $v - 1) {
            my $px = exp($lx->[$i]);
            my $py = exp($ly->[$i]);
            push @kl_terms, $py * ($ly->[$i] - $lx->[$i]);
            my $pd = abs($px - $py);
            $a{max_probability} = $pd if $pd > $a{max_probability};
        }
        my $kl = ksum(\@kl_terms);
        $kl = 0.0 if $kl < 0;
        $a{kl} += $kl;
        # Indices are ordered by token ID, giving deterministic tie breaks.
        my @cx = sort { $x[$b] <=> $x[$a] || $a <=> $b } 0 .. $v - 1;
        my @ry = sort { $y[$b] <=> $y[$a] || $a <=> $b } 0 .. $v - 1;
        my %chosen = map { $cx[$_] => 1 } 0 .. $k - 1;
        my $overlap = 0;
        for my $i (0 .. $k - 1) { ++$overlap if $chosen{$ry[$i]} }
        my $agrees = $cx[0] == $ry[0];
        $a{top1} += $agrees ? 1 : 0;
        $a{topk} += $overlap / $k;
        my $row = {
            position => $position, input_id => $c->{ids}[$position],
            candidate_top1 => $c->{vocab}[$cx[0]], reference_top1 => $r->{vocab}[$ry[0]],
            top1_agrees => boolean($agrees), topk_overlap => $overlap / $k,
            reference_to_candidate_kl => $kl,
        };
        if ($position + 1 < $t) {
            my $next_id = $c->{ids}[$position + 1];
            my $target = $column{$next_id};
            my $cnll = -$lx->[$target];
            my $rnll = -$ly->[$target];
            # Difference form also preserves a small NLL change when both
            # individual NLLs round to the same very large number.
            my $dnll = (($x[$cx[0]] - $y[$ry[0]])
                        - ($x[$target] - $y[$target]))
                       + ($ly->[$ry[0]] - $lx->[$cx[0]]);
            my ($crank, $rrank) = (1, 1);
            for my $i (0 .. $v - 1) {
                ++$crank if $x[$i] > $x[$target]
                    || ($x[$i] == $x[$target] && $i < $target);
                ++$rrank if $y[$i] > $y[$target]
                    || ($y[$i] == $y[$target] && $i < $target);
            }
            $row->{next_id} = $next_id;
            $row->{reference_nll} = $rnll;
            $row->{candidate_nll} = $cnll;
            $row->{nll_increase} = $dnll;
            $row->{reference_target_rank} = $rrank;
            $row->{candidate_target_rank} = $crank;
            $a{cnll} += $cnll; $a{rnll} += $rnll; $a{dnll} += $dnll;
            $a{crank} += $crank; $a{rrank} += $rrank;
        }
        push @positions, $row;
    }
    close($cf) or fail("cannot close candidate logits: $!");
    close($rf) or fail("cannot close reference logits: $!");
    verify_consumed_capture($c, $csha);
    verify_consumed_capture($r, $rsha);
    $common{consumed_artifact_hashes_verified} = $TRUE;
    $common{input_hashes_match_before_after_comparison} = $TRUE;
    my $elements = $t * $v;
    my %metrics = (
        positions => $t, vocabulary => $v,
        logit_rmse => sqrt($a{sq} / $elements),
        mean_absolute_logit_error => $a{absolute} / $elements,
        maximum_absolute_logit_error => $a{maximum},
        relative_l2_logit_error => $a{ref_sq} > 0 ? sqrt($a{sq} / $a{ref_sq}) : undef,
        centered_logit_rmse => sqrt($a{centered_sq} / $elements),
        mean_cosine_similarity => $a{cosine_count} ? $a{cosine} / $a{cosine_count} : undef,
        mean_reference_to_candidate_kl => $a{kl} / $t,
        maximum_absolute_probability_error => $a{max_probability},
        top1_agreement => $a{top1} / $t, topk => $k, mean_topk_overlap => $a{topk} / $t,
        teacher_forced_labels => $t - 1,
    );
    if ($t > 1) {
        my $labels = $t - 1;
        my $rnll = $a{rnll} / $labels;
        my $cnll = $a{cnll} / $labels;
        my $rppl = finite_exp($rnll);
        my $cppl = finite_exp($cnll);
        $metrics{reference_mean_nll} = $rnll; $metrics{candidate_mean_nll} = $cnll;
        $metrics{mean_nll_increase} = $a{dnll} / $labels;
        $metrics{reference_perplexity} = $rppl; $metrics{candidate_perplexity} = $cppl;
        $metrics{perplexity_overflow} = boolean(!defined($rppl) || !defined($cppl));
        $metrics{reference_mean_target_rank} = $a{rrank} / $labels;
        $metrics{candidate_mean_target_rank} = $a{crank} / $labels;
    }
    my (@criteria, $failed);
    $failed = 0;
    for my $rule (@rules) {
        my ($option, $metric, $sense) = @$rule;
        next unless exists $limits{$option};
        my $limit = $limits{$option};
        my $actual = $metrics{$metric};
        my $passes = defined($actual) && isfinite($actual)
            && ($sense eq 'min' ? $actual >= $limit : $actual <= $limit);
        $failed = 1 unless $passes;
        push @criteria, {
            metric => $metric, sense => $sense, limit => $limit, actual => $actual,
            passes => boolean($passes),
        };
    }
    my $status = $incomplete ? 'diagnostic_missing_provenance'
        : $failed ? 'failed_declared_limits'
        : @criteria ? 'passed_declared_limits_unattested' : 'metrics_only';
    emit_report({
        %common, status => $status, quality_pass => $failed ? $FALSE : undef,
        declared_limits_pass => @criteria ? boolean(!$failed) : undef,
        metrics => \%metrics, declared_criteria => \@criteria, positions => \@positions,
    }, $o->{out});
    return 2 if $incomplete && !$o->{diagnostic};
    return $failed ? 3 : 0;
}
my $exit_code;
my $ok = eval {
    my $mode = shift(@ARGV) // '';
    fail("usage: model_quality freeze|compare [options]")
        unless $mode eq 'freeze' || $mode eq 'compare';
    my $o = options($mode, \@ARGV);
    $exit_code = $mode eq 'freeze' ? freeze_capture($o) : compare_capture($o);
    1;
};
if (!$ok) {
    my $error = $@ || 'unknown error';
    $error =~ s/\s+\z//;
    print STDERR $JSON->encode({
        status => 'invalid_capture', error => $error, quality_pass => $FALSE,
    }), "\n";
    exit 2;
}
exit $exit_code;
