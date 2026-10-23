package ResidentConfig;
use strict;
use warnings;
use Exporter 'import';
our @EXPORT_OK = qw(parse_config validation_plan file_fnv);

my @REQUIRED = qw(rev kv_host ple_disk maxt layers nblk dies attn_dies split
    attn_map img_rows two_position prefill_chunk cubin cubin_fnv);
my $LEGACY_SHA = 'e0ce409d7f23f8ee924cf312c77eac4fbce89bfca6d180c6328f55b12e771eb2';

sub _token {
    my ($value, $name) = @_;
    die "$name is not a safe protocol token\n" unless defined($value) && !ref($value)
        && $value =~ /\A[\x21-\x3c\x3e-\x7e]+\z/;
    return "$value";
}
sub _shape {
    my ($config) = @_;
    die "Config must be a hash reference\n" unless ref($config) eq 'HASH';
    for my $key (keys %$config) {
        die "Malformed config key\n" unless $key =~ /\A[a-z][a-z0-9_]*\z/;
        _token($config->{$key}, $key);
    }
    die "Missing required config key $_\n" for grep {!exists $config->{$_}} @REQUIRED;
}
sub _uint {
    my ($value, $name) = @_;
    die "$name must be a nonnegative decimal integer\n"
        unless defined($value) && !ref($value) && $value =~ /\A(?:0|[1-9][0-9]*)\z/;
    return "$value";
}
sub _list {
    my ($value, $name) = @_;
    my @values = split(/,/, $value, -1);
    _uint($_, $name) for @values;
    return @values;
}
sub parse_config {
    my ($line) = @_;
    die "Config line must be text\n" unless defined($line) && !ref($line);
    $line =~ s/\n\z//;
    die "Config line contains control or non-ASCII characters\n" if $line =~ /[^\x20-\x7e]/;
    my @words = split(/ +/, $line, -1);
    die "Missing config response prefix\n" unless @words && shift(@words) eq 'config';
    my %config;
    for my $word (@words) {
        my ($key, $value) = $word =~ /\A([a-z][a-z0-9_]*)=([\x21-\x3c\x3e-\x7e]+)\z/;
        die "Malformed config token\n" unless defined($key);
        die "Duplicate config key $key\n" if exists $config{$key};
        $config{$key} = $value;
    }
    _shape(\%config);
    return \%config;
}
sub validation_plan {
    my ($config, $baseline_sha, $candidate_cubin) = @_;
    _shape($config);
    die "Invalid baseline engine SHA-256\n" unless defined($baseline_sha) && !ref($baseline_sha)
        && $baseline_sha =~ /\A[0-9a-f]{64}\z/;
    _token($candidate_cubin, 'candidate cubin');
    die "Candidate cubin must be an absolute path\n" unless $candidate_cubin =~ m{\A/};
    _uint($config->{$_}, $_) for qw(kv_host ple_disk maxt nblk dies img_rows two_position prefill_chunk);
    my %required_values = (layers=>'48/48', nblk=>'512', dies=>'6',
        kv_host=>'0', ple_disk=>'1', prefill_chunk=>'0');
    for my $key (sort keys %required_values) {
        die "Unsupported resident layout: $key=$config->{$key}\n"
            unless $config->{$key} eq $required_values{$key};
    }
    die "Resident maxt must be at least 512 and fit an int\n"
        if $config->{maxt} < 512 || $config->{maxt} > 2147483647;
    die "Resident img_rows must be positive and fit an unsigned int\n"
        if $config->{img_rows} < 1 || $config->{img_rows} > 4294967295;
    die "Resident two_position must be 0 or 1\n" unless $config->{two_position} =~ /\A[01]\z/;
    die "Invalid cubin_fnv\n" unless $config->{cubin_fnv} =~ /\A[0-9a-f]{16}\z/;
    my @attn_dies = _list($config->{attn_dies}, 'attn_dies');
    my %attn;
    for my $die (@attn_dies) {
        die "Invalid attn_dies layout\n" if $die > 5 || $attn{$die}++;
    }
    my @split = _list($config->{split}, 'split');
    die "split must contain six entries\n" unless @split == 6;
    my @map = _list($config->{attn_map}, 'attn_map');
    die "attn_map must contain twelve entries\n" unless @map == 12;
    die "attn_map references an unlisted die\n" for grep {!exists $attn{$_}} @map;
    my ($trace, $basis);
    if (exists $config->{trace_tokens}) {
        $trace = _uint($config->{trace_tokens}, 'trace_tokens');
        die "trace_tokens exceeds the supported 256 positions\n" if $trace > 256;
        $basis = "baseline effective trace_tokens=$trace";
    } elsif ($config->{rev} eq 'edc66d9' && $baseline_sha eq $LEGACY_SHA) {
        $trace = '0';
        # edc66d9e28819001b50112c63b58831f21becc28 has no FLASH_TRACE
        # implementation anywhere in kern; this exception is artifact-specific.
        $basis = "legacy engine edc66d9 with captured SHA-256 $LEGACY_SHA ignores FLASH_TRACE; restore value 0";
    } else {
        die "Baseline lacks effective trace_tokens; restore plan is unknown\n";
    }
    my %restore = (FLASH_ENGINE_MAXT=>"$config->{maxt}", FLASH_MAXIMG=>"$config->{img_rows}",
        FLASH_CUBIN=>"$config->{cubin}", FLASH_TRACE=>$trace,
        FLASH_TWO_POSITION=>"$config->{two_position}");
    my %candidate = (FLASH_ENGINE_MAXT=>'512', FLASH_MAXIMG=>'1',
        FLASH_CUBIN=>"$candidate_cubin", FLASH_TRACE=>'4');
    return {baseline=>{%$config}, restore=>\%restore,
        candidates=>[{%candidate, FLASH_TWO_POSITION=>'0'}, {%candidate, FLASH_TWO_POSITION=>'1'}],
        trace_restore_basis=>$basis};
}
# Match engine.c file_fnv without floating-point conversion of a 64-bit hash.
# Multiplication uses two 32-bit limbs; all intermediate values fit in 42 bits.
sub file_fnv {
    open my $f, '<:raw', $_[0] or die "Read cubin fingerprint $_[0]: $!";
    my ($hi, $lo) = (0xcbf29ce4, 0x84222325);
    while (1) {
        my $n = read($f, my $buffer, 65536);
        die "Read cubin fingerprint: $!" unless defined($n);
        last unless $n;
        for my $byte (unpack('C*', $buffer)) {
            $lo ^= $byte;
            my $product = $lo * 0x1b3;
            $hi = ($hi * 0x1b3 + $lo * 0x100 + int($product / 4294967296)) % 4294967296;
            $lo = $product % 4294967296;
        }
    }
    close $f or die "Close cubin fingerprint: $!";
    return sprintf('%08x%08x', $hi, $lo);
}
1;
