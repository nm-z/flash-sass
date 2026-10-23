#!/usr/bin/env perl
# Recalculate the historical 7f37f6b tables without upgrading their provenance.
use strict;
use warnings;
use JSON::PP;
use File::Spec;
my $directory = shift @ARGV // die "usage: $0 HISTORICAL_DIRECTORY\n";
die "unexpected arguments\n" if @ARGV;
my $revision = '7f37f6bdc69a0286869ead4d0e8231fa601b23c1';
sub read_text {
    my ($path) = @_;
    open my $in, '<', $path or die "$path: $!\n";
    local $/;
    my $text = <$in>;
    close $in or die "$path: $!\n";
    return $text;
}
sub median {
    my @v = sort { $a <=> $b } @{$_[0]};
    return @v % 2 ? $v[@v / 2] : ($v[@v / 2 - 1] + $v[@v / 2]) / 2;
}
sub p95 {
    my @v = sort { $a <=> $b } @{$_[0]};
    my $rank = int(0.95 * @v);
    $rank++ if $rank < 0.95 * @v;
    return $v[$rank - 1];
}
my (%files, %cross_storage);
for my $storage (qw(local nfs)) {
    my $stem = File::Spec->catfile($directory, "flash-ple-7f37f6b-$storage");
    my $log = read_text("$stem.log");
    die "$storage: missing tested revision or final success record\n"
        unless $log =~ /\brevision=\Q$revision\E\b/ && $log =~ /^pass=1\b/m;
    die "$storage: reported failure\n" if $log =~ /^pass=0\b/m;
    my ($trials) = $log =~ /\btrials=([0-9]+)\b/;
    my ($parallel) = $log =~ /\breaders=([0-9]+)\b/;
    my ($seed) = $log =~ /\bseed=([0-9]+)\b/;
    die "$storage: invalid trial configuration\n"
        unless $trials && $parallel && $parallel >= 2 && $parallel <= 16 && defined $seed;
    my @lines = split /\n/, read_text("$stem.csv");
    my $header = shift @lines;
    die "$storage: unexpected legacy CSV schema\n"
        unless $header eq 'phase,trial,readers,resident_pages,pages,hint_ns,read_ns,total_ns,hash,bytes,errors,hint_error,offsets';
    my (%groups, %pairs, %seen);
    for my $line (@lines) {
        $line =~ s/\r\z//;
        my @fields = split /,/, $line, -1;
        die "$storage: malformed row\n" unless @fields == 13;
        my %r; @r{split /,/, $header} = @fields;
        die "$storage: invalid phase\n" unless $r{phase} =~ /\A(?:warm|unprimed)\z/;
        for my $key (qw(trial readers resident_pages pages hint_ns read_ns total_ns bytes errors hint_error)) {
            die "$storage: non-integer $key\n" unless $r{$key} =~ /\A[0-9]+\z/;
        }
        die "$storage: unsuccessful or inconsistent batch\n"
            unless $r{bytes} == 1440 && !$r{errors} && !$r{hint_error}
                && $r{pages} >= 16 && $r{resident_pages} <= $r{pages}
                && $r{hint_ns} + $r{read_ns} == $r{total_ns} && $r{total_ns} > 0
                && ($r{readers} == 1 || $r{readers} == $parallel)
                && $r{hash} =~ /\A[0-9a-f]{16}\z/
                && $r{offsets} =~ /\A"[0-9]+(?:;[0-9]+){15}"\z/;
        my $limit = $r{phase} eq 'warm' ? $trials * 8 : $trials;
        die "$storage: trial out of range\n" unless $r{trial} < $limit;
        my $key = join ':', @r{qw(phase trial readers)};
        die "$storage: duplicate row\n" if $seen{$key}++;
        my $class = $r{resident_pages} == 0 ? 'client_absent'
            : $r{resident_pages} == $r{pages} ? 'client_resident' : 'client_mixed';
        push @{$groups{"$r{phase}:$class:$r{readers}"}}, 0 + $r{total_ns};
        $pairs{$r{phase}}{$r{trial}}{$r{readers}} = \%r;
        $cross_storage{$storage}{$key} = {offsets => $r{offsets}, hash => $r{hash}};
    }
    die "$storage: incomplete file\n" unless @lines == $trials * 18;
    my @matched;
    for my $phase (qw(unprimed warm)) {
        my $limit = $phase eq 'warm' ? $trials * 8 : $trials;
        for my $trial (0 .. $limit - 1) {
            my ($a, $b) = @{$pairs{$phase}{$trial}}{1, $parallel};
            die "$storage: missing arm\n" unless $a && $b;
            if ($phase eq 'warm') {
                die "$storage: warm inputs or outputs differ\n"
                    unless $a->{offsets} eq $b->{offsets} && $a->{hash} eq $b->{hash};
                push @matched, $a->{total_ns} / $b->{total_ns}
                    if $a->{resident_pages} == $a->{pages} && $b->{resident_pages} == $b->{pages};
            }
        }
    }
    my @summary;
    for my $key (sort keys %groups) {
        my ($phase, $class, $readers) = split /:/, $key;
        push @summary, {phase => $phase, cache_class => $class, readers => 0 + $readers,
            batches => scalar @{$groups{$key}}, median_ns => median($groups{$key}),
            p95_ns => p95($groups{$key})};
    }
    $files{$storage} = {batches => scalar @lines, seed => 0 + $seed, groups => \@summary,
        fully_resident_warm_pairs => scalar @matched,
        median_paired_serial_over_parallel => @matched ? median(\@matched) : undef};
}
die "local/NFS seeds differ\n" unless $files{local}{seed} == $files{nfs}{seed};
for my $key (keys %{$cross_storage{local}}) {
    my ($a, $b) = ($cross_storage{local}{$key}, $cross_storage{nfs}{$key});
    die "local/NFS sampled inputs differ at $key\n"
        unless $b && $a->{offsets} eq $b->{offsets} && $a->{hash} eq $b->{hash};
}
print JSON::PP->new->canonical->pretty->encode({
    status => 'historical_replayed', source_sha => $revision, data => \%files,
    limitations => [
        'Original final pass records are present; this replay does not recreate missing command-exit files.',
        'Unprimed arms are randomized disjoint inputs, not same-offset cold pairs.',
        'Warm comparisons cover measured client-cache residency only.',
        'No new GPU, disk benchmark, model validation, or issue acceptance is claimed.'
    ]
});
