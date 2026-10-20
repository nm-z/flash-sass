#!/usr/bin/env perl
# Parser/planning regression fixtures only: never open CUDA or a GPU lock.
use strict;
use warnings;
use FindBin;
use Test::More;
use JSON::PP;

require "$FindBin::Bin/check_qsa_engine_requests.pl";
my $json = JSON::PP->new->canonical;

sub raw_scope {
    my ($dies, $tokens, $layers) = @_;
    return {
        revision => 'abcdef0', layers => $layers, nblk => 512,
        tokens => $tokens, snapshots => '1,2,3,5',
        visible => join(',', reverse 0 .. $dies - 1),
        manifests => [map { "/unused/die-$_.manifest" } 0 .. $dies - 1],
        holder_pids => [map { 1000 + $_ } 0 .. $dies - 1],
        split_shares => join(',', map { $_ + 1 } 0 .. $dies - 1),
        attn_map => join(',', map { ($_ + 1) % $dies } 0 .. 11),
    };
}

sub fixture {
    my ($o, $mode, $kv) = @_;
    my $plan = request_plan($o->{snapshot_list});
    my $paired = $mode eq 'scalar' ? 0 : 1;
    my $chunk = $paired ? 3 : 0;
    my $text = "config rev=$o->{revision} kv_host=$kv ple_disk=1 maxt=$o->{maxt} layers=$o->{layers}/48 nblk=$o->{nblk} dies=$o->{dies} split=$o->{split_shares} attn_map=$o->{attn_map} attn_dies=$o->{attn_dies} two_position=$paired prefill_chunk=$chunk\n";
    for my $r (@$plan) {
        my $base = $r->{prompt} eq 'A' ? 100 : $r->{prompt} eq 'B' ? 200 : 300;
        my @ids = map { $_ < $r->{resume} ? 0 : ($base + $_) % 248320 }
            0 .. $o->{tokens} - 1;
        @ids = map { ($base + $_) % 248320 }
            $o->{tokens} - 1 .. $o->{tokens} if $r->{n_new};
        my $positions = $o->{tokens} + ($r->{n_new} // 0);
        my $pairs = expected_pairs($r, $mode, $o->{tokens});
        my $chunks = $r->{n_new} && $paired ? int(($o->{tokens} + 1) / 3) : 0;
        $text .= 'tokens: ' . join(' ', @ids) . "\n";
        $text .= 'time: 0 ms for ' . ($positions - $r->{resume}) . " positions run ($r->{resume} from the snapshot; $pairs paired positions; $chunks prefill chunks)\n";
        $text .= "statehash positions=$positions paired_positions=$pairs prefill_chunks=$chunks\n";
        for my $die (0 .. $o->{dies} - 1) {
            for my $layer (owned_layers($o, $die)) {
                $text .= sprintf("statehash die=%d layer=%d state=%016x\n",
                    $die, $layer, $base + 1000 * $die + $layer);
            }
            $text .= sprintf("statehash die=%d ple=%016x logits=%016x\n",
                $die, $base + 1000 * $die + 60, $base + 1000 * $die + 70);
        }
        $text .= "statehash done\n";
    }
    return $text;
}

my ($positive, $negative) = (0, 0);
for my $shape ([1, 9, 8], [2, 6, 8], [2, 7, 8], [2, 8, 8],
               [2, 10, 8], [6, 9, 48], [6, 2052, 48]) {
    my $o = raw_scope(@$shape);
    $o->{snapshots} .= ',2048,2051' if $o->{tokens} == 2052;
    request_scope($o);
    my $references = {};
    my $plan = request_plan($o->{snapshot_list});
    for my $kv (0, 1) {
        for my $mode ('scalar', 'chunk3') {
            my $capture = fixture($o, $mode, $kv);
            my $r = eval { parse_capture($capture, $o, $mode, $kv, $plan, $references) };
            ok($r && @{$r->{requests}} == @$plan,
                "synthetic $o->{dies}-die/$o->{layers}-layer/$o->{tokens}-token kv$kv $mode capture")
                or diag $@;
            $positive++;
        }
    }
    is(scalar(keys %{$references->{A}{fingerprint}{ple}}), $o->{dies},
        'reference contains every PLE/logits shard');
}

my $o = raw_scope(6, 9, 48);
request_scope($o);
my $plan = request_plan($o->{snapshot_list});
my $text = fixture($o, 'chunk3', 0);
my $references = {};
parse_capture($text, $o, 'chunk3', 0, $plan, $references);
my @mutations = (
    ['die-5 recurrent state', sub { $_[0] =~ s/(die=5 layer=0 state=)[0-9a-f]+/${1}ffffffffffffffff/ }],
    ['die-5 attention state', sub { $_[0] =~ s/(die=5 layer=19 state=)[0-9a-f]+/${1}ffffffffffffffff/ }],
    ['die-5 PLE', sub { $_[0] =~ s/(die=5 ple=)[0-9a-f]+/${1}ffffffffffffffff/ }],
    ['die-5 logits', sub { $_[0] =~ s/(die=5 ple=[0-9a-f]+ logits=)[0-9a-f]+/${1}ffffffffffffffff/ }],
    ['missing remote layer', sub { $_[0] =~ s/^statehash die=5 layer=0 .*\n//m }],
    ['missing remote attention', sub { $_[0] =~ s/^statehash die=5 layer=19 .*\n//m }],
    ['missing remote PLE/logits', sub { $_[0] =~ s/^statehash die=5 ple=.*\n//m }],
    ['duplicate remote layer', sub { $_[0] =~ s/^(statehash die=5 layer=0 .*\n)/$1$1/m }],
    ['duplicate remote PLE/logits', sub { $_[0] =~ s/^(statehash die=5 ple=.*\n)/$1$1/m }],
    ['unexpected die', sub { $_[0] =~ s/die=5 layer=0 /die=6 layer=0 / }],
    ['attention on wrong owner', sub { $_[0] =~ s/die=1 layer=3 /die=0 layer=3 / }],
    ['wrong revision', sub { $_[0] =~ s/rev=abcdef0/rev=abcdef1/ }],
    ['duplicate config field', sub { $_[0] =~ s/rev=abcdef0/rev=abcdef1 rev=abcdef0/ }],
    ['wrong die count', sub { $_[0] =~ s/dies=6 /dies=1 / }],
    ['wrong split', sub { $_[0] =~ s/split=1,2,3,4,5,6/split=1,1,1,1,1,1/ }],
    ['wrong attention map', sub { $_[0] =~ s/attn_map=1,2/attn_map=0,2/ }],
    ['wrong active dies', sub { $_[0] =~ s/attn_dies=0,1/attn_dies=1/ }],
    ['wrong budget', sub { $_[0] =~ s/nblk=512/nblk=1/ }],
    ['wrong capacity', sub { $_[0] =~ s/maxt=128/maxt=256/ }],
    ['wrong paired count', sub { $_[0] =~ s/8 paired positions/6 paired positions/ }],
    ['out-of-vocabulary token', sub { $_[0] =~ s/tokens: 100 /tokens: 248320 / }],
    ['mismatched generation token', sub { $_[0] =~ s/tokens: 308 309/tokens: 308 310/ }],
    ['wrong chunk count', sub { $_[0] =~ s/6 paired positions; 3 prefill chunks/6 paired positions; 4 prefill chunks/ }],
    ['incomplete final response', sub { $_[0] =~ s/statehash done\n\z// }],
    ['engine error', sub { $_[0] = "error: failed generation\n$_[0]"; 1 }],
);
for my $mutation (@mutations) {
    my $bad = $text;
    die "mutation did not match: $mutation->[0]\n" unless $mutation->[1]->($bad);
    my $passed = eval {
        parse_capture($bad, $o, 'chunk3', 0, $plan,
            $json->decode($json->encode($references))); 1;
    };
    ok(!$passed, "reject $mutation->[0]");
    $negative++;
}

# Independent known answers for chunk tails and snapshot boundaries: fixture
# production uses expected_pairs, so accepting fixtures alone cannot test it.
for my $case ([6, 4], [7, 4], [8, 4], [9, 6], [10, 6], [2052, 1368]) {
    is(expected_pairs({n_new => 1}, 'chunk3', $case->[0]), $case->[1],
        "paired generation-prefix positions at length $case->[0]");
}
is(expected_pairs({resume => 2051, snap => 0}, 'chunk3', 2052), 0,
   'resume at final prompt position leaves no paired prefix');
is(expected_pairs({resume => 0, snap => 2051}, 'chunk3', 2052), 2050,
   'snapshot immediately before final position splits paired prefix');
is(expected_pairs({resume => 0, snap => 0}, 'scalar', 2052), 0,
   'scalar-launch reference never claims paired work');

my @bad_scopes = (
    ['duplicate device', sub { $_[0]{visible} = '0,0' }],
    ['too many devices', sub { $_[0]{visible} = '0,1,2,3,4,5,6' }],
    ['missing manifest', sub { pop @{$_[0]{manifests}} }],
    ['duplicate manifest', sub { $_[0]{manifests}[1] = $_[0]{manifests}[0] }],
    ['missing holder', sub { pop @{$_[0]{holder_pids}} }],
    ['duplicate holder', sub { $_[0]{holder_pids}[1] = $_[0]{holder_pids}[0] }],
    ['bad holder PID', sub { $_[0]{holder_pids}[0] = -1 }],
    ['missing shares', sub { delete $_[0]{split_shares} }],
    ['missing map', sub { delete $_[0]{attn_map} }],
    ['zero share', sub { $_[0]{split_shares} = '1,0,1,1,1,1' }],
    ['malformed share', sub { $_[0]{split_shares} = '1,1x,1,1,1,1' }],
    ['short map', sub { $_[0]{attn_map} = '0,1' }],
    ['out-of-range owner', sub { $_[0]{attn_map} = join(',', (6) x 12) }],
    ['negative owner', sub { $_[0]{attn_map} = join(',', (-1) x 12) }],
    ['unreachable default snapshot', sub { $_[0]{tokens} = 5 }],
    ['capacity overflow', sub { $_[0]{tokens} = 262143 }],
    ['duplicate snapshot', sub { $_[0]{snapshots} = '1,1' }],
    ['snapshot past prompt', sub { $_[0]{snapshots} = '9' }],
);
for my $case (@bad_scopes) {
    my $bad = raw_scope(6, 9, 48);
    $case->[1]->($bad);
    local $ENV{SPLIT_SHARES}; delete $ENV{SPLIT_SHARES};
    local $ENV{ATTN_MAP}; delete $ENV{ATTN_MAP};
    my $passed = eval { request_scope($bad); 1 };
    ok(!$passed, "reject scope: $case->[0]");
}

my $deep = raw_scope(6, 2052, 48);
$deep->{snapshots} = '1,2,3,5,2048,2051';
request_scope($deep);
is($deep->{maxt}, 2176, '2052 positions plus parser/generation headroom');
my $deep_plan = request_plan($deep->{snapshot_list});
my $cmd = commands_for($deep_plan, $deep->{tokens});
my @gen = grep { /^gen / } @$cmd;
is(scalar @gen, 22, 'deep snapshot positions add explicit save/resume/reset requests');
is(scalar(split / /, $gen[0]), 2054, 'teacher command has exactly 2052 IDs');
like(join("\n", @$cmd), qr/^snap 2051$/m, 'production-boundary snapshot is not dropped');
my $last = raw_scope(1, 262142, 8);
request_scope($last);
is($last->{maxt}, 262144, 'maximum prompt respects capacity and generation headroom');
my @long_gen = grep { /^gen / } @{commands_for(request_plan([1]), $last->{tokens})};
my @long_ids = split / /, $long_gen[0];
splice @long_ids, 0, 2;
ok(@long_ids == 262142 && !(grep { $_ >= 248320 } @long_ids),
    'long synthetic prompts wrap at the vocabulary boundary');

diag "$positive complete captures and $negative corrupt captures tested; CPU fixtures only, no GPU validation";
done_testing();
