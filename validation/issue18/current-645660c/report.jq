def stats:
  sort | length as $n |
  if $n == 0 then error("empty timing group") else
  {samples:$n,min:.[0],median:(if $n%2 == 1 then .[($n/2|floor)] else (.[($n/2)-1]+.[($n/2)])/2 end),max:.[-1]} end;
def acceptance($s;$expected):
  if $s.expected != $expected or $s.completed != $expected or $s.passed != true or
     ($s.cases|length) != $expected or (all($s.cases[];.passed == true)|not)
  then error("incomplete or failed " + $s.suite) else
  {suite:$s.suite,expected:$s.expected,completed:$s.completed,passed:$s.passed} end;
($padding[0]) as $p | ($direct[0]) as $d | ($controls[0]) as $c | ($timing[0]) as $t |
if ($engine|length) != 80 or $engine_status[0].workload_exit != 0 or $engine_status[0].restore_verified != true or
   $cli[0].status != "PASS" or ($cli[0].cases|length) != 46 or (all($cli[0].cases[];.passed == true)|not)
then error("engine acceptance or restoration incomplete") else
{
  implementation:"645660cd27a0ab7c2f28faeb9bddfcf2f33886cd",
  runner_revision:"cee182df737fd7501060288402f14202b9a186fe",
  completed_utc:($completed|rtrimstr("\n")),
  control:"f88dfc56c3045023b32ca7eec62b0dd7bfd3f9a2",
  control_change:"Only kern/attnpart.sass and kern/attnpartn.sass replaced with unmasked aa0b898607f0c070bda100e10dfb6bb6391e55a4 versions; same candidate C drivers.",
  acceptance:[acceptance($p;1536),acceptance($d;10),acceptance($c;26),acceptance($t;40)],
  cli:{cases:($cli[0].cases|length),passed:all($cli[0].cases[];.passed),guard:$cli[0].guard,binaries:$cli[0].binaries},
  padding_by_arm:($p.cases|group_by([.K,.V])|map({K:.[0].K,V:.[0].V,cases:length,passed:all(.[];.passed)})),
  padding_by_placement_and_kind:($p.cases|group_by([.placement,.kind])|map({placement:.[0].placement,kind:.[0].kind,cases:length,passed:all(.[];.passed)})),
  sequential_kv_positions:([$p.cases[]|select(.kind=="kv")|.n]|add),
  direct_positions:([$d.cases[].iterations]|add),
  controls_by_outcome:($c.cases|group_by(.expectation)|map({expectation:.[0].expectation,cases:length,passed:all(.[];.passed),exit_codes:(map(.exit_code)|unique)})),
  component_timings:($t.cases|group_by([.kind,.placement])|map(
    . as $g |
    [range(0;length;2) as $i | $g[$i:$i+2] |
      (map(select(.image=="candidate"))[0]) as $a |
      (map(select(.image=="parent"))[0]) as $b |
      if $a==null or $b==null or $a.microseconds <= 0 or $b.microseconds <= 0 then error("bad component pair") else
      {pair:($i/2+1),candidate_us:$a.microseconds,control_us:$b.microseconds,change_percent:(100*($a.microseconds/$b.microseconds-1))} end] as $pairs |
    {kind:.[0].kind,placement:.[0].placement,visible:.[0].n,budget:.[0].budget,iterations:.[0].iterations,
     candidate_us:([$pairs[].candidate_us]|stats),control_us:([$pairs[].control_us]|stats),
     paired_change_percent:([$pairs[].change_percent]|stats),pairs:$pairs})),
  engine:{
    requests:($engine|length),fixtures:4,comparisons:32,guard_checks:84,logits_per_comparison:248320,token_ids_per_comparison:3,
    restore_verified:$engine_status[0].restore_verified,
    timings:($engine|group_by([.placement,.stage])|map(
      (map(select(.variant=="candidate"))[0]) as $a |
      (map(select(.variant=="control"))[0]) as $b |
      if length!=2 or $a==null or $b==null or $a.run_ms<=0 or $b.run_ms<=0 then error("bad engine pair") else
      {placement:.[0].placement,stage:.[0].stage,category:(.[0].stage|sub("-r[0-3]$";"")),candidate_ms:$a.run_ms,control_ms:$b.run_ms,
       change_percent:(100*($a.run_ms/$b.run_ms-1)),candidate_build_ms:$a.build_ms,control_build_ms:$b.build_ms} end) as $pairs |
      {by_stage:($pairs|group_by([.placement,.category])|map({placement:.[0].placement,stage:.[0].category,
        candidate_ms:(map(.candidate_ms)|stats),control_ms:(map(.control_ms)|stats),paired_change_percent:(map(.change_percent)|stats)})),pairs:$pairs})
  },
  limitations:[
    "The compact padding fixture isolates attention and merge; the direct QSA fixture uses complete synthetic history; neither is an end-to-end maximum-context model run.",
    "Actual-model reuse fixtures use 96/64-token prompts and snapshots 16 through 19, all 48 layers and actual PLE. They compare three emitted token IDs and every final logit, not every intermediate generation-step logit buffer.",
    "Canaries detect allocation-boundary writes. Read extents are established by the documented load derivation; internal host-arena subarray boundaries are not individually guarded.",
    "Component timing uses five alternating image pairs of 200 iterations. Actual-model timing uses four matched requests per stage and placement and includes cold-page effects; it is not a long-context throughput benchmark."
  ]
} end
