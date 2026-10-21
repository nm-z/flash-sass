-- Read flattened `gpuq logs` output; no kernel execution or timing allocation.
-- usage: lua kern/phase/profile_report.lua PROFILE_TEXT OUTPUT_DIRECTORY
local input=assert(arg[1]);local dir=assert(arg[2])
local arms={baseline={},['kernel-candidate']={}};local names={};local arm,die
local expected={baseline={},['kernel-candidate']={}}
for line in io.lines(input)do
 local a,d=line:match('^profile arm=([%w%-]+) die=(%d+)')
 if a then arm=a;die=tonumber(d);assert(arms[arm]);arms[arm][die]={} end
 local phases,ms=line:match('^prof: (%d+) phases, ([%d%.]+) ms')
 if phases then expected[arm][die]={n=tonumber(phases)-1,us=tonumber(ms)*1000}end
 local op,n,us=line:match('^%s+([%w_]+)%s+(%d+) phases%s+([%d%.]+) us')
 if op then assert(arm);assert(not arms[arm][die][op]);arms[arm][die][op]={n=tonumber(n),us=tonumber(us)};names[op]=true end
end
for name,a in pairs(arms)do for d=0,5 do
 assert(a[d],'missing die');local n,us=0,0
 for _,v in pairs(a[d])do n=n+v.n;us=us+v.us end
 local e=assert(expected[name][d],'missing profile header')
 assert(n==e.n,'profile phase count does not reconcile')
 assert(math.abs(us-e.us)<10,'profile time does not reconcile within print precision')
end end
local function value(a,d,op)return arms[a][d][op] or {n=0,us=0}end
local function stats(a,op)
 local sum,lo,hi=0,math.huge,0
 for d=0,5 do local v=value(a,d,op);sum=sum+v.us;lo=math.min(lo,v.n);hi=math.max(hi,v.n)end
 return sum/6,lo==hi and tostring(lo)or string.format('%d–%d',lo,hi)
end
local function total(a,ops)
 local sum=0;for _,op in ipairs(ops)do local v=stats(a,op);sum=sum+v end;return sum
end
local ordered={};for op in pairs(names)do ordered[#ordered+1]=op end
table.sort(ordered,function(a,b)local x=stats('kernel-candidate',a);local y=stats('kernel-candidate',b);if x==y then return a<b end;return x>y end)
local csv=assert(io.open(dir..'/operations.csv','w'))
csv:write('die,operation,baseline_phases,baseline_us_per_token,candidate_phases,candidate_us_per_token\n')
for d=0,5 do for _,op in ipairs(ordered)do local b=value('baseline',d,op);local c=value('kernel-candidate',d,op);csv:write(string.format('%d,%s,%d,%.1f,%d,%.1f\n',d,op,b.n,b.us,c.n,c.us))end end
csv:close()
local md=assert(io.open(dir..'/operations.md','w'))
md:write('# Complete saved PROF_ALL comparison\n\nSource: canonical archy GPUQ job `03bff6fb-db1f-4844-9d91-0e2e381ea17f`, fetched with `gpuq logs ID 10000`. No new GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.\n\nThese are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.\n\n| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |\n|---|---:|---:|---:|---:|---:|\n')
for _,op in ipairs(ordered)do local b,bn=stats('baseline',op);local c,cn=stats('kernel-candidate',op);md:write(string.format('| `%s` | %s | %.1f | %s | %.1f | %+.1f |\n',op,bn,b,cn,c,c-b))end
local bt,ct=0,0;for _,op in ipairs(ordered)do bt=bt+stats('baseline',op);ct=ct+stats('kernel-candidate',op)end
md:write(string.format('| **All recorded intervals** | 1,177–1,229 | **%.1f** | 498 | **%.1f** | **%+.1f** |\n',bt,ct,ct-bt))
md:write('\nThe uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.\n\n## Comparisons that can be formed without allocating shared opcode pools\n\n| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |\n|---|---|---|---:|---:|---:|\n')
local families={
 {'All residual sum + GR down reads',{'hcwsum','bf16hcrx'},{'grwrite_read','hcwsum','bf16hcrx'}},
 {'Activation producers, quantization, attention partition/merge',{'bf16hcup','conv1d','dnstep','dnnormx','plerows','actq256','attnpartn','attnmerge'},{'hcupq_x','gdn_full','ple_rowsq','qsa_part_merge'}},
 {'QSA score + selection',{'idxscore','idxsel'},{'qsa_score_select'}},
 {'All PLE normalizations + gate',{'rms2560','plegate'},{'rms2560','plegate_norm'}},
 {'Q6 slice bodies (publication excluded on baseline)',{'q6kk1536','q6kk1536acc'},{'q6kk1536'}},
}
for _,r in ipairs(families)do local b=total('baseline',r[2]);local c=total('kernel-candidate',r[3]);md:write(string.format('| %s | `%s` | `%s` | %.1f | %.1f | %+.1f |\n',r[1],table.concat(r[2],'` + `'),table.concat(r[3],'` + `'),b,c,c-b))end
md:write('\nThe first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.\n\n')
for d=0,5 do
 md:write(string.format('## Die %d\n\n| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |\n|---|---:|---:|---:|---:|\n',d))
 for _,op in ipairs(ordered)do local b=value('baseline',d,op);local c=value('kernel-candidate',d,op);md:write(string.format('| `%s` | %d | %.1f | %d | %.1f |\n',op,b.n,b.us,c.n,c.us))end
 md:write('\n')
end
md:close()
