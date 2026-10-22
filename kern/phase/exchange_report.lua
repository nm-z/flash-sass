-- Input TSV: arm position die epoch producer_phase consumer_phase producer_op
-- consumer_op ready published start_min start_max acquire arrival0..arrival5.
-- Clocks are never subtracted across GPUs without causal offset bounds.
local rows,groups={},{}
for line in io.lines(assert(arg[1]))do
 local r={};for v in line:gmatch('[^\t]+')do r[#r+1]=tonumber(v)or v end
 assert(#r==19,'expected 19 TSV columns')
 rows[#rows+1]=r
 local key=r[1]..':'..r[2];groups[key]=groups[key]or {};local g=groups[key]
 g[r[4]]=g[r[4]]or {};assert(not g[r[4]][r[3]],'duplicate exchange');g[r[4]][r[3]]=r
end
assert(#rows>0,'empty trace')
local offsets={}
for key,g in pairs(groups)do
 local lo,hi={[0]=0},{[0]=0}
 local bound={};for i=0,5 do bound[i]={};for j=0,5 do bound[i][j]=i==j and 0 or math.huge end end
 for _,e in pairs(g)do
  for d=0,5 do assert(e[d],'missing die at epoch')end
  for i=0,5 do for j=0,5 do if i~=j then
   bound[i][j]=math.min(bound[i][j],e[j][14+i]-e[i][9])
  end end end
 end
 for k=0,5 do for i=0,5 do for j=0,5 do bound[i][j]=math.min(bound[i][j],bound[i][k]+bound[k][j])end end end
 for d=0,5 do lo[d]=-bound[0][d];hi[d]=bound[d][0] end
 for d=0,5 do assert(lo[d]<=hi[d],'inconsistent causal clock bounds: '..key..' die '..d)end
 offsets[key]={lo,hi,bound}
end
local dir=assert(arg[2]);local csv=assert(io.open(dir..'/exchanges.csv','w'))
csv:write('arm,position,die,epoch,producer_phase,consumer_phase,producer_op,consumer_op,finish_to_acquire_us,publication_us,pre_acquire_us,acquire_span_us,producer_skew_lower_us,producer_skew_upper_us,communication_poll_lower_us,communication_poll_upper_us,clock_bound_width_us\n')
local sums,operations={},{}
for _,r in ipairs(rows)do
 local key=r[1]..':'..r[2];local e=groups[key][r[4]];local off=offsets[key];local lo,hi=off[1],off[2];local d=r[3]
 local latest_lo,latest_hi=-math.huge,-math.huge;local bound=off[3]
 for i=0,5 do latest_lo=math.max(latest_lo,e[i][9]-bound[d][i]);latest_hi=math.max(latest_hi,e[i][9]+bound[i][d])end
 local skew_lo=math.max(0,latest_lo-r[9]);local skew_hi=math.max(0,latest_hi-r[9])
 local comm_lo=math.max(0,r[13]-latest_hi);local comm_hi=math.max(0,r[13]-latest_lo)
 local vals={r[13]-r[9],r[10]-r[9],r[12]-r[9],r[13]-r[11],skew_lo,skew_hi,comm_lo,comm_hi,hi[d]-lo[d]}
 for i=1,8 do csv:write(tostring(r[i]),',')end
 for i,v in ipairs(vals)do csv:write(string.format('%.3f%s',v/1000,i==#vals and '\n'or ','))end
 local sk=r[1]..':'..d;sums[sk]=sums[sk]or {arm=r[1],die=d,n=0,tokens={},v={0,0,0,0,0,0,0,0,0}}
 local s=sums[sk];s.n=s.n+1;s.tokens[r[2]]=true;for i,v in ipairs(vals)do s.v[i]=s.v[i]+v end
 local ok=r[1]..':'..r[7]..':'..r[8]
 operations[ok]=operations[ok]or {arm=r[1],producer=r[7],consumer=r[8],n=0,samples={},v={0,0,0,0,0,0,0,0,0}}
 local o=operations[ok];o.n=o.n+1;o.samples[r[2]..':'..d]=true;for i,v in ipairs(vals)do o.v[i]=o.v[i]+v end
end
csv:close()
local md=assert(io.open(dir..'/exchange-summary.md','w'))
md:write('# Per-exchange timing\n\nAll values below are microseconds per token, summed over exchanges on each die. `exchanges.csv` contains every measured exchange. Finish is the latest of the die’s 16 CTA body-completion observations. Acquire is the latest of its 16 completion observations. Publication and polling instrumentation add overhead; these are instrumented timings, not uninstrumented kernel timings.\n\nClock offsets are bounded independently for each token using the causal fact that a receiving GPU cannot observe a producer flag before that producer finishes. Relative drift within one token is assumed negligible. Producer skew and communication/polling are intervals, not a falsely precise link-latency measurement. The latter includes publication, transport, cache invalidation, polling and CTA rendezvous.\n\n| Arm | Die | Exchanges/token | Finish → acquire | Producer skew bounds | Communication/poll bounds |\n|---|---:|---:|---:|---:|---:|\n')
local keys={};for k in pairs(sums)do keys[#keys+1]=k end;table.sort(keys)
for _,k in ipairs(keys)do local s=sums[k];local nt=0;for _ in pairs(s.tokens)do nt=nt+1 end;local q=nt*1000
 md:write(string.format('| %s | %d | %.0f | %.3f | %.3f–%.3f | %.3f–%.3f |\n',s.arm,s.die,s.n/nt,s.v[1]/q,s.v[5]/q,s.v[6]/q,s.v[7]/q,s.v[8]/q))
end
md:close()
local oc=assert(io.open(dir..'/exchange-operations.csv','w'))
oc:write('arm,producer,consumer,observed_exchanges,finish_to_acquire_us_per_token_per_die,skew_lower_us,skew_upper_us,communication_poll_lower_us,communication_poll_upper_us\n')
local opkeys={};for k in pairs(operations)do opkeys[#opkeys+1]=k end;table.sort(opkeys)
for _,k in ipairs(opkeys)do local o=operations[k];local samples=0;for _ in pairs(o.samples)do samples=samples+1 end;local q=samples*1000
 oc:write(string.format('%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n',o.arm,o.producer,o.consumer,o.n,o.v[1]/q,o.v[5]/q,o.v[6]/q,o.v[7]/q,o.v[8]/q))
end
oc:close();print(string.format('exchange_report records=%d clock_groups=%d',#rows,(function()local n=0;for _ in pairs(groups)do n=n+1 end;return n end)()))
