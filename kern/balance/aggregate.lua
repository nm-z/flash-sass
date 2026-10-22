-- Diagnostic means across a complete contiguous trace window. This does not
-- replace the per-observation gate or discard any failing observation.
local path=assert(arg[1],'usage: aggregate.lua TABLE.tsv PHASES POSITIONS')
local phases=assert(tonumber(arg[2]));local positions=assert(tonumber(arg[3]))
assert(phases>0 and phases<500 and positions>0 and positions<=256)
assert(phases%1==0 and positions%1==0,'dimensions must be integers')
local f=assert(io.open(path));local header=assert(f:read('l'))
assert(header:find('cycle_coverage_valid',1,true),'requires cycle-enabled table')
local rows,seen,records,request={}, {},0,nil
local function fields(line)
 local t={};for value in (line..'\t'):gmatch('(.-)\t')do t[#t+1]=value end;return t
end
local function number(s)
 local n=assert(tonumber(s),'invalid numeric field');assert(n==n and n>=0 and n<1e12);return n
end
for line in f:lines()do
 local a=fields(line);assert(#a==30,'unexpected table width')
 local req,pos,phase=number(a[1]),number(a[2]),number(a[3])
 assert(phase<phases and phase%1==0 and pos%1==0)
 if request then assert(request==req,'do not pool different requests')else request=req end
 assert(a[14]=='1' and a[29]=='1','incomplete timing coverage')
 seen[pos]=seen[pos]or {};assert(not seen[pos][phase],'duplicate observation');seen[pos][phase]=true
 local p=rows[phase]
 if not p then p={layer=a[4],op=a[5],ns={},cycles={},n=0,failed_ns=0,failed_cycles=0,worst=0};rows[phase]=p end
 assert(p.layer==a[4] and p.op==a[5],'operation changed inside window')
 p.n=p.n+1;records=records+1
 p.failed_ns=p.failed_ns+(a[15]=='0'and 1 or 0)
 p.failed_cycles=p.failed_cycles+(a[30]=='0'and 1 or 0)
 p.worst=math.max(p.worst,number(a[28]))
 for d=1,6 do
  p.ns[d]=(p.ns[d]or 0)+math.floor(number(a[5+d])*1000+0.5)
  p.cycles[d]=(p.cycles[d]or 0)+number(a[21+d])
 end
end
f:close()
local np,lo,hi=0,math.huge,0
for pos in pairs(seen)do np=np+1;lo=math.min(lo,pos);hi=math.max(hi,pos)end
assert(np==positions and hi-lo+1==positions and records==phases*positions,'incomplete window')
local function deviation(values)
 local sum=0;for _,v in ipairs(values)do sum=sum+v end;assert(sum>0)
 local worst,pass=0,true
 for _,v in ipairs(values)do
  worst=math.max(worst,math.abs(6*v/sum-1)*100)
  if 60*v<9*sum or 60*v>11*sum then pass=false end
 end
 return worst,pass
end
local names={'phase','layer','op','positions'}
for d=0,5 do names[#names+1]='die'..d..'_mean_us'end
for d=0,5 do names[#names+1]='die'..d..'_mean_cycles'end
for _,s in ipairs({'mean_ns_max_deviation_pct','mean_cycles_max_deviation_pct','failing_ns_observations','failing_cycle_observations','worst_cycle_observation_pct','mean_ns_within_10pct','mean_cycles_within_10pct'})do names[#names+1]=s end
print(table.concat(names,'\t'))
local bad_ns,bad_cycles=0,0
for phase=0,phases-1 do
 local p=assert(rows[phase]);assert(p.n==positions)
 local dn,pn=deviation(p.ns);local dc,pc=deviation(p.cycles)
 bad_ns=bad_ns+(pn and 0 or 1);bad_cycles=bad_cycles+(pc and 0 or 1)
 local out={phase,p.layer,p.op,p.n}
 for d=1,6 do out[#out+1]=string.format('%.6f',p.ns[d]/positions/1000)end
 for d=1,6 do out[#out+1]=string.format('%.6f',p.cycles[d]/positions)end
 out[#out+1]=string.format('%.6f',dn);out[#out+1]=string.format('%.6f',dc)
 out[#out+1]=p.failed_ns;out[#out+1]=p.failed_cycles;out[#out+1]=string.format('%.6f',p.worst)
 out[#out+1]=pn and 1 or 0;out[#out+1]=pc and 1 or 0
 print(table.concat(out,'\t'))
end
io.stderr:write(string.format('balance_window phases=%d positions=%d records=%d mean_ns_failing=%d mean_cycles_failing=%d diagnostic_only=1 original_per_observation_gate_unchanged=1\n',phases,positions,records,bad_ns,bad_cycles))
