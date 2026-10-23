-- List-schedule arithmetic and loads, retaining known completion-slot assignments
-- and waiting at consumers, destination/address reuse, or barrier exhaustion.
-- Stores, volatile loads, branches, predicates, fences and unknown operations
-- delimit regions: no unresolved read/write alias crosses a scheduling boundary.
-- Static issue clocks exclude unknown memory waits, so a cache hit cannot hide
-- a fixed-result hazard. A separate latency estimate only ranks ready work.
-- Usage: lua dag_memory.lua IN OUT REPORT.csv
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local regs={};for n=0,254 do regs['R'..n]='R'..n end
for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 local left,right=line:match('^%s*([%d,%-%s]+)%s*:%s*([^#]+)')
 if left then
  right=right:match('^%s*(.-)%s*$');local ns={}
  for part in left:gmatch('[^,%s]+')do
   local a,b=part:match('^(%d+)%-(%d+)$')
   if a then for n=tonumber(a),tonumber(b)do ns[#ns+1]=n end
   else ns[#ns+1]=assert(tonumber(part))end
  end
  local prefix,a,b=right:match('^([%w_]+)<(%d+)%-(%d+)>$')
  if prefix then
   assert(#ns==tonumber(b)-tonumber(a)+1)
   for n=tonumber(a),tonumber(b)do
    local at=n-tonumber(a)+1;regs[prefix..string.format('%0'..#a..'d',n)]='R'..ns[at]
    if at==1 and ns[at]%2==0 then regs[prefix]='R'..ns[at]end
   end
  else assert(#ns==1);regs[right]='R'..ns[1]end
 end
end
local alu={MOV=true,MOV32I=true,FADD=true,FMUL=true,FFMA=true,FMNMX=true,FSET=true,
 IADD=true,IADD32I=true,ISCADD=true,XMAD=true,LOP=true,LOP32I=true,LOP3=true,
 SHL=true,SHR=true,SHF=true,BFE=true,BFI=true,SEL=true,IMNMX=true,PRMT=true,
 ICMP=true,FCMP=true}
local shift={ISCADD=true,SHL=true,SHR=true,SHF=true,BFE=true,BFI=true,IMNMX=true}
local function regset(x)
 local set={};x=x:gsub('0[xX]%x+','')
 for name in x:gmatch('[%a_][%w_]*')do
  if regs[name]then set[regs[name]]=true
  elseif name:match('^P[0-6]$')then set[name]=true end
 end
 return set
end
local function parse(line)
 local ctl,ins=line:match('^(%S+)%s+([^;]+);')
 if not ctl then return end
 local wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):(%x)$')
 if not stall then return end
 local pred,body=ins:match('^(@!?P[0-6])%s+(.+)')
 body=body or ins
 local full,args=body:match('^(%S+)%s*(.*)');local kind=full:match('^[^.]+')
 local cycles=tonumber(stall,16);if cycles>11 and y~='Y'then cycles=6 end
 local mask=tonumber(wait,16)or 0
 if kind=='NOP' and not pred and rd=='-' and wr=='-' then return {nop=true,cycles=cycles,line=line,old_wait=mask}end
 if kind=='LDG' or kind=='LDS' then
  if full:find('.CV',1,true)then return end -- polling/volatile reads stay ordered
  if not tonumber(wr)then return end -- unknown completion contract stays anchored
  local dest,address=args:match('^([%w_]+),%s*%[([^%]]+)%]$')
  local base=address and address:match('^([%w_]+)')
  if not dest or not regs[dest] or not base or (base~='RZ' and not regs[base])then return end
  local tail=address:sub(#base+1)
  if tail~='' and not tail:match('^%+0x%x+$') and not tail:match('^%-0x%x+$')then return end
  -- Descriptor/control banks are populated by the persistent dispatcher and
  -- composite-entry paths. Their incoming completion contract is outside the
  -- local operation DAG; preserve that prologue until ownership is modeled.
  local shared_offset=tonumber(tail:match('^%+0x(%x+)$')or '',16)
  if kind=='LDS' and base=='RZ' and shared_offset and shared_offset>=0xbf00 then return end
  local width=tonumber(full:match('%.(%d+)$'))or 32
  if width~=32 and width~=64 and width~=128 then return end
  local writes,reads={},{};local first=tonumber(regs[dest]:sub(2))
  for r=first,first+width//32-1 do if r>254 then return end;writes['R'..r]=true end
  if base~='RZ'then
   local r=tonumber(regs[base]:sub(2));reads['R'..r]=true
   if full:find('.E',1,true)then if r>=254 then return end;reads['R'..(r+1)]=true end
  end
  if pred then reads[pred:match('P[0-6]')]=true end
  return {line=line,ins=ins:gsub('%.reuse',''),reads=reads,writes=writes,
   -- Completion groups must share the exact load form and predicate mask.
   -- A scalar or differently predicated load is not evidence that an older
   -- vector/other-mask result has completed on every participating lane.
   latency=0,memory=kind=='LDG' and 200 or 30,pipe=full..':'..(pred or ''),class='memory',throughput=2,
   cycles=cycles,old_wait=mask,old_read=tonumber(rd),old_write=tonumber(wr),
   children={},parents=0,release=0}
 end
 if rd~='-' or wr~='-' then return end
 if not alu[kind] then return end
 local dest,rest=args:match('^([^,]+),%s*(.*)');if not dest then return end
 local name=dest:match('^([%w_]+)');if not regs[name]then return end
 local writes={[regs[name]]=true};local reads=regset(rest)
 if pred then reads[pred:match('P[0-6]')]=true;reads[regs[name]]=true end
 if dest:find('.CC',1,true)then writes.CC=true end
 if full:match('%.X$')then reads.CC=true end
 local compare=kind=='ICMP' or kind=='FCMP'
 return {line=line,ins=ins:gsub('%.reuse',''),reads=reads,writes=writes,
  latency=compare and 13 or 6,class=compare and 'compare' or (shift[kind] and 'shift' or 'alu'),
  throughput=(compare or shift[kind])and 2 or 1,cycles=cycles,old_wait=mask,children={},parents=0,release=0}
end
local out,buffer,report={},{},{'operator,line,instructions,loads,edges,original_estimated_cycles,scheduled_estimated_cycles,scheduled_issue_cycles,reordered_instructions,completion_waits,barrier_reuse_waits'}
local op,body,line_no='',false,0
local regions,moved,saved,total_edges,loads_total,waits_total,reuse_total=0,0,0,0,0,0,0
local function flush()
 if #buffer==0 then return end
 local nodes={};local old=0;local original_slots={};local loads=0
 for _,v in ipairs(buffer)do
  for b=1,6 do if v.old_wait&(1<<(b-1))~=0 then old=math.max(old,original_slots[b]or 0);original_slots[b]=nil end end
  if v.memory then
   loads=loads+1
   if v.old_write then original_slots[v.old_write]=math.max(original_slots[v.old_write]or 0,old+v.memory)end
   if v.old_read then original_slots[v.old_read]=math.max(original_slots[v.old_read]or 0,old+20)end
  end
  old=old+v.cycles
  if not v.nop then nodes[#nodes+1]=v;v.id=#nodes end
 end
 for _,at in pairs(original_slots)do old=math.max(old,at)end
 local function original()for _,v in ipairs(buffer)do out[#out+1]=v.line end;buffer={}end
 if #nodes<8 or loads==0 then original();return end
 local writes,readers={},{};local edges=0;local previous_load
 local function edge(a,b,latency)
  if not a or a==b then return end
  if not a.children[b.id]then edges=edges+1;b.parents=b.parents+1 end
  a.children[b.id]=math.max(a.children[b.id] or 0,latency)
 end
 for _,v in ipairs(nodes)do
  if v.memory then edge(previous_load,v,2);previous_load=v end
  for r in pairs(v.reads)do local p=writes[r];if p then edge(p,v,p.latency)end end
  for r in pairs(v.writes)do
   local p=writes[r];if p then edge(p,v,p.latency)end
   for _,reader in pairs(readers[r] or {})do edge(reader,v,1)end
   readers[r]={};writes[r]=v
  end
  for r in pairs(v.reads)do readers[r]=readers[r]or {};readers[r][v.id]=v end
 end
 for i=#nodes,1,-1 do
  local v=nodes[i];v.critical=v.memory or v.latency
  for id,lat in pairs(v.children)do v.critical=math.max(v.critical,lat+nodes[id].critical)end
 end
 local ready={};for _,v in ipairs(nodes)do if v.parents==0 then ready[#ready+1]=v end end
 local schedule={};local clock,estimate,prev=-1,-1,nil
 local slots={};local waits,reuses=0,0
 local function needed(v)
  local mask=0
  for b,slot in pairs(slots)do
   local hazard=false
   for _,owner in ipairs(slot.owners)do
    if slot.mode=='write'then
     for r in pairs(v.reads)do if owner.writes[r]then hazard=true end end
     for r in pairs(v.writes)do
      if owner.writes[r]or (not owner.old_read and owner.reads[r])then hazard=true end
     end
    else for r in pairs(v.writes)do if owner.reads[r]then hazard=true end end end
   end
   if hazard then mask=mask|(1<<(b-1))end
  end
  local reuse=false
  if v.memory then
   for _,entry in ipairs({{v.old_write,'write'},{v.old_read,'read'}})do
    local b,mode=entry[1],entry[2];local prior=b and slots[b]
    if prior and (prior.mode~=mode or prior.pipe~=v.pipe)then mask=mask|(1<<(b-1));reuse=true end
   end
  end
  return mask,v.old_write,reuse
 end
 while #ready>0 do
  local best,at,issue,bestmask,bestslot,bestreuse
  for i,v in ipairs(ready)do
   local fixed=math.max(v.release,clock+(prev and prev.class==v.class and v.throughput or 1))
   local mask,slot,reuse=needed(v);local time=estimate+fixed-clock
   for b=1,6 do if mask&(1<<(b-1))~=0 then time=math.max(time,slots[b].done)end end
   if not best or time<at or (time==at and (v.critical>ready[best].critical or
      (v.critical==ready[best].critical and v.id<ready[best].id)))then
    best,at,issue,bestmask,bestslot,bestreuse=i,time,fixed,mask,slot,reuse
   end
  end
  local v=table.remove(ready,best);v.time=issue;v.wait=bestmask;v.bar=bestslot
  clock=issue;estimate=at;prev=v;schedule[#schedule+1]=v
  if bestmask~=0 then waits=waits+1 end;if bestreuse then reuses=reuses+1 end
  for b=1,6 do if bestmask&(1<<(b-1))~=0 then slots[b]=nil end end
  if v.memory then
   for _,entry in ipairs({{v.old_write,'write'},{v.old_read,'read'}})do
    local b,mode=entry[1],entry[2]
    if b then
     if not slots[b]then slots[b]={mode=mode,pipe=v.pipe,owners={},done=0}end
     assert(slots[b].mode==mode and slots[b].pipe==v.pipe,'unretired scoreboard owner')
     slots[b].owners[#slots[b].owners+1]=v
     slots[b].done=math.max(slots[b].done,estimate+(mode=='write'and v.memory or 20))
    end
   end
  end
  for id,lat in pairs(v.children)do
   local c=nodes[id];c.release=math.max(c.release,clock+lat);c.parents=c.parents-1
   if c.parents==0 then ready[#ready+1]=c end
  end
 end
 assert(#schedule==#nodes,'cyclic dependency graph')
 local cycles=13+estimate+14
 for _,slot in pairs(slots)do cycles=math.max(cycles,13+slot.done+13)end
 if old-cycles<16 or cycles>old*.98 then original();return end
 local reorder=0;for i,v in ipairs(schedule)do if v.id~=i then reorder=reorder+1 end end
 out[#out+1]='3f:-:-:Y:d NOP; # DAG region live-in completion/fixed guard'
 for i,v in ipairs(schedule)do
  local gap=i==#schedule and 1 or schedule[i+1].time-v.time
  assert(gap>=1 and gap<=13,'schedule gap')
  -- A load must not reassign a completion slot in the instruction that waits
  -- for its previous owner. Retire it first, including shared/global pipe
  -- changes. Extra issue time only widens the fixed-latency constraints.
  local mask=v.wait
  if v.bar and mask~=0 then
   out[#out+1]=string.format('%02x:-:-:-:6 NOP; # DAG retire completion before load issue',mask)
   mask=0
  end
  out[#out+1]=string.format('%s:%s:%s:%s:%x %s; # DAG memory scheduled',
   mask~=0 and string.format('%02x',mask)or '--',v.old_read and tostring(v.old_read)or '-',
   v.bar and tostring(v.bar)or '-',gap>4 and 'Y' or '-',gap,v.ins)
 end
 out[#out+1]='3f:-:-:Y:d NOP; # DAG region live-out completion/fixed guard'
 report[#report+1]=string.format('%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d',op,line_no-#buffer,#nodes,loads,edges,old,cycles,27+clock,reorder,waits,reuses)
 regions=regions+1;moved=moved+reorder;saved=saved+old-cycles;total_edges=total_edges+edges
 loads_total=loads_total+loads;waits_total=waits_total+waits;reuse_total=reuse_total+reuses
 buffer={}
end
for line in s:gmatch('[^\n]+')do
 line_no=line_no+1
 local nextop=line:match('^([%w_]+)_SETUP:$')
 if nextop then flush();op=nextop end
 if line=='embrow_SETUP:'then body=true end
 if line=='POSITION_END:'then flush();body=false end
 local v=body and parse(line)
 if v then buffer[#buffer+1]=v;if #buffer>=512 then flush()end
 else flush();out[#out+1]=line end
end
flush();assert(regions>0,'no profitable scheduling regions')
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(out,'\n'),'\n');f:close()
f=assert(io.open(assert(arg[3]),'w'));f:write(table.concat(report,'\n'),'\n');f:close()
io.stderr:write(string.format('dag_memory regions=%d dependency_edges=%d moved=%d loads=%d completion_waits=%d barrier_reuse_waits=%d estimated_cycles_saved=%d arithmetic_reassociated=0\n',regions,total_edges,moved,loads_total,waits_total,reuse_total,saved))
