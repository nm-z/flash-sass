-- List-schedule fixed-latency arithmetic islands in the complete scalar image.
-- Memory, scoreboards, branches, predicate definitions and fences are anchors.
-- RAW/WAR/WAW dependencies use physical registers, including CC and predicates.
-- Floating-point expressions are never reassociated: accumulator RAW edges
-- preserve their exact order. Select only islands whose predicted savings
-- exceed conservative13-cycle live-in/live-out guards. Static cycles are not
-- a throughput measurement. Usage: lua dag_schedule.lua IN OUT REPORT.csv
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
 if not stall or wait~='--' or rd~='-' or wr~='-' then return end
 local pred,body=ins:match('^(@!?P[0-6])%s+(.+)')
 body=body or ins
 local full,args=body:match('^(%S+)%s*(.*)');local kind=full:match('^[^.]+')
 local cycles=tonumber(stall,16);if cycles>11 and y~='Y'then cycles=6 end
 if kind=='NOP' and not pred then return {nop=true,cycles=cycles,line=line}end
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
  throughput=(compare or shift[kind])and 2 or 1,cycles=cycles,children={},parents=0,release=0}
end
local out,buffer,report={},{},{'operator,line,instructions,edges,original_static_cycles,scheduled_guarded_cycles,reordered_instructions'}
local op,body,line_no='',false,0
local regions,moved,saved,total_edges=0,0,0,0
local function flush()
 if #buffer==0 then return end
 local nodes={};local old=0
 for _,v in ipairs(buffer)do old=old+v.cycles;if not v.nop then nodes[#nodes+1]=v;v.id=#nodes end end
 local function original()for _,v in ipairs(buffer)do out[#out+1]=v.line end;buffer={}end
 if #nodes<8 then original();return end
 local writes,readers={},{};local edges=0
 local function edge(a,b,latency)
  if not a or a==b then return end
  if not a.children[b.id]then edges=edges+1;b.parents=b.parents+1 end
  a.children[b.id]=math.max(a.children[b.id] or 0,latency)
 end
 for _,v in ipairs(nodes)do
  for r in pairs(v.reads)do local p=writes[r];if p then edge(p,v,p.latency)end end
  for r in pairs(v.writes)do
   local p=writes[r];if p then edge(p,v,p.latency)end
   for _,reader in pairs(readers[r] or {})do edge(reader,v,1)end
   readers[r]={};writes[r]=v
  end
  for r in pairs(v.reads)do readers[r]=readers[r]or {};readers[r][v.id]=v end
 end
 for i=#nodes,1,-1 do
  local v=nodes[i];v.critical=v.latency
  for id,lat in pairs(v.children)do v.critical=math.max(v.critical,lat+nodes[id].critical)end
 end
 local ready={};for _,v in ipairs(nodes)do if v.parents==0 then ready[#ready+1]=v end end
 local schedule={};local clock,prev=-1,nil
 while #ready>0 do
  local best,at
  for i,v in ipairs(ready)do
   local time=math.max(v.release,clock+(prev and prev.class==v.class and v.throughput or 1))
   if not best or time<at or (time==at and (v.critical>ready[best].critical or
      (v.critical==ready[best].critical and v.id<ready[best].id)))then best,at=i,time end
  end
  local v=table.remove(ready,best);v.time=at;clock=at;prev=v;schedule[#schedule+1]=v
  for id,lat in pairs(v.children)do
   local c=nodes[id];c.release=math.max(c.release,at+lat);c.parents=c.parents-1
   if c.parents==0 then ready[#ready+1]=c end
  end
 end
 assert(#schedule==#nodes,'cyclic dependency graph')
 local cycles=13+clock+13
 if old-cycles<8 then original();return end
 local reorder=0;for i,v in ipairs(schedule)do if v.id~=i then reorder=reorder+1 end end
 out[#out+1]='--:-:-:Y:d NOP; # DAG live-in fixed-result guard'
 for i,v in ipairs(schedule)do
  local gap=i==#schedule and 13 or schedule[i+1].time-v.time
  assert(gap>=1 and gap<=13,'schedule gap')
  out[#out+1]=string.format('--:-:-:%s:%x %s; # DAG scheduled',gap>4 and 'Y' or '-',gap,v.ins)
 end
 report[#report+1]=string.format('%s,%d,%d,%d,%d,%d,%d',op,line_no-#buffer,#nodes,edges,old,cycles,reorder)
 regions=regions+1;moved=moved+reorder;saved=saved+old-cycles;total_edges=total_edges+edges
 buffer={}
end
for line in s:gmatch('[^\n]+')do
 line_no=line_no+1
 local nextop=line:match('^([%w_]+)_SETUP:$')
 if nextop then flush();op=nextop end
 if line=='embrow_SETUP:'then body=true end
 if line=='POSITION_END:'then flush();body=false end
 local v=body and parse(line)
 if v then buffer[#buffer+1]=v;if #buffer>=256 then flush()end
 else flush();out[#out+1]=line end
end
flush();assert(regions>0,'no profitable scheduling regions')
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(out,'\n'),'\n');f:close()
f=assert(io.open(assert(arg[3]),'w'));f:write(table.concat(report,'\n'),'\n');f:close()
io.stderr:write(string.format('dag_schedule regions=%d dependency_edges=%d moved=%d static_cycles_saved=%d memory_order_changed=0 arithmetic_reassociated=0\n',regions,total_edges,moved,saved))
