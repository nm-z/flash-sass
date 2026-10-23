-- Batch only adjacent, independent memory operations whose diagnostic waits
-- use the same scoreboard slots and memory pipeline. Keep the final wait
-- and all original wait bits. Long bursts preserve an18-cycle entry window
-- for preceding fixed results, then issue independent operations two cycles apart.
-- Physical register aliases and vector/address widths are checked explicitly.
-- Predicated operations share a burst only when their predicates are identical;
-- adjacent memory-operation/NOP pairs cannot modify that predicate between issues.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local regs={};for n=0,254 do regs['R'..n]=n end
for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 local left,right=line:match('^%s*([%d,%-%s]+)%s*:%s*([^#]+)')
 if left then
  right=right:match('^%s*(.-)%s*$');local ns={}
  for part in left:gmatch('[^,%s]+')do
   local a,b=part:match('^(%d+)%-(%d+)$')
   if a then for n=tonumber(a),tonumber(b)do ns[#ns+1]=n end else ns[#ns+1]=assert(tonumber(part))end
  end
  local prefix,a,b=right:match('^([%w_]+)<(%d+)%-(%d+)>$')
  if prefix then
   assert(#ns==tonumber(b)-tonumber(a)+1)
   for n=tonumber(a),tonumber(b)do
    local at=n-tonumber(a)+1;regs[prefix..string.format('%0'..#a..'d',n)]=ns[at]
    if ns[at]%2==0 and at<#ns and not regs[prefix]then regs[prefix]=ns[at]end
   end
  elseif #ns==1 then regs[right]=ns[1] end
 end
end
local lines={};for line in (s..'\n'):gmatch('(.-)\n')do lines[#lines+1]=line end
local function clocks(y,n)n=tonumber(n,16);return n>11 and y~='Y' and 6 or n end
local function parse(i)
 if not lines[i] then return end
 local ctl,space,ins=lines[i]:match('^(%S+)(%s+)(.+)$')
 if not ctl then return end
 local wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):(%x)$')
 if not wr or (wr=='-' and rd=='-') then return end
 local pred,load=ins:match('^(@!?P[0-6])%s+(.+)$')
 if not pred then pred='';load=ins end
 local op,dest,addr=load:match('^(%S+)%s+([%w_]+),%s*%[([^%]]+)%];')
 local store=false
 if not op then
  op,addr,dest=load:match('^(%S+)%s+%[([^%]]+)%],%s*([%w_]+);')
  store=true
 end
 local kind=op and op:match('^[^.]+')
 if (not store and kind~='LDG' and kind~='LDS') or
    (store and kind~='STG' and kind~='STS') or not regs[dest] then return end
 local nm,ny,nt=(lines[i+1] or ''):match('^(%x+):%-:%-:([^:]+):(%x)%s+NOP; # diagnostic explicit completion$')
 if not nm then return end
 local mask=wr~='-' and 1<<(tonumber(wr)-1) or 0
 if rd~='-' then mask=mask|(1<<(tonumber(rd)-1))end
 if tonumber(nm,16)~=mask then return end
 local writes,reads={},{};local width=tonumber(op:match('%.(%d+)$')) or 32
 for n=regs[dest],regs[dest]+math.max(1,width//32)-1 do
  if store then reads[n]=true else writes[n]=true end
 end
 for name in addr:gsub('0[xX]%x+',''):gmatch('[%a_][%w_]*')do
  if regs[name]then reads[regs[name]]=true
  elseif name~='RZ' and name~='reuse' then return end
 end
 if op:find('.E',1,true)then local name=addr:match('^%s*([%w_]+)');if name and regs[name]then reads[regs[name]+1]=true end end
 return {line=lines[i],nop=lines[i+1],space=space,ins=ins,wait=wait,mask=mask,rd=rd,wr=wr,kind=kind,
         writes=writes,reads=reads,predicate=pred,cycles=clocks(y,stall)+clocks(ny,nt)}
end
local function overlaps(a,b)for r in pairs(a)do if b[r]then return true end end;return false end
local function union(a,b)for r in pairs(b)do a[r]=true end end
local out={};local i,groups,removed,predicated,saved,stores=1,0,0,0,0,0
while i<=#lines do
 local first=parse(i)
 if not first then out[#out+1]=lines[i];i=i+1 else
  local group={first};local writes,reads={},{};union(writes,first.writes);union(reads,first.reads)
  local j=i+2
  while true do
   local next=parse(j)
   if not next or next.kind~=first.kind or next.rd~=first.rd or next.wr~=first.wr or
      next.predicate~=first.predicate or
      ((tonumber(next.wait,16) or 0)&first.mask)~=0 or
      overlaps(next.reads,writes) or overlaps(next.writes,writes) or overlaps(next.writes,reads) then break end
   group[#group+1]=next;union(writes,next.writes);union(reads,next.reads);j=j+2
  end
  if #group>1 then
   groups=groups+1;removed=removed+#group-1
   if first.predicate~='' then predicated=predicated+1 end
   if first.kind=='STG' or first.kind=='STS' then stores=stores+1 end
  end
  for n,v in ipairs(group)do
   if n==#group then out[#out+1]=v.line;out[#out+1]=v.nop else
    -- The first operation retains time for preceding fixed registers and
    -- predicates to mature before a later operation reads them. Alias checks
    -- exclude intra-burst result and operand-lifetime dependencies. The final
    -- wait covers every result and address/data-register read; memory fences
    -- and the instruction order of stores remain unchanged.
    local cycles=v.cycles
    if #group>=4 then cycles=n==1 and math.max(v.cycles,18) or 2 end
    saved=saved+v.cycles-cycles
    local delay=math.min(cycles,13)
    out[#out+1]=table.concat({v.wait,v.rd,v.wr,'Y',string.format('%x',delay)},':')..v.space..v.ins
    if cycles>delay then out[#out+1]=string.format('--:-:-:Y:%x NOP; # incoming fixed-result spacing',cycles-delay)end
   end
  end
  i=j
 end
end
assert(groups>0,'no independent memory bursts found')
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(out,'\n'));f:close()
io.stderr:write(string.format('load_bursts groups=%d predicated_groups=%d store_groups=%d intermediate_completion_waits_removed=%d static_cycles_removed=%d arithmetic_reordered=0\n',groups,predicated,stores,removed,saved))
