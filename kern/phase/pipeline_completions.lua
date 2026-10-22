-- Retire a completion barrier before assigning it to another execution pipe.
-- Same-pipe groups retain their batching. This does not add model phases or
-- alter operands; it makes the shared/global/SFU handoff explicit.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local pending,out,count={},{},0
local sfu={F2F=true,F2I=true,I2F=true,I2I=true,MUFU=true}
for line in s:gmatch('[^\n]+')do
 if line:match('^[%w_]+:$')then pending={} end
 local ctl,ins=line:match('^(%S+)%s+(.+);')
 local wait,rd,wr,y,stall
 if ctl then wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if stall then
  local op=ins:match('^@!?P[0-6]%s+(%w+)')or ins:match('^(%w+)')
  local pipe=(op=='LDG'or op=='LDL'or op=='ATOM')and 'global'or
   (op=='LDS'and 'shared'or(op=='LDC'and 'constant'or
   ((op=='S2R'or op=='CS2R')and 'special'or(op=='SHFL'and 'shuffle'or(sfu[op]and 'sfu'or nil)))))
  local mask=tonumber(wait,16)or 0
  for b=1,6 do if mask&(1<<(b-1))~=0 then pending[b]=nil end end
  local b=tonumber(wr)
  if b and pipe then
   if pending[b]and(pending[b]~=pipe or pipe=='special')then
    out[#out+1]=string.format('%02x:-:-:-:6 NOP; # retire prior pipeline before completion-barrier reuse',1<<(b-1))
    count=count+1
   end
   pending[b]=pipe
  end
  if op=='BRA'or op=='BRX'or op=='EXIT'then pending={} end
 end
 out[#out+1]=line
end
assert(count>20,'unexpected mixed-pipeline completion geometry '..count)
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(out,'\n'),'\n');f:close()
io.stderr:write(string.format('pipeline_completions explicit_handoffs=%d\n',count))
