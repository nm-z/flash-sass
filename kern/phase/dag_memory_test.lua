-- CPU contract regression: actual emitted waits must protect vector aliases
-- and both halves of a pending load address. No GPU timing is inferred.
local source={'<REGISTER_MAPPING>','  16-19 : v<0-3>','  18 : alias_v2',
 '  80,81 : address<0-1>','  81 : address_hi_alias','</REGISTER_MAPPING>',
 'embrow_SETUP:'}
local function add(x)source[#source+1]=x end
local function region(label,load,reuse,wait)
 add(label..':');add(load);add(wait..':-:-:-:6 NOP;')
 for r=100,111 do add(string.format('--:-:-:Y:d MOV32I R%d, 0x%x;',r,r))end
 add('--:-:-:Y:d '..reuse..';')
 add('--:-:-:Y:d IADD R65, alias_v2, R100;')
 add('--:-:-:Y:d FFMA R60, v0, R70, R60;')
 add('--:-:-:Y:d FFMA R60, v1, R71, R60;')
 add('--:-:-:-:6 MEMBAR.GL;')
end
region('SHARED_ADDRESS','--:-:1:-:2 LDS.128 v0, [address0];','IADD address0, address0, 0x10','01')
region('GLOBAL_ADDRESS','--:2:1:-:2 LDG.E.CG.128 v0, [address];','IADD address_hi_alias, address_hi_alias, 0x1','03')
region('GLOBAL_NO_READ_BARRIER','--:-:1:-:2 LDG.E.CG.128 v0, [address];','IADD address_hi_alias, address_hi_alias, 0x1','01')
add('--:-:-:-:6 UNKNOWN_OP R1, R2; # preserve unsupported instruction')
add('POSITION_END:');add('--:-:-:-:6 EXIT;')
local input=os.tmpname();local output=os.tmpname();local report=os.tmpname()
local f=assert(io.open(input,'w'));f:write(table.concat(source,'\n'),'\n');f:close()
local saved=arg;local scheduler=arg[1]or 'phase/dag_memory.lua'
arg={input,output,report};dofile(scheduler);arg=saved
f=assert(io.open(output));local text=f:read('a');f:close()
assert(text:find('# DAG memory scheduled',1,true),'fixture was not scheduled')
assert(text:find('UNKNOWN_OP R1, R2; # preserve unsupported instruction',1,true),'unknown instruction changed')
local regs={v0=16,v1=17,v2=18,v3=19,alias_v2=18,address=80,address0=80,address1=81,address_hi_alias=81}
local function rn(name)return regs[name]or tonumber(name:match('^R(%d+)$'))end
local function reads(s)
 local set={};s=s:gsub('0x%x+','')
 for word in s:gmatch('[%a_][%w_]*')do local r=rn(word);if r then set[r]=true end end
 return set
end
local pending={};local checked,loads,fma=0,0,0
local function protected(set,field,ins)
 for _,slot in pairs(pending)do for _,owner in ipairs(slot)do
  for r in pairs(set)do assert(not owner[field][r],string.format('pending %s R%d at %s',field,r,ins))end
 end end
end
local function owner(b,read,write)
 if not b then return end;pending[b]=pending[b]or {};pending[b][#pending[b]+1]={read=read,write=write}
end
for line in text:gmatch('[^\n]+')do
 local ctl,ins=line:match('^(%S+)%s+([^;]+);')
 if ctl then
  local wait,rd,wr=ctl:match('^([^:]+):([^:]+):([^:]+):')
  local mask=tonumber(wait,16)or 0
  for b=1,6 do if mask&(1<<(b-1))~=0 then pending[b]=nil end end
  local op,args=ins:match('^(%S+)%s*(.*)')
  if op:match('^LD[GS]')then
   local dest,base=args:match('^(%w+),%s*%[([%w_]+)')
   local width=tonumber(op:match('%.(%d+)$'))or 32
   local r=assert(rn(base));local read={[r]=true};if op:find('.E',1,true)then read[r+1]=true end
   local write={};r=assert(rn(dest));for n=r,r+width//32-1 do write[n]=true end
   protected(read,'write',ins);protected(write,'read',ins);protected(write,'write',ins)
   rd,wr=tonumber(rd),assert(tonumber(wr));owner(wr,rd and {}or read,write);if rd then owner(rd,read,{})end
   loads=loads+1
  elseif op=='MOV32I'or op=='IADD'or op=='FFMA'then
   local dest,rest=args:match('^([%w_]+),%s*(.*)');local write={[assert(rn(dest))]=true}
   protected(reads(rest),'write',ins);protected(write,'read',ins);protected(write,'write',ins);checked=checked+1
   if op=='FFMA'then
    local term=fma%2==0 and 'v0, R70' or 'v1, R71'
    assert(rest:find(term,1,true),'floating accumulation order changed');fma=fma+1
   end
  end
 end
end
assert(loads==3 and fma==6 and checked>=40,'fixture coverage')
os.remove(input);os.remove(output);os.remove(report)
print('dag_memory_test: vector alias, 64-bit upper address, address without read barrier, accumulation order and unknown boundary PASS')
