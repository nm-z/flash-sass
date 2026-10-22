-- Conservative same-name, straight-line fixed-latency RAW candidates.
-- Variable completion waits can add cycles; aliases and CFG edges are not
-- modeled. This report is a review aid, not a hardware scheduling proof.
local f=assert(io.open(assert(arg[1])));local source=f:read('a');f:close()
local names={}
for line in assert(source:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 local rhs=line:match(':%s*(.-)%s*$')
 if rhs then
  local prefix,a,b=rhs:match('^([%w_]+)<(%d+)%-(%d+)>$')
  if prefix then for n=tonumber(a),tonumber(b)do names[prefix..n]=true;names[prefix..string.format('%02d',n)]=true end
  else names[rhs]=true end
 end
end
local fixed={MOV=true,MOV32I=true,FADD=true,FMUL=true,FFMA=true,FMNMX=true,FSEL=true,
 IADD=true,IADD32I=true,ISCADD=true,XMAD=true,LOP=true,LOP32I=true,SHL=true,SHR=true,
 BFE=true,BFI=true,SEL=true,IMNMX=true,FLO=true,POPC=true,PRMT=true,
 F2F=true,F2I=true,I2F=true,I2I=true,MUFU=true}
local conversion={F2F=true,F2I=true,I2F=true,I2I=true,MUFU=true}
local defs,cycle,line_no,count={},0,0,0
f=assert(io.open(assert(arg[2]),'w'));f:write('line,register,producer_line,minimum_cycles,consumer\n')
for line in source:gmatch('[^\n]+')do
 line_no=line_no+1
 if line:match('^[%w_]+:$')then defs={};cycle=0 end
 local ctl,text=line:match('^(%S+)%s+(.+);')
 local wait,rd,wr,y,stall
 if ctl then wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if stall then
  local op,args=text:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
  if not op then op,args=text:match('^(%S+)%s*(.*)')end
  op=op:match('^[^.]+')
  local dest,rest=args:match('^([^,]+),%s*(.*)')
  local reg=dest and dest:match('^([%w_]+)')
  local hasdest=reg and names[reg] and not (op:match('^ST') or op=='RED')
  local used={};for r in (hasdest and rest or args):gmatch('[%a_][%w_]*')do if names[r]then used[r]=true end end
  local read_delay=(conversion[op] or op=='LDG' or op=='STG') and 4 or ((op=='LDS' or op=='STS' or op=='SHFL') and 2 or 0)
  local waited=tonumber(wait,16)or 0
  for _,d in pairs(defs)do if d.bar and (waited&(1<<(d.bar-1)))~=0 then d.latency=0 end end
  for r in pairs(used)do local d=defs[r];local age=d and cycle+read_delay-d.cycle
  if d and age<d.latency and not(d.bar and (waited&(1<<(d.bar-1)))~=0)then
   f:write(string.format('%d,%s,%d,%d,"%s"\n',line_no,r,d.line,age,text));count=count+1
  end end
  if hasdest then defs[reg]=fixed[op] and {cycle=cycle,line=line_no,latency=conversion[op] and 8 or 6,bar=tonumber(wr)} or nil end
  local clocks=tonumber(stall,16);if clocks>11 and y~='Y'then clocks=6 end;cycle=cycle+clocks
  if op=='BRA' or op=='BRX' or op=='EXIT'then defs={};cycle=0 end
 end
end
f:close();io.stderr:write(string.format('register_latency same_name_candidates=%d\n',count))
