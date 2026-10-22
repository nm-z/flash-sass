-- Repair fixed-result dependencies shortened by head specialization. Variable
-- completions are made explicit by serial_dependencies before this pass.
-- Retain arithmetic order; insert only latency required by a live dependency.
local f=assert(io.open(assert(arg[1])));local source=f:read('a');f:close()
local regs={};for n=0,254 do regs['R'..n]=n end
for line in assert(source:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 local left,right=line:match('^%s*([%d,%-%s]+)%s*:%s*([^#]+)')
 if left then
  right=right:match('^%s*(.-)%s*$');local numbers={}
  for part in left:gmatch('[^,%s]+')do
   local a,b=part:match('^(%d+)%-(%d+)$')
   if a then for n=tonumber(a),tonumber(b)do numbers[#numbers+1]=n end
   else numbers[#numbers+1]=assert(tonumber(part))end
  end
  local prefix,a,b=right:match('^([%w_]+)<(%d+)%-(%d+)>$')
  if prefix then
   assert(#numbers==tonumber(b)-tonumber(a)+1)
   for n=tonumber(a),tonumber(b)do
    local at=n-tonumber(a)+1;regs[prefix..string.format('%0'..#a..'d',n)]=numbers[at]
    if numbers[at]%2==0 and at<#numbers and not regs[prefix]then regs[prefix]=numbers[at]end
   end
  else assert(#numbers==1,right);regs[right]=numbers[1]end
 end
end
local selected={attnpartn=true,kvgather=true,attnqkq=true,attnquant=true}
local convert={F2F=true,F2I=true,I2F=true,I2I=true,MUFU=true}
local scalar={MOV=true,MOV32I=true,FADD=true,FMUL=true,FFMA=true,FMNMX=true,FSET=true,ISET=true,
 IADD=true,IADD32I=true,IADD3=true,ISCADD=true,XMAD=true,LOP=true,LOP32I=true,LOP3=true,
 SHL=true,SHR=true,SHF=true,BFE=true,BFI=true,SEL=true,IMNMX=true,FLO=true,POPC=true,PRMT=true,LEA=true}
local no_dest={STG=true,STS=true,STL=true,RED=true,BRA=true,BRX=true,EXIT=true,NOP=true,
 BAR=true,MEMBAR=true,ISETP=true,FSETP=true,PSETP=true,VOTE=true}
local function regset(s)
 local out={};s=s:gsub('0[xX]%x+','')
 for name in s:gmatch('[%a_][%w_]*')do if regs[name]then out[regs[name]]=true end end
 return out
end
local function vector(out,name,n)
 if regs[name]then for r=regs[name],regs[name]+n-1 do out[r]=true end end
end
local op,cycle,defs,reads='',0,{},{};local output={};local gaps=0
for line in source:gmatch('[^\n]+')do
 op=line:match('^([%w_]+)_SETUP:$')or op
 if line:match('^[%w_]+:$')then cycle=0;defs={};reads={}end
 local ctl,space,ins=line:match('^(%S+)(%s+)(.+);')
 local wait,rd,wr,y,stall
 if ctl then wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if selected[op]and stall then
  local full,args=ins:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
  if not full then full,args=ins:match('^(%S+)%s*(.*)')end
  local kind=full:match('^[^.]+');local operands={}
  for x in args:gmatch('[^,]+')do operands[#operands+1]=x:match('^%s*(.-)%s*$')end
  local dest,first={},1;local bits=tonumber(full:match('%.(%d+)$'))
  local width=bits==128 and 4 or(bits==64 and 2 or 1)
  if not no_dest[kind]then
   local at=kind=='SHFL'and 2 or 1;local name=operands[at]and operands[at]:match('^([%w_]+)')
   if name and regs[name]then vector(dest,name,width);first=at+1 end
  end
  local src=regset(table.concat(operands,',',first))
  for name in args:gmatch('%[([%w_]+)')do if full:find('.E',1,true)then vector(src,name,2)end end
  if kind=='STG'or kind=='STS'or kind=='STL'then
   local name=operands[#operands]and operands[#operands]:match('^([%w_]+)');if name then vector(src,name,width)end
  end
  local mask=tonumber(wait,16)or 0
  for _,d in pairs(defs)do if d.bar and mask&(1<<(d.bar-1))~=0 then d.retired=true end end
  for _,list in pairs(reads)do for _,r in ipairs(list)do if r.bar and mask&(1<<(r.bar-1))~=0 then r.retired=true end end end
  local delay=(convert[kind]or kind=='LDG'or kind=='STG'or kind=='ATOM'or kind=='RED')and 4 or
   ((kind=='LDS'or kind=='STS'or kind=='SHFL')and 2 or 0)
  local latency=convert[kind]and 8 or((kind=='ICMP'or kind=='FCMP')and 13 or(scalar[kind]and 6 or nil))
  local gap=0
  for r in pairs(src)do local d=defs[r];if d and not d.retired then
   if d.latency then gap=math.max(gap,d.cycle+d.latency-cycle-delay)
   else error('unretired variable input R'..r..': '..ins)end
  end end
  for r in pairs(dest)do
   local d=defs[r]
   if d and not d.retired then
    if d.latency and latency then gap=math.max(gap,d.cycle+d.latency-cycle-latency)
    elseif not d.latency then error('unretired variable overwrite R'..r..': '..ins)end
   end
   for _,old in ipairs(reads[r]or {})do if not old.retired then gap=math.max(gap,old.cycle+old.delay-cycle)end end
  end
  if gap>0 then
   assert(gap<=13);output[#output+1]=string.format('--:-:-:Y:%x NOP; # live fixed-result dependency',gap)
   cycle=cycle+gap;gaps=gaps+1
  end
  for r in pairs(dest)do defs[r]={cycle=cycle,latency=latency,bar=tonumber(wr)};reads[r]={}end
  for r in pairs(src)do reads[r]=reads[r]or {};reads[r][#reads[r]+1]={cycle=cycle,delay=delay,bar=tonumber(rd)or tonumber(wr)}end
  local clocks=tonumber(stall,16);if clocks>11 and y~='Y'then clocks=6 end;cycle=cycle+clocks
 end
 output[#output+1]=line
end
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(output,'\n'),'\n');f:close()
io.stderr:write(string.format('balanced_attention fixed_dependency_gaps=%d\n',gaps))
