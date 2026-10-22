-- Straight-line physical-register dependency audit, including aliases/vector
-- operands. Labels/branches end analysis; predicates are conservatively active.
-- This reports candidates for review, not a full CFG or hardware proof.
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
   assert(#numbers==tonumber(b)-tonumber(a)+1,right)
   for n=tonumber(a),tonumber(b)do
    local at=n-tonumber(a)+1;regs[prefix..string.format('%0'..#a..'d',n)]=numbers[at]
    -- MaxAs creates the unnumbered alias at the first eligible even vector,
    -- only if that exact name is not already a scalar (e.g. lo2 versus lo20).
    if numbers[at]%2==0 and at<#numbers and not regs[prefix]then regs[prefix]=numbers[at]end
   end
  else assert(#numbers==1,right);regs[right]=numbers[1]end
 end
end
local conversion={F2F=true,F2I=true,I2F=true,I2I=true,MUFU=true}
local scalar={MOV=true,MOV32I=true,FADD=true,FMUL=true,FFMA=true,FMNMX=true,FSET=true,ISET=true,
 IADD=true,IADD32I=true,IADD3=true,ISCADD=true,XMAD=true,LOP=true,LOP32I=true,LOP3=true,SHL=true,SHR=true,
 SHF=true,BFE=true,BFI=true,SEL=true,IMNMX=true,FLO=true,POPC=true,PRMT=true,LEA=true}
local no_dest={STG=true,STS=true,STL=true,RED=true,BRA=true,BRX=true,EXIT=true,NOP=true,
 BAR=true,MEMBAR=true,ISETP=true,FSETP=true,PSETP=true,VOTE=true}
local defs,reads,barriers,cycle,line_no,count,unknown={},{},{},0,0,0,{}
f=assert(io.open(assert(arg[2]),'w'));f:write('kind,line,resource,prior_line,minimum_cycles,required_cycles,instruction,prior_instruction\n')
local function report(kind,r,old,age,required,text)
 local resource=(kind=='BARRIER_REUSE' and 'B' or 'R')..r
 count=count+1;f:write(string.format('%s,%d,%s,%d,%d,%d,"%s","%s"\n',kind,line_no,resource,old.line,age,required,text,old.text))
end
local function regset(text)
 text=text:gsub('0[xX]%x+','')
 local out={};for name in text:gmatch('[%a_][%w_]*')do local r=regs[name];if r then out[r]=true end end;return out
end
local function vector(out,name,width)
 local r=regs[name];if r then for n=r,r+width-1 do out[n]=true end end
end
for line in source:gmatch('[^\n]+')do
 line_no=line_no+1
 if line:match('^[%w_]+:$')then defs={};reads={};barriers={};cycle=0 end
 local ctl,text=line:match('^(%S+)%s+(.+);')
 local wait,rd,wr,y,stall
 if ctl then wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if stall then
  local full,args=text:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
  if not full then full,args=text:match('^(%S+)%s*(.*)')end
  local op=full:match('^[^.]+');local operands={};for a in args:gmatch('[^,]+')do operands[#operands+1]=a:match('^%s*(.-)%s*$')end
  local dst={};local first_source=1;local width=tonumber(full:match('%.(%d+)$'))
  width=width==128 and 4 or (width==64 and 2 or 1)
  if not no_dest[op]then
   local at=op=='SHFL' and 2 or 1;local name=operands[at] and operands[at]:match('^([%w_]+)')
   if name and regs[name]then vector(dst,name,width);first_source=at+1 end
  end
  local src=regset(table.concat(operands,',',first_source))
  for name in args:gmatch('%[([%w_]+)')do if regs[name] and full:find('.E',1,true)then vector(src,name,2)end end
  if op=='STG' or op=='STS' or op=='STL'then
   local name=operands[#operands] and operands[#operands]:match('^([%w_]+)');if name then vector(src,name,width)end
  end
  local waited=tonumber(wait,16)or 0
  for b=1,6 do if waited&(1<<(b-1))~=0 then barriers[b]=nil end end
  for _,d in pairs(defs)do if d.bar and waited&(1<<(d.bar-1))~=0 then d.retired=true end end
  for _,list in pairs(reads)do for _,r in ipairs(list)do if r.bar and waited&(1<<(r.bar-1))~=0 then r.retired=true end end end
  local delay=(conversion[op] or op=='LDG' or op=='STG' or op=='ATOM' or op=='RED')and 4 or ((op=='LDS' or op=='STS' or op=='SHFL')and 2 or 0)
  local latency=conversion[op] and 8 or ((op=='FCMP' or op=='ICMP')and 13 or (scalar[op] and 6 or nil))
  local pipeline=(op=='LDG' or op=='LDL' or op=='ATOM')and 'global' or
      (op=='LDS' and 'shared' or (op=='LDC' and 'constant' or
      ((op=='S2R' or op=='CS2R')and 'special' or (op=='SHFL' and 'shuffle' or (conversion[op] and 'sfu' or nil)))))
  local wb=tonumber(wr)
  if wb and pipeline then
   local old=barriers[wb]
   if old and (old.pipeline~=pipeline or pipeline=='special')then
    report('BARRIER_REUSE',wb,old,cycle-old.cycle,-1,text)
   end
   barriers[wb]={pipeline=pipeline,cycle=cycle,line=line_no,text=text}
  end
  for r in pairs(src)do local d=defs[r];if d and not d.retired then
   local age=cycle+delay-d.cycle
   if d.latency and age<d.latency then report('RAW',r,d,age,d.latency,text)
   elseif not d.latency then report('VARIABLE_RAW',r,d,age,-1,text)end
  end end
  for r in pairs(dst)do
   local d=defs[r];local age=d and cycle-d.cycle
   if d and not d.retired and not d.latency then report('VARIABLE_WAW',r,d,age,-1,text)
   elseif d and not d.retired and latency and age+latency<d.latency then report('WAW',r,d,age+latency,d.latency,text)end
   for _,old in ipairs(reads[r]or {})do
    if not old.retired and cycle-old.cycle<old.delay then report('WAR',r,old,cycle-old.cycle,old.delay,text)end
   end
   defs[r]={cycle=cycle,line=line_no,text=text,latency=latency,bar=tonumber(wr)};reads[r]={}
  end
  for r in pairs(src)do
   reads[r]=reads[r]or {};reads[r][#reads[r]+1]={cycle=cycle,line=line_no,text=text,delay=delay,bar=tonumber(rd)or tonumber(wr)}
  end
  if next(dst) and not latency and not(full:match('^LD')or op=='SHFL'or op=='S2R'or op=='CS2R'or op=='ATOM')then unknown[op]=true end
  local clocks=tonumber(stall,16);if clocks>11 and y~='Y'then clocks=6 end;cycle=cycle+clocks
  if op=='BRA'or op=='BRX'or op=='EXIT'then defs={};reads={};barriers={};cycle=0 end
 end
end
f:close();local names={};for op in pairs(unknown)do names[#names+1]=op end;table.sort(names)
io.stderr:write(string.format('physical_dependencies candidates=%d unknown_classes=%s scope=straight_line\n',count,table.concat(names,',')))
