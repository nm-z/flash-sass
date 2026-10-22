-- Explicit completion scheduling for full-model images. Diagnostic modes
-- cover the frame/body; selected mode confines both completion waits and fixed
-- spacing to named operator bodies. Preserve instruction order, phases and
-- dispatch entries. Selected mode is used for the production PLE candidate.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local scope=arg[3] or 'body';assert(scope=='body' or scope=='frame' or scope=='all')
local fixed_mode=arg[4] or ''
assert(fixed_mode=='' or fixed_mode=='none' or fixed_mode=='fixed' or fixed_mode=='fixedbody' or fixed_mode=='fixedframe')
local fixed_ops={};for op in (arg[5] or ''):gmatch('[%w_]+')do fixed_ops[op]=true end
local filtered=next(fixed_ops)~=nil;local active_op=''
local selected=arg[6]=='selected';assert(not selected or filtered)
local out={};local n=0;local in_body=false
for line in s:gmatch('[^\n]+')do
 active_op=line:match('^([%w_]+)_SETUP:$') or active_op
 if line=='embrow_SETUP:' then in_body=true end
 if line=='POSITION_END:' then in_body=false end
 local enabled=scope=='all' or (scope=='body' and in_body) or (scope=='frame' and not in_body)
 if selected then enabled=enabled and in_body and fixed_ops[active_op] end
 local fixed=fixed_mode=='fixed' or (fixed_mode=='fixedbody' and in_body) or (fixed_mode=='fixedframe' and not in_body)
 if filtered then fixed=fixed and fixed_ops[active_op] end
 local ctl,space,ins=line:match('^(%S+)(%s+)(.+);')
 local wait,read,write,yield,stall
 if ctl then wait,read,write,yield,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if stall and enabled then
  local op=ins:match('^@!?P%d%s+(%S+)')or ins:match('^(%S+)')
  local load=op:match('^LD[GCSL]?%.')or op=='LDG'or op=='LDS'or op:match('^S2R')or op:match('^CS2R')or op:match('^SHFL%.')or op:match('^MUFU%.')or op:match('^ATOM%.')
  local store=op:match('^ST[GLS]?%.')or op=='STG'or op=='STS'or op:match('^RED%.')
  if load and write=='-' then write='6' end
  if store and read=='-' then read='6' end
  if fixed then yield='Y';stall='d' end
  local mask=0
  if read~='-' then mask=mask|(1<<(tonumber(read)-1)) end
  if write~='-' then mask=mask|(1<<(tonumber(write)-1)) end
  if mask~=0 then
   line=table.concat({wait,read,write,yield,tonumber(stall,16)<2 and '2' or stall},':')..space..ins..';'
   out[#out+1]=line;out[#out+1]=string.format('%02x:-:-:-:6 NOP; # diagnostic explicit completion',mask);n=n+1
  else
   if fixed then line=table.concat({wait,read,write,yield,stall},':')..space..ins..';' end
   out[#out+1]=line
  end
 else out[#out+1]=line end
end
assert(n>(selected and 50 or (scope=='frame' and 10 or 10000)),'unexpected dependency geometry '..n)
f=assert(io.open(assert(arg[2]),'w'));f:write(table.concat(out,'\n'),'\n');f:close()
io.stderr:write(string.format('serial_dependencies explicit_completions=%d scope=%s fixed_latency=%s fixed_ops=%s selected=%s\n',n,scope,fixed_mode,arg[5] or 'all',tostring(selected)))
