-- Audit/fix fixed predicate RAW latency within straight-line SASS blocks.
-- Labels reset dataflow: this does not claim to prove inter-block CFG edges.
-- Usage: lua predicate_latency.lua INPUT OUTPUT CSV [audit]
local input,output,csv=assert(arg[1]),assert(arg[2]),assert(arg[3])
local audit=arg[4]=='audit';local defs={};local cycle=0;local result={};local line_no=0;local hazards=0
local rows={'line,predicate,producer_line,minimum_age,required_age,inserted_nops,consumer'}
local function quote(x)return '"'..tostring(x):gsub('"','""')..'"'end
for line in io.lines(input)do
 line_no=line_no+1
 if line:match('^[%w_]+:$')then defs={};cycle=0 end
 local ctl,ins=line:match('^([^%s]+)%s+(.+);')
 local wait,rd,wr,yield,stall
 if ctl then wait,rd,wr,yield,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
 if stall then
  local pred=ins:match('^@!?P[0-6]');local op,args=ins:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
  if not op then op,args=ins:match('^(%S+)%s*(.*)')end
  local destinations={};local rest=args
  if op:match('SETP')then
   local a,b,tail=args:match('^([^,]+),%s*([^,]+),%s*(.*)')
   if a then destinations[a:gsub('%s','')]=true;destinations[b:gsub('%s','')]=true;rest=tail end
  elseif op:match('^VOTE%.')or op:match('^SHFL%.')then
   local a,tail=args:match('^([^,]+),%s*(.*)');if a then destinations[a:gsub('%s','')]=true;rest=tail end
  end
  local uses={};for p in ((pred or '')..' '..(rest or '')):gmatch('P[0-6]')do uses[p]=true end
  local missing=0;local found={}
  local waited=tonumber(wait,16)or 0
  for p in pairs(uses)do local d=defs[p];if d and not(d.bar and (waited&(1<<(d.bar-1)))~=0) and cycle-d.cycle<13 then
   local age=cycle-d.cycle;missing=math.max(missing,13-age);found[#found+1]={p,d.line,age}
  end end
  local nops=math.ceil(missing/6)
  if not audit then for n=1,nops do result[#result+1]='--:-:-:-:6      NOP; # fixed predicate RAW latency';cycle=cycle+6 end end
  for _,h in ipairs(found)do
   hazards=hazards+1;local values={line_no,h[1],h[2],h[3],13,audit and 0 or nops,ins};local q={}
   for _,v in ipairs(values)do q[#q+1]=quote(v)end;rows[#rows+1]=table.concat(q,',')
  end
  for p in pairs(destinations)do if p:match('^P[0-6]$')then defs[p]={cycle=cycle,line=line_no,bar=tonumber(wr)}end end
  local clocks=tonumber(stall,16);if clocks>11 and yield~='Y'then clocks=6 end
  cycle=cycle+clocks
 end
 result[#result+1]=line
end
local f=assert(io.open(output,'w'));f:write(table.concat(result,'\n'),'\n');f:close()
f=assert(io.open(csv,'w'));f:write(table.concat(rows,'\n'),'\n');f:close()
io.stderr:write(string.format('predicate_latency input=%s hazards=%d mode=%s scope=straight_line_fixed_predicates\n',input,hazards,audit and 'audit' or 'fix'))
