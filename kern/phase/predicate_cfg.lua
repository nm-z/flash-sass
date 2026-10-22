-- Bounded shortest-path predicate RAW audit across labels and direct branches.
-- Unknown BRX targets end a path. Report candidates, not a correctness proof.
local input,output=assert(arg[1]),assert(arg[2]);local ins,labels,pending={},{},{}
for line in io.lines(input)do
 local label=line:match('^([%w_]+):$')
 if label then pending[#pending+1]=label end
 local ctl,text=line:match('^(%S+)%s+(.+);')
 if ctl then
  local wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')
  if stall then
   local pred=text:match('^@!?P[0-6]');local op,args=text:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
   if not op then op,args=text:match('^(%S+)%s*(.*)')end
   local defs,uses={},{};local rest=args
   if op:match('SETP')then
    local a,b,tail=args:match('^([^,]+),%s*([^,]+),%s*(.*)')
    if a then defs[a:gsub('%s','')]=true;defs[b:gsub('%s','')]=true;rest=tail end
   elseif op:match('^VOTE%.')then
    local a,tail=args:match('^([^,]+),%s*(.*)');defs[a]=true;rest=tail
   end
   for p in ((pred or '')..' '..(rest or '')):gmatch('P[0-6]')do uses[p]=true end
   local cost=tonumber(stall,16);if cost>11 and y~='Y'then cost=6 end
   local n=#ins+1;for _,l in ipairs(pending)do labels[l]=n end;pending={}
   ins[n]={text=text,defs=defs,uses=uses,cost=cost,op=op,args=args,pred=pred,wr=wr}
  end
 end
end
local function successors(i)
 local a=ins[i]
 if a.op=='EXIT' or a.op=='BRX' then return {} end
 if a.op=='BRA' then
  local target=assert(labels[a.args:match('^([%w_]+)')],a.text)
  return a.pred and {target,i+1}or {target}
 end
 return {i+1}
end
local f=assert(io.open(output,'w'));f:write('producer_instruction,consumer_instruction,predicate,minimum_cycles,producer,consumer\n')
local hazards=0
for start,a in ipairs(ins)do
 if a.wr=='-' then for p in pairs(a.defs)do if p:match('^P[0-6]$')then
  local queue,head,best={},1,{}
  for _,i in ipairs(successors(start))do queue[#queue+1]={i,a.cost}end
  while queue[head]do
   local i,age=table.unpack(queue[head]);head=head+1;local b=ins[i]
   if b and age<13 and (not best[i] or age<best[i])then
    best[i]=age
    if b.uses[p] then
     f:write(string.format('%d,%d,%s,%d,"%s","%s"\n',start,i,p,age,a.text,b.text));hazards=hazards+1
    end
    if not b.defs[p]then for _,j in ipairs(successors(i))do queue[#queue+1]={j,age+b.cost}end end
   end
  end
 end end end
end
f:close();io.stderr:write(string.format('predicate_cfg instructions=%d candidates=%d unknown_indirect_edges=true\n',#ins,hazards))
