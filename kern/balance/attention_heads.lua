-- Specialize the expanded production packet to two live query heads. Keep
-- every retained head's operations/reduction order; remove independent heads.
return function(body)
 local state,out,pending={}, {},0
 local function role(name)
  local kind,n=name:match('^attnpartn_([%a]+)(%d+)$');n=tonumber(n)
  if not n then return nil end
  if kind=='sacc'then return n%6>=2 end
  if kind=='p'then return n%8>=2 end
  if kind=='q'or kind=='vacc'or kind=='st'then return n//4>=2 end
  if kind=='mx'or kind=='den'or kind=='scale'or kind=='tm'or kind=='dp'or
     kind=='pa'or kind=='pb'or kind=='ga'or kind=='gb'or kind=='ra'or kind=='rb'then return n>=2 end
 end
 local no_dest={STG=true,STS=true,STL=true,BRA=true,BRX=true,NOP=true,BAR=true,
  MEMBAR=true,ISETP=true,FSETP=true,PSETP=true,VOTE=true,RED=true}
 local removed=0
 for line in body:gmatch('[^\n]+')do
  if line:match('^[%w_]+:$')then state={}end
  local ctl,space,ins=line:match('^(%S+)(%s+)(.+);')
  local wait,rd,wr,y,stall
  if ctl then wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')end
  if not stall then out[#out+1]=line else
   local op,args=ins:match('^@!?P[0-6]%s+(%S+)%s*(.*)')
   if not op then op,args=ins:match('^(%S+)%s*(.*)')end
   local base=op:match('^[^.]+');local dest
   if not no_dest[base]then
    dest=args:match('^([%w_]+)');if base=='SHFL'then dest=args:match('^[^,]+,%s*([%w_]+)')end
   end
   -- Vector transfers of head scalars keep only the two live elements. Vector
   -- Q/K/V dimension transfers retain all dimensions of a live head.
   local vector_name=dest
   if base=='STG'or base=='STS'then vector_name=args:match(',%s*([%w_]+)$')end
   if vector_name then
    local kind,n=vector_name:match('^attnpartn_([%a]+)(%d+)$');n=tonumber(n)
    if n and(kind=='p'or kind=='tm'or kind=='mx'or kind=='den'or kind=='ga'or
       kind=='gb'or kind=='ra'or kind=='rb'or kind=='pa'or kind=='pb')and
       op:find('.128',1,true)then
     op=op:gsub('%.128','.64');ins=ins:gsub('%.128','.64',1)
    end
   end
   local bad=dest and role(dest)==true or false
   local sources=args
   if dest then
    if base=='SHFL'then sources=args:match('^[^,]+,[^,]+,%s*(.*)')or ''
    else sources=args:match('^[^,]+,%s*(.*)')or ''end
   end
   for name in sources:gmatch('attnpartn_[%w_]+')do
    if role(name)==true or state[name]then bad=true end
   end
   if bad and dest and role(dest)==false then
    error('live query-head result depends on excluded head: '..ins)
   end
   if dest then state[dest]=bad end
   if bad then
    removed=removed+1;pending=pending|(tonumber(wait,16)or 0)
   else
    local mask=(tonumber(wait,16)or 0)|pending;pending=0
    out[#out+1]=table.concat({mask==0 and '--'or string.format('%02x',mask),rd,wr,y,stall},':')..space..ins..';'
   end
  end
 end
 assert(removed>400,'unexpected two-head specialization geometry '..removed)
 io.stderr:write(string.format('attention_heads removed=%d live_heads_per_unit=2\n',removed))
 return table.concat(out,'\n')..'\n'
end
