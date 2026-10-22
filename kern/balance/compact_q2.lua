-- Fold the five identical half-block dot bodies into one loop. Bind each
-- half-block's operands into free, bank-aligned registers without overwriting
-- the original activations, weights or sums. No arithmetic is reassociated.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local maps={[=[
    180-195 : cq_act<00-15>
    196,197 : cq_q<0-1>
    199 : cq_nsum
    200 : cq_d
    202 : cq_chunk
    203 : cq_scale_address
]=]}
for _,tag in ipairs({'q2moegn','q2moedn'}) do
 local first=assert(s:find(tag..'_SETUP:',1,true))
 local last=assert(s:find('\n[%w_]+_SETUP:',first+1))
 local body=s:sub(first,last-1);local chunks,starts,ends={},{},{}
 for i=0,4 do
  starts[i]=assert(body:find('[^\n]*F2F%.F32%.F16 '..tag..'_t, '..tag..'_d'..i..';'))
  local _,stop=assert(body:find('FFMA '..tag..'_accf, '..tag..'_d'..i..', '..tag..'_u, '..tag..'_accf;',starts[i],true))
  ends[i]=stop
  local x=body:sub(starts[i],stop)
  x=x:gsub(tag..'_act(%d%d)',function(n)return 'cq_act'..string.format('%02d',tonumber(n)-16*i)end)
  x=x:gsub(tag..'_q(%d)',function(n)return 'cq_q'..(tonumber(n)-2*i)end)
  x=x:gsub(tag..'_nsum'..i,'cq_nsum'):gsub(tag..'_d'..i,'cq_d')
  x=x:gsub('%['..tag..'_smaddr%+0x%x+%]','[cq_scale_address]')
  x=x:gsub('^[^ ]+','03:-:4:-:2')
  chunks[i]=x
  if i>0 then
   assert(x==chunks[0],'half-block arithmetic differs: '..tag..' '..i)
   assert(body:sub(ends[i-1]+1,starts[i]-1):match('^%s*$'),'unexpected inter-block work')
  end
 end
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:Y:d')..' '..x..';'end
 local function label(x)out[#out+1]=x..':'end
 local prefix=tag..'_COMPACT_'
 e('NOP','3f:-:-:Y:d');e('MOV cq_chunk, RZ')
 label(prefix..'DISPATCH')
 for i=0,3 do
  e(string.format('ISETP.EQ.U32.AND P6, PT, cq_chunk, 0x%x, PT',i))
  e('@P6 BRA '..prefix..'BIND'..i)
 end
 e('BRA '..prefix..'BIND4')
 for i=0,4 do
  label(prefix..'BIND'..i)
  for j=0,15 do e(string.format('MOV cq_act%02d, %s_act%02d',j,tag,16*i+j),'--:-:-:-:1') end
  e(string.format('MOV cq_q0, %s_q%d',tag,2*i),'--:-:-:-:1')
  e(string.format('MOV cq_q1, %s_q%d',tag,2*i+1),'--:-:-:-:1')
  e(string.format('MOV cq_d, %s_d%d',tag,i),'--:-:-:-:1')
  e(string.format('MOV cq_nsum, %s_nsum%d',tag,i),'--:-:-:-:1')
  e(string.format('IADD cq_scale_address, %s_smaddr, 0x%x',tag,i*4))
  e('BRA '..prefix..'DOT')
 end
 label(prefix..'DOT');out[#out+1]=chunks[0]
 e('IADD cq_chunk, cq_chunk, 0x1')
 e('ISETP.LT.U32.AND P6, PT, cq_chunk, 0x5, PT')
 e('@P6 BRA '..prefix..'DISPATCH')
 body=body:sub(1,starts[0]-1)..table.concat(out,'\n')..body:sub(ends[4]+1)
 s=s:sub(1,first-1)..body..s:sub(last)
 io.stderr:write('compact_q2 '..tag..' half_blocks=5 arithmetic_body_copies=1 operand_registers=20\n')
end
local at=assert(s:find('</REGISTER_MAPPING>',1,true))
s=s:sub(1,at-1)..table.concat(maps,'\n')..'\n'..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
