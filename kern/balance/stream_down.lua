-- A two-body software pipeline for all five Q2 down half-blocks. Alternating
-- register banks eliminate operand-copy dispatch; each body prefetches the
-- next block's activations, codes and scales while computing the current one.
-- The five FP32 FMAs and final lane reduction retain their original order.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local a=assert(s:find('q2moedn_SETUP:',1,true))
local b=assert(s:find('\n[%w_]+_SETUP:',a+1));local body=s:sub(a,b-1)
assert(not body:find('q2moedn_PREFETCH_DONE:',1,true),'use native down input')
local function once(old,new)
 local p,q=assert(body:find(old,1,true));assert(not body:find(old,q+1,true),'ambiguous '..old)
 body=body:sub(1,p-1)..new..body:sub(q+1)
end
local function remove_line(ins)
 local p=assert(body:find(ins,1,true));assert(not body:find(ins,p+#ins,true))
 local first=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local last=assert(body:find('\n',p,true));body=body:sub(1,first-1)..body:sub(last+1)
end
-- Preserve address construction, replacing only the old eager data loads.
for chunk=0,4 do
 for j=0,3 do remove_line(string.format('LDG.E.CG.128 q2moedn_act%02d, [q2moedn_xp+0x%x];',chunk*16+j*4,chunk*256+j*16))end
 remove_line(string.format('LDG.E.CG q2moedn_s%d, [q2moedn_sp+0x%x];',chunk,chunk*16))
 remove_line(string.format('LDG.E.CG q2moedn_as%d, [q2moedn_ap+0x%x];',chunk,chunk*16))
 remove_line(string.format('STS [q2moedn_smaddr+0x%x], q2moedn_as%d;',chunk*4,chunk))
 remove_line(string.format('IADD3 q2moedn_nsum%d, RZ, -q2moedn_s%d, RZ;',chunk,chunk))
 remove_line(string.format('LDG.E.CI.U16 q2moedn_d%d, [q2moedn_ws+0x%x];',chunk,chunk*4))
 remove_line(string.format('LDG.E.CI.64 q2moedn_q%d, [q2moedn_wq+0x%x];',chunk*2,chunk*32))
end
local first=assert(body:find('[^\n]*F2F%.F32%.F16 q2moedn_t, q2moedn_d0;'))
local _,last=assert(body:find('FFMA q2moedn_accf, q2moedn_d4, q2moedn_u, q2moedn_accf;',first,true))
local _,end0=assert(body:find('FFMA q2moedn_accf, q2moedn_d0, q2moedn_u, q2moedn_accf;',first,true))
local core=body:sub(first,end0)
assert(core:find('LDS q2moedn_d0, [q2moedn_smaddr+0x0];',1,true))
-- The new global AS load replaces the temporary shared scale slot.
core=core:gsub('[^\n]*LDS q2moedn_d0, %[%s*q2moedn_smaddr%+0x0%];\n','')
core=core:gsub('^%S+','--:-:4:-:2')
core=core:gsub('FFMA q2moedn_accf, q2moedn_d0,','FFMA q2moedn_accf, st_as0,')
local function codegen()
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function l(x)out[#out+1]=x..':'end
 return out,e,l
end
local out,e,l=codegen()
for _,x in ipairs({{'st_wq','q2moedn_wq'},{'st_ws','q2moedn_ws'},
                  {'st_xp','q2moedn_xp'},{'st_sp','q2moedn_sp'},{'st_ap','q2moedn_ap'}})do
 e('MOV '..x[1]..'0, '..x[2]..'0');e('MOV '..x[1]..'1, '..x[2]..'1')
end
e('MOV st_chunk, RZ')
local function load(bank)
 local slot=bank+1
 local ctl=string.format('--:6:%d:-:2',slot)
 e(string.format('LDG.E.CI.U16 q2moedn_d%d, [st_ws]',bank),ctl)
 e(string.format('LDG.E.CI.64 q2moedn_q%d, [st_wq]',2*bank),ctl)
 for j=0,3 do e(string.format('LDG.E.CG.128 q2moedn_act%02d, [st_xp+0x%x]',16*bank+4*j,16*j),ctl)end
 e(string.format('LDG.E.CG q2moedn_nsum%d, [st_sp]',bank),ctl)
 e(string.format('LDG.E.CG st_as%d, [st_ap]',bank),ctl)
 e('NOP','20:-:-:-:6') -- retire address reads, keep results in flight
end
local function advance()
 for _,x in ipairs({{'st_wq',32},{'st_ws',4},{'st_xp',256},{'st_sp',16},{'st_ap',16}})do
  e(string.format('IADD %s0.CC, %s0, 0x%x',x[1],x[1],x[2]))
  e(string.format('IADD.X %s1, %s1, RZ',x[1],x[1]))
 end
end
local function arithmetic(bank)
 local x=core
 if bank==1 then
  x=x:gsub('q2moedn_act(%d%d)',function(n)return 'q2moedn_act'..string.format('%02d',tonumber(n)+16)end)
  x=x:gsub('q2moedn_q([01])',function(n)return 'q2moedn_q'..(tonumber(n)+2)end)
  x=x:gsub('q2moedn_d0','q2moedn_d1'):gsub('q2moedn_nsum0','q2moedn_nsum1'):gsub('st_as0','st_as1')
 end
 out[#out+1]=x
end
load(0)
l('q2moedn_STREAM_A')
e('IADD3 q2moedn_nsum0, RZ, -q2moedn_nsum0, RZ','01:-:-:-:6')
e('IADD st_chunk, st_chunk, 0x1')
e('ISETP.LT.U32.AND P6, PT, st_chunk, 0x5, PT','--:-:-:Y:d')
e('NOP');e('@!P6 BRA q2moedn_STREAM_A_LAST','--:-:-:Y:d')
advance();load(1)
l('q2moedn_STREAM_A_LAST');arithmetic(0)
e('@!P6 BRA q2moedn_STREAM_DONE','--:-:-:Y:d')
l('q2moedn_STREAM_B')
e('IADD3 q2moedn_nsum1, RZ, -q2moedn_nsum1, RZ','02:-:-:-:6')
e('IADD st_chunk, st_chunk, 0x1')
advance();load(0);arithmetic(1)
e('BRA q2moedn_STREAM_A','--:-:-:Y:d')
l('q2moedn_STREAM_DONE')
body=body:sub(1,first-1)..table.concat(out,'\n')..body:sub(last+1)
s=s:sub(1,a-1)..body..s:sub(b)
local maps=[=[
    180,181 : st_wq<0-1>
    182,183 : st_ws<0-1>
    184,185 : st_xp<0-1>
    186,187 : st_sp<0-1>
    188,189 : st_ap<0-1>
    190,191 : st_as<0-1>
    192 : st_chunk
]=]
local at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..maps..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('stream_down half_blocks=5 arithmetic_bodies=2 operand_copies=0 memory_results=slots1,2 address_reads=slot6 fp32_order=preserved\n')
