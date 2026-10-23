-- Full routed-down path with exact integer activation-pair lookup tables.
-- Each expert builds320 tables of16 combinations in shared memory. A padded
-- seventeen-entry stride spreads the banks. The two-lane split still joins
-- integer partials before the unchanged FP32 accumulation/reduction.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local a=assert(s:find('q2moedn_SETUP:',1,true))
local b=assert(s:find('\n[%w_]+_SETUP:',a+1));local body=s:sub(a,b-1)
assert(body:find('q2moedn_SPLIT_DOT:',1,true),'eight-lane input required')
local function once(old,new)
 local p,q=assert(body:find(old,1,true));assert(not body:find(old,q+1,true),'ambiguous '..old)
 body=body:sub(1,p-1)..new..body:sub(q+1)
end
local function gen()
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function label(x)out[#out+1]=x..':'end
 return out,e,label
end
-- Table construction is now the only reader of the activation vector.
-- Remove the superseded per-row register loads and their address setup.
do
 local p=assert(body:find('SHL q2moedn_t1, sd_lane, 0x6;',1,true))
 local first=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local q=assert(body:find('SHL q2moedn_t1, sd_lane, 0x2;',p,true))
 local last=assert(body:sub(1,q-1):match('.*()\n'))
 local removed=body:sub(first,last)
 local _,loads=removed:gsub('LDG%.E%.CG%.128 q2moedn_act','')
 assert(loads==10,'register-activation load coverage')
 body=body:sub(1,first-1)..body:sub(last+1)
end
-- All eight warps participate in table construction for every expert. Invalid
-- tail warps skip matrix loads/stores only after the table-construction join.
once('--:-:-:Y:d @P6 BRA q2moedn_END;\n--:-:1:-:2 LDS br_rows,',
     '--:-:1:-:2 LDS br_rows,')
once('--:-:-:Y:d @P6 BRA q2moedn_END;\nq2moedn_WIDE_TAIL_OK:',
     'q2moedn_WIDE_TAIL_OK:')
local out,e,l=gen()
e('MOV lt_pair, tid')
e('XMAD lt_offset, q2moedn_e, 0x500, RZ','--:-:-:Y:d')
e('IADD lt_base0.CC, q2moedn_param_X0, lt_offset')
e('IADD.X lt_base1, q2moedn_param_X1, RZ')
l('q2moedn_LUT_BUILD')
e('SHL lt_offset, lt_pair, 0x2')
e('IADD lt_x0.CC, lt_base0, lt_offset');e('IADD.X lt_x1, lt_base1, RZ')
e('LDG.E.CG lt_act, [lt_x]','--:-:3:-:2')
e('ISCADD lt_address, lt_pair, lt_pair, 0x4')
e('SHL lt_address, lt_address, 0x2');e('IADD lt_address, lt_address, 0x2000')
e('BFE lt_a, lt_act, 0x1000','04:-:-:-:6')
e('BFE lt_b, lt_act, 0x1010')
e('MOV lt_v0, RZ','--:-:-:-:1');e('MOV lt_v1, lt_a','--:-:-:-:1')
e('IADD lt_v2, lt_a, lt_a');e('IADD lt_v3, lt_v2, lt_a')
for q=0,3 do
 -- The padded68-byte stride is only four-byte aligned. Scalar stores also
 -- distribute the producer accesses across all32 banks.
 for j=0,3 do e(string.format('STS [lt_address+0x%x], lt_v%d',q*16+j*4,j),'--:2:-:-:2')end
 e('NOP','02:-:-:-:6')
 if q<3 then
  for i=0,3 do e(string.format('IADD lt_v%d, lt_v%d, lt_b',i,i),i==3 and '--:-:-:-:6' or '--:-:-:-:1')end
 end
end
e('IADD lt_pair, lt_pair, 0x100')
e('ISETP.LT.U32.AND P6, PT, lt_pair, 0x140, PT','--:-:-:Y:d')
e('NOP');e('@P6 BRA q2moedn_LUT_BUILD','--:-:-:Y:d')
e('BAR.SYNC 0x0','0c:-:-:Y:d')
e('ISETP.GE.U32.AND P6, PT, q2moedn_row, q2moedn_param_rows, PT','--:-:-:Y:d')
e('NOP');e('@P6 BRA q2moedn_LUT_ROW_DONE','--:-:-:Y:d')
-- Pair0 corresponds to the original half-block lane plus its16-value half.
e('SHL lt_pair, sd_lane, 0x4');e('ISCADD lt_pair, sd_part, lt_pair, 0x3')
e('ISCADD lt_tablebase, lt_pair, lt_pair, 0x4');e('SHL lt_tablebase, lt_tablebase, 0x2')
e('IADD lt_tablebase, lt_tablebase, 0x2000')
once('q2moedn_LOOP:',table.concat(out,'\n')..'\nq2moedn_LOOP:')
once('IADD q2moedn_e, q2moedn_e, 0x1;', [=[NOP;
q2moedn_LUT_ROW_DONE:
0c:-:-:Y:d BAR.SYNC 0x0;
--:-:-:-:6 IADD q2moedn_e, q2moedn_e, 0x1;]=])
for chunk=0,4 do
 local p=assert(body:find('LOP32I.AND q2moedn_m0, q2moedn_q'..(2*chunk)..',',1,true))
 local first=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local q=assert(body:find('I2F.F32.S32 q2moedn_dotf, q2moedn_dot;',p,true))
 local last=assert(body:sub(1,q-1):match('.*()\n'))
 out,e,l=gen()
 -- Eight independent address chains; six issues separate dependent stages.
 for j=0,7 do
  e(string.format('BFE.U32 q2moedn_m%d, q2moedn_q%d, 0x%x',j,2*chunk,0x400+j*4),'--:-:-:-:1')
 end
 for j=0,7 do
  e(string.format('ISCADD q2moedn_m%d, q2moedn_m%d, lt_tablebase, 0x2',j,j),'--:-:-:-:1')
 end
 for j=0,7 do
  e(string.format('LDS lt_v%d, [q2moedn_m%d+0x%x]',j,j,(chunk*64+j)*68),'--:-:3:-:2')
 end
 e('IADD3 lt_v0, lt_v0, lt_v1, q2moedn_nsum'..chunk,'04:-:-:-:6')
 e('IADD3 lt_v2, lt_v2, lt_v3, lt_v4')
 e('IADD3 lt_v5, lt_v5, lt_v6, lt_v7')
 e('IADD3 q2moedn_dot, lt_v0, lt_v2, lt_v5')
 e('SHFL.BFLY PT, sd_peer, q2moedn_dot, 0x1, 0x1f','--:-:3:-:3')
 e('IADD q2moedn_dot, q2moedn_dot, sd_peer','04:-:-:-:6')
 body=body:sub(1,first-1)..table.concat(out,'\n')..'\n'..body:sub(last+1)
end
s=s:sub(1,a-1)..body..s:sub(b)
local maps=[=[
    213 : lt_pair
    214 : lt_offset
    215 : lt_address
    216 : lt_act
    217 : lt_a
    218 : lt_b
    220,221 : lt_base<0-1>
    222,223 : lt_x<0-1>
    228-235 : lt_v<0-7>
    236 : lt_tablebase
]=]
assert(0x2000+320*17*4<0xbf00,'LUT overlaps descriptor storage')
local at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..maps..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('down_lut pairs=320 entries=16 stride=17 shared_bytes=21760 integer_arithmetic=exact\n')
