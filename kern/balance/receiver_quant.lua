-- Reconstruct the exact quantization of the received FP32 Z vector locally.
-- The preceding native HCUP5 leaves five immutable descriptor copies at8000.
-- The builder pairs these operations immediately; the ordinary incoming peer
-- acquire completes every FP32 shard before this prefix starts. One local
-- whole-grid join makes all five quantized activations ready for the consumer.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(text,a,b)
 local p,q=assert(text:find(a,1,true));assert(not text:find(a,q+1,true),'ambiguous '..a)
 return text:sub(1,p-1)..b..text:sub(q+1)
end
local maps={[=[
    200 : lq_group
    201 : lq_position
    202 : lq_index
    203 : lq_temp
    204 : lq_descriptor
    205 : lq_die
    206 : lq_epoch
    207 : lq_word
    208,209 : lq_source<0-1>
    210,211 : lq_table<0-1>
    212,213 : lq_ready<0-1>
    214,215 : lq_flag<0-1>
    216 : lq_value
    218,219 : lq_control<0-1>
]=]}
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function label(x)out[#out+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
label('prequant_SETUP');e('NOP','3f:-:-:Y:d')
e('MOV lq_control0, ctl0');e('MOV lq_control1, ctl1')
e('LDG.E.CG lq_die, [lq_control+0xc]','--:-:1:-:2')
e('LDS.64 lq_ready0, [RZ+0xbf98]','--:-:2:-:2')
e('LDS lq_epoch, [RZ+0xbf74]','--:-:3:-:2')
e('ISCADD lq_group, cta, warp, 0x3','07:-:-:-:6')
label('LOCAL_QUANT_GROUP')
pred('ISETP.GE.U32.AND P6, PT, lq_group, 0x190, PT')
e('@P6 BRA LOCAL_QUANT_DONE','--:-:-:Y:d')
e('MOV lq_position, RZ')
for p=1,4 do
 pred(string.format('ISETP.GE.U32.AND P4, PT, lq_group, 0x%x, PT',p*80))
 e(string.format('@P4 MOV lq_position, 0x%x',p))
end
e('XMAD lq_temp, lq_position, 0x50, RZ','--:-:-:Y:d')
e('IADD lq_temp, lq_group, -lq_temp');e('ISCADD lq_index, lq_temp, lane, 0x5')
e('SHL lq_descriptor, lq_position, 0x7');e('IADD lq_descriptor, lq_descriptor, 0x8000')
e('LDS.64 lq_source0, [lq_descriptor+0x38]','--:-:1:-:2')
e('LDS.64 lq_table0, [lq_descriptor+0x58]','--:-:1:-:2')
e('LDS lq_word, [lq_descriptor+0x30]','--:-:1:-:2')
e('SHL lq_temp, lq_die, 0x3')
e('IADD lq_source0.CC, lq_source0, lq_temp','01:-:-:-:6');e('IADD.X lq_source1, lq_source1, RZ')
e('LDG.E.CG.64 lq_source0, [lq_source]','--:-:1:-:2')
e('LOP32I.AND lq_word, lq_word, 0xffff0000')
e('IADD lq_source0.CC, lq_source0, lq_word','01:-:-:-:6');e('IADD.X lq_source1, lq_source1, RZ')
e('SHL lq_temp, lq_index, 0x2')
e('IADD lq_source0.CC, lq_source0, lq_temp');e('IADD.X lq_source1, lq_source1, RZ')
e('LDG.E.CG lq_value, [lq_source]','--:-:1:-:2')
e('SHL lq_temp, lq_die, 0x5')
e('IADD lq_table0.CC, lq_table0, lq_temp');e('IADD.X lq_table1, lq_table1, RZ')
e('NOP','01:-:-:-:6')
local quant=dofile('balance/../phase/quant_emit.lua')(maps)
local q=quant('lqq',1,0xbf98,
 '--:-:-:-:6 MOV %_v0, lq_value;\n--:-:-:-:6 MOV %_index, lq_index;',
 '--:-:-:-:6 IADD lq_group, lq_group, 0x80;\n--:-:-:-:6 BRA LOCAL_QUANT_GROUP;')
q=rep(q,'--:-:1:-:2 LDS.64 lqq_table0, [RZ+0xbf98];',
 '--:-:-:-:6 MOV lqq_table0, lq_table0;\n--:-:-:-:6 MOV lqq_table1, lq_table1;')
q=rep(q,'lqq_peer, 0x6, PT;','lqq_peer, 0x1, PT;')
out[#out+1]=q
label('LOCAL_QUANT_DONE')
e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
e('SHL lq_temp, cta, 0x2');e('IADD lq_flag0.CC, lq_ready0, lq_temp');e('IADD.X lq_flag1, lq_ready1, RZ')
pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
e('@P5 STG.E.WT [lq_flag], lq_epoch','--:2:-:-:2');e('NOP','02:-:-:-:6')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA LOCAL_QUANT_JOIN','--:-:-:Y:d')
e('LOP32I.AND lq_temp, lane, 0xf');e('SHL lq_temp, lq_temp, 0x2')
e('IADD lq_flag0.CC, lq_ready0, lq_temp');e('IADD.X lq_flag1, lq_ready1, RZ')
label('LOCAL_QUANT_WAIT')
e('LDG.E.CV lq_word, [lq_flag]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.GE.U32.AND P5, PT, lq_word, lq_epoch, PT')
pred('VOTE.ALL P5, P5');e('@!P5 BRA LOCAL_QUANT_WAIT','--:-:-:Y:d')
label('LOCAL_QUANT_JOIN');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
e('BRA batch_SETUP','--:-:-:Y:d')
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=rep(s,'ISETP.GE.U32.AND P6, PT, op, 0x6e, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6f, PT;')
s=rep(s,'--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',[=[--:-:-:-:6 BRA prequant_SETUP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6      BRA POSITION_END;

embrow_SETUP:]=])
s=rep(s,'\nPOSITION_END:','\n'..table.concat(out,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('receiver_quant positions=5 groups=400 existing_peer_acquire=1 local_grid_joins=1\n')
