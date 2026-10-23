-- One die prepares each complete32-value group, publishes its packed codes
-- and metadata, then all dies acquire all96 preparation producers.
return function(source,maps)
 maps[#maps+1]=[=[
    16 : bact_gate
    17 : bact_up
    18 : bact_value
    19 : bact_t1
    20 : bact_t2
    21 : bact_t3
    22 : bact_t4
    23 : bact_t5
    24 : bact_log2e
    25 : bact_ln2hi
    26 : bact_ln2lo
    27 : bact_c720
    28 : bact_c120
    29 : bact_c24
    30 : bact_c6
    31 : bact_chalf
    32 : bact_cone
    33 : bact_mneg1
    164-175 : mb_dest<00-11>
    200,201 : mb_prepare<0-1>
    204,205 : mb_g<0-1>
    206,207 : mb_u<0-1>
    204-207 : mb_sources<0-3>
    208,209 : mb_ready<0-1>
    210,211 : mb_targets<0-1>
    212 : mb_index
    213 : mb_token
    214 : mb_layer
    215 : mb_epoch
    216,217 : mb_address<0-1>
    218 : mb_value
    219 : mb_flags
    220 : mb_temp
    222,223 : mb_map<0-1>
    224 : mb_offset
    225 : mb_group
    226 : mb_die
]=]
 local out={};local function e(ins,ctl)out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'end
 local function l(x)out[#out+1]=x..':'end
 local function w(mask)e('NOP',mask..':-:-:-:6')end
 e('LDS mb_flags, [RZ+0xbff8]','3f:-:1:-:2');w('01')
 e('LOP32I.AND mb_flags, mb_flags, 0x200000')
 e('ISETP.EQ.U32.AND P6, PT, mb_flags, RZ, PT');e('@P6 BRA BM_PREP_DONE')
 l('BM_DISTRIBUTED_PREPARE')
 e('LDS.64 mb_prepare0, [RZ+0xbfe8]','--:-:1:-:2');w('01')
 e('LDG.E.CG.128 mb_sources0, [mb_prepare]','--:-:1:-:2');w('01')
 for _,x in ipairs({{'mb_targets0',16},{'mb_ready0',24}})do
  e(string.format('LDG.E.CG.64 %s, [mb_prepare+0x%x]',x[1],x[2]),'--:-:1:-:2');w('01')
 end
 e('LDS mb_token, [RZ+0xbf70]','--:-:1:-:2')
 e('LDS mb_layer, [RZ+0xbff4]','--:-:1:-:2')
 e('LDS mb_die, [RZ+0xbfe0]','--:-:1:-:2');w('01')
 e('IADD mb_epoch, mb_token, 0x1');e('ISCADD mb_epoch, mb_epoch, mb_layer, 0x6')
 local constants=assert(source:match('([^\n]*MOV32I ga_log2e,.-)\n[^\n]*MOV32I ga_c127,'))
 out[#out+1]=constants:gsub('ga_','bact_')
 e('ISCADD mb_index, cta, warp, 0x3')
 e('XMAD mb_index, mb_index, 0x6, mb_die');e('ISCADD mb_index, mb_index, lane, 0x5')
 e('ISETP.GE.U32.AND P6, PT, mb_index, 0x1b80, PT');e('@P6 BRA BM_PREP_QUANT_DONE')
 l('BM_LOCAL_INPUTS')
 e('SHL mb_temp, mb_index, 0x2')
 e('IADD mb_g0.CC, mb_g0, mb_temp');e('IADD.X mb_g1, mb_g1, RZ')
 e('IADD mb_u0.CC, mb_u0, mb_temp');e('IADD.X mb_u1, mb_u1, RZ')
 e('LDG.E.CG bact_gate, [mb_g]','--:-:1:-:1')
 e('LDG.E.CG bact_up, [mb_u]','--:-:1:-:2');w('01')
 local first=assert(source:find('FADD ga_t4, -ga_gate, RZ;',1,true))
 first=assert(source:sub(1,first):match('.*\n()'))
 local _,last=assert(source:find('FMUL ga_value, ga_value, q2moegn_accf;',first,true))
 out[#out+1]=source:sub(first,last):gsub('ga_','bact_'):gsub('q2moegn_accf','bact_up')
 local quant=dofile('balance/../phase/quant_emit.lua')(maps)
 local q=quant('bmq',1,0xbfe8,
   '--:-:-:Y:d MOV %_v0, bact_value;\n--:-:-:Y:d MOV %_index, mb_index;','')
 local cut=assert(q:find('01:-:-:-:6 MOV bmq_peer, RZ;',1,true));q=q:sub(1,cut-1)
 local n;q,n=q:gsub('[^\n]*LDS%.64 bmq_table0, %[%s*RZ%+0xbfe8%];',
   '--:-:-:Y:d MOV bmq_table0, mb_targets0;\n--:-:-:Y:d MOV bmq_table1, mb_targets1;')
 assert(n==1,'quantization target pointer geometry');out[#out+1]=q
 e('NOP','3f:-:-:Y:d')
 e('SHL mb_temp, lane, 0x1')
 e('SHFL.IDX PT, bmq_packed0, bmq_q0, mb_temp, 0x1f','--:-:5:-:2');w('10')
 e('IADD mb_temp, mb_temp, 0x1')
 e('SHFL.IDX PT, bmq_packed1, bmq_q0, mb_temp, 0x1f','--:-:5:-:2');w('10')
 e('LOP32I.AND mb_value, bmq_packed0, 0xffff');e('SHL bmq_packed1, bmq_packed1, 0x10')
 e('LOP.OR mb_value, mb_value, bmq_packed1')
 e('SHFL.IDX PT, bmq_packed0, bmq_s16, 0x0, 0x1f','--:-:5:-:2');w('10')
 e('SHFL.IDX PT, bmq_packed1, bmq_s16, 0x10, 0x1f','--:-:5:-:2');w('10')
 for _,x in ipairs({{16,'bmq_s32'},{17,'bmq_packed0'},{18,'bmq_packed1'},{19,'bmq_scale'}})do
  e(string.format('ISETP.EQ.U32.AND P0, PT, lane, 0x%x, PT',x[1]));e('@P0 MOV mb_value, '..x[2])
 end
 l('BM_PACKET_STAGE')
 e('SHL mb_temp, warp, 0x7');e('ISCADD mb_temp, lane, mb_temp, 0x2');e('IADD mb_temp, mb_temp, 0x2000')
 e('ISETP.LT.U32.AND P2, PT, lane, 0x14, PT')
 e('@P2 STS [mb_temp], mb_value','--:2:-:-:2');w('02')
 l('BM_PREP_QUANT_DONE')
 e('BAR.SYNC 0x0')
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA BM_PREP_FLAGS_SENT')
 e('MOV mb_group, RZ');e('ISETP.LT.U32.AND P2, PT, lane, 0x14, PT')
 for i=0,2 do e(string.format('LDG.E.CG.128 mb_dest%02d, [mb_targets+0x%x]',4*i,16*i),'--:-:1:-:2') end
 w('01')
 l('BM_PACKET_SEND')
 e('ISCADD mb_index, cta, mb_group, 0x3');e('XMAD mb_index, mb_index, 0x6, mb_die')
 e('ISETP.GE.U32.AND P6, PT, mb_index, 0xdc, PT');e('@P6 BRA BM_PACKET_SENT')
 e('SHL mb_temp, mb_group, 0x7');e('ISCADD mb_temp, lane, mb_temp, 0x2');e('IADD mb_temp, mb_temp, 0x2000')
 e('@P2 LDS mb_value, [mb_temp]','--:-:1:-:2');w('01')
 e('SHL mb_offset, mb_index, 0x7');e('ISCADD mb_offset, lane, mb_offset, 0x2')
 for peer=0,5 do
  e(string.format('IADD mb_address0.CC, mb_dest%02d, mb_offset',2*peer))
  e(string.format('IADD.X mb_address1, mb_dest%02d, RZ',2*peer+1))
  e('@P2 STG.E.WT [mb_address], mb_value','--:2:-:-:2');w('02')
 end
 e('IADD mb_group, mb_group, 0x1');e('ISETP.LT.U32.AND P6, PT, mb_group, 0x8, PT')
 e('@P6 BRA BM_PACKET_SEND')
 l('BM_PACKET_SENT');e('MEMBAR.SYS','3f:-:-:Y:d')
 e('LDG.E.CG.64 mb_map0, [mb_prepare+0x20]','--:-:1:-:2');w('01')
 e('SHL mb_temp, lane, 0x3')
 e('IADD mb_map0.CC, mb_map0, mb_temp');e('IADD.X mb_map1, mb_map1, RZ')
 e('ISETP.LT.U32.AND P5, PT, lane, 0x6, PT')
 e('@P5 LDG.E.CG.64 mb_address0, [mb_map]','--:-:1:-:2');w('01')
 e('ISCADD mb_temp, mb_die, cta, 0x4');e('SHL mb_temp, mb_temp, 0x2')
 e('IADD mb_address0.CC, mb_address0, mb_temp');e('IADD.X mb_address1, mb_address1, RZ')
 e('@P5 STG.E.WT [mb_address], mb_epoch','--:2:-:-:2');w('02')
 l('BM_PREP_FLAGS_SENT')
 -- Flags are mirrored into each destination. Poll only local memory;
 -- six warps each acquire one producer die's16 flags, then join locally.
 e('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT');e('@P6 BRA BM_PREP_BARRIER')
 e('LOP32I.AND mb_temp, lane, 0xf');e('ISCADD mb_temp, warp, mb_temp, 0x4');e('SHL mb_temp, mb_temp, 0x2')
 e('IADD mb_address0.CC, mb_ready0, mb_temp');e('IADD.X mb_address1, mb_ready1, RZ')
 l('BM_PREP_WAIT')
 e('LDG.E.CV mb_value, [mb_address]','--:-:1:-:2');w('01')
 e('ISETP.EQ.U32.AND P5, PT, mb_value, mb_epoch, PT')
 e('VOTE.ALL P5, P5');e('@!P5 BRA BM_PREP_WAIT')
 l('BM_PREP_BARRIER')
 e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
 l('BM_PREP_DONE')
 return table.concat(out,'\n')..'\n'
end
