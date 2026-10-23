-- Compact paired MoE columns. Scalar and B4 share the same local arithmetic.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i,j=s:find(a,1,true);assert(i,'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local maps={[=[
    16-47 : pc_weight<00-31>
    48-111 : pc_x<00-63>
    112-115 : pc_dot<0-3>
    116-119 : pc_value<0-3>
    120-123 : pc_coef<0-3>
    124-127 : pc_sum<0-3>
    128,129 : pc_W<0-1>
    130,131 : pc_G<0-1>
    132,133 : pc_ptr<0-1>
    134,135 : pc_other<0-1>
    136-143 : pc_qbase<00-07>
    144-151 : pc_asbase<00-07>
    152-155 : pc_rank<0-3>
    156 : pc_id
    157 : pc_row
    158 : pc_rowlocal
    159 : pc_lane
    160 : pc_d
    161 : pc_format
    162 : pc_record
    163 : pc_recbytes
    164 : pc_scalebytes
    165 : pc_codebytes
    166 : pc_block
    167 : pc_half
    168-175 : pc_codes<0-7>
    176 : pc_hi
    177 : pc_shift
    178 : pc_offset
    179 : pc_tmp
    180 : pc_tmp2
    181 : pc_k
    182 : pc_exp
    185 : pc_scratch
    188-191 : pc_lut<0-3>
    200 : pc_count
    201 : pc_blocks
    202 : pc_channels
    203 : pc_layer
    206 : pc_position
    207 : pc_limit
    208 : pc_descriptor
    210,211 : pc_config<0-1>
    212,213 : pc_gate<0-1>
    214,215 : pc_up<0-1>
    216,217 : pc_targets<0-1>
    218,219 : pc_ready<0-1>
    220,221 : pc_addr<0-1>
    222 : pc_index
    223 : pc_word
    224 : pc_epoch
    225 : pc_iter
    226 : pc_temp
    228,229 : pc_transfer<0-1>
    231 : pc_code
    232 : pc_memoffset
    233 : pc_rowbase
    234 : pc_base_index
    235 : pc_timer
    236 : pc_start
    16 : pcact_gate
    17 : pcact_up
    18 : pcact_value
    19 : pcact_t1
    20 : pcact_t2
    21 : pcact_t3
    22 : pcact_t4
    23 : pcact_t5
    24 : pcact_log2e
    25 : pcact_ln2hi
    26 : pcact_ln2lo
    27 : pcact_c720
    28 : pcact_c120
    29 : pcact_c24
    30 : pcact_c6
    31 : pcact_chalf
    32 : pcact_cone
    33 : pcact_mneg1
]=]}
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
local function addr(dst,base,offset)
 e('IADD '..dst..'0.CC, '..base..'0, '..offset);e('IADD.X '..dst..'1, '..base..'1, RZ')
end
local function wait(mask)e('NOP',mask..':-:-:-:6')end
l('PCOL_SETUP');e('MOV pc_count, 0x1');e('MOV pc_base_index, pidx');e('BRA PCOL_CACHE')
l('PCOL4_SETUP');e('MOV pc_count, 0x4');e('IADD pc_base_index, pidx, 0x1')
l('PCOL_CACHE');wait('3f')
pred('ISETP.LT.U32.AND P4, PT, warp, pc_count, PT');pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
e('IADD pc_temp, pc_base_index, warp');e('SHL pc_temp, pc_temp, 0x7');addr('pc_addr','prog','pc_temp')
e('SHL pc_temp, lane, 0x2');addr('pc_addr','pc_addr','pc_temp');e('MOV pc_word, RZ')
e('@P4 LDG.E.CG pc_word, [pc_addr]','--:-:1:-:2');e('ISCADD pc_memoffset, warp, pc_temp, 0x7');e('IADD pc_memoffset, pc_memoffset, 0x8000')
e('@P5 STS [pc_memoffset], pc_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS pc_blocks, [RZ+0x8020]','--:-:1:-:2');e('LDS pc_layer, [RZ+0x8028]','--:-:1:-:2')
e('LDS.64 pc_ready0, [RZ+0x8040]','--:-:1:-:2');e('LDS pc_epoch, [RZ+0xbf70]','--:-:1:-:2');wait('01')
e('SHL pc_channels, pc_blocks, 0x6');e('XMAD pc_limit, pc_channels, 0xb, RZ')
e('IADD pc_epoch, pc_epoch, 0x1');e('ISCADD pc_epoch, pc_epoch, pc_layer, 0x6')
-- Exact existing SiLU and per-32 activation quantization, without peer copies.
local constants=assert(s:match('([^\n]*MOV32I ga_log2e,.-)\n[^\n]*MOV32I ga_c127,'))
o[#o+1]=constants:gsub('ga_','pcact_')
local start=assert(s:find('FADD ga_t4, -ga_gate, RZ;',1,true));start=(s:sub(1,start-1):match('.*()\n') or 0)+1
local _,finish=assert(s:find('FMUL ga_value, ga_value, q2moegn_accf;',start,true))
local activation=s:sub(start,finish):gsub('ga_','pcact_'):gsub('q2moegn_accf','pcact_up')
local quant=dofile('phase/quant_emit.lua')(maps)
for p=0,3 do
 pred(string.format('ISETP.LE.U32.AND P6, PT, pc_count, 0x%x, PT',p));e('@P6 BRA PCOL_QUANT_DONE_'..p)
 e(string.format('LDS.64 pc_config0, [RZ+0x%x]',0x8018+p*128),'--:-:1:-:2');wait('01')
 e('LDG.E.CG.64 pc_gate0, [pc_config]','--:-:1:-:2');e('LDG.E.CG.64 pc_up0, [pc_config+0x8]','--:-:1:-:2')
 e('IADD pc_targets0.CC, pc_config0, 0x10');e('IADD.X pc_targets1, pc_config1, RZ');wait('01')
 e('ISCADD pc_index, cta, tid, 0x8');l('PCOL_QUANT_LOOP_'..p)
 pred('ISETP.GE.U32.AND P6, PT, pc_index, pc_limit, PT');e('@P6 BRA PCOL_QUANT_DONE_'..p)
 e('SHL pc_temp, pc_index, 0x2');addr('pc_addr','pc_gate','pc_temp')
 e('LDG.E.CG pcact_gate, [pc_addr]','--:2:1:-:2');wait('02');addr('pc_addr','pc_up','pc_temp')
 e('LDG.E.CG pcact_up, [pc_addr]','--:-:1:-:2');wait('01');o[#o+1]=activation
 local tag='pcq'..p
 local q=quant(tag,1,0xbfe8,'--:-:-:-:6 MOV %_v0, pcact_value;\n--:-:-:-:6 MOV %_index, pc_index;',
  '--:-:-:-:6 IADD pc_index, pc_index, 0x1000;\n--:-:-:Y:d BRA PCOL_QUANT_LOOP_'..p..';')
 local n;q,n=q:gsub('[^\n]*LDS%.64 '..tag..'_table0, %[RZ%+0xbfe8%];',
  '--:-:-:-:6 MOV '..tag..'_table0, pc_targets0;\n--:-:-:-:6 MOV '..tag..'_table1, pc_targets1;');assert(n==1)
 q,n=q:gsub(tag..'_peer, 0x6, PT',tag..'_peer, 0x1, PT');assert(n==1)
 o[#o+1]=q;l('PCOL_QUANT_DONE_'..p)
end
e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0');e('SHL pc_temp, cta, 0x2');addr('pc_addr','pc_ready','pc_temp')
pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT');e('@P5 STG.E.WT [pc_addr], pc_epoch','--:2:-:-:2');wait('02')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA PCOL_QUANT_JOIN')
e('LOP32I.AND pc_temp, lane, 0xf');e('SHL pc_temp, pc_temp, 0x2');addr('pc_addr','pc_ready','pc_temp')
l('PCOL_QUANT_WAIT');e('LDG.E.CV pc_word, [pc_addr]','--:-:1:-:2');wait('01')
pred('ISETP.GE.U32.AND P5, PT, pc_word, pc_epoch, PT');pred('VOTE.ALL P5, P5');e('@P5 BRA PCOL_QUANT_JOIN')
e('S2R pc_timer, SR_GLOBALTIMERHI','--:-:6:-:2');e('LDS pc_start, [RZ+0xbf5c]','--:-:1:-:2')
e('IADD pc_timer, pc_timer, -pc_start','21:-:-:-:6');pred('ISETP.GE.U32.AND P5, PT, pc_timer, 0x46, PT')
pred('VOTE.ANY P5, P5');e('@P5 BRA BEXP');e('BRA PCOL_QUANT_WAIT')
l('PCOL_QUANT_JOIN');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
-- Each CTA privately caches four routing lists. Unused positions have invalid IDs.
pred('ISETP.GE.U32.AND P6, PT, warp, 0x4, PT');e('@P6 BRA PCOL_ROUTE_JOIN')
e('SHL pc_descriptor, warp, 0x7');e('IADD pc_descriptor, pc_descriptor, 0x8000')
e('LDS.64 pc_config0, [pc_descriptor+0x18]','--:-:1:-:2');wait('01')
pred('ISETP.LT.U32.AND P4, PT, warp, pc_count, PT')
for _,v in ipairs({'pc_gate0','pc_gate1','pc_up0','pc_up1'})do e('MOV '..v..', RZ')end
for _,v in ipairs({{'pc_gate0',16},{'pc_up0',40},{'pc_addr0',48},{'pc_transfer0',56}})do
 e(string.format('@P4 LDG.E.CG.64 %s, [pc_config+0x%x]',v[1],v[2]),'--:-:1:-:2')
end
wait('01');e('SHL pc_temp, lane, 0x2');addr('pc_addr','pc_addr','pc_temp');addr('pc_transfer','pc_transfer','pc_temp')
e('MOV32I pc_word, 0xffffffff');e('MOV pc_code, RZ');pred('ISETP.LT.U32.AND P5, PT, lane, 0xa, P4')
e('@P5 LDG.E.CG pc_word, [pc_addr]','--:-:1:-:2');e('@P5 LDG.E.CG pc_code, [pc_transfer]','--:-:1:-:2');wait('01')
e('ISCADD pc_memoffset, warp, pc_temp, 0x6');pred('ISETP.LT.U32.AND P5, PT, lane, 0x10, PT')
e('@P5 STS [pc_memoffset+0x8800], pc_word','--:2:-:-:2');e('@P5 STS [pc_memoffset+0x8900], pc_code','--:2:-:-:2')
e('SHL pc_memoffset, warp, 0x5');pred('ISETP.EQ.U32.AND P5, PT, lane, RZ, PT')
e('@P5 STS.64 [pc_memoffset+0x8400], pc_gate0','--:2:-:-:2');e('@P5 STS.64 [pc_memoffset+0x8408], pc_up0','--:2:-:-:2')
e('MOV pc_code, RZ');e('@P4 LDG.E.CG.64 pc_addr0, [pc_config+0x40]','--:-:1:-:2');wait('01')
e('@P4 LDG.E.CG pc_code, [pc_addr]','--:-:1:-:2');e('SHL pc_memoffset, warp, 0x2')
e('@P5 STS [pc_memoffset+0x8a00], pc_code','01:2:-:-:2');wait('02')
l('PCOL_ROUTE_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d')
-- Union records retain per-position rank, so final accumulation remains rank ordered.
e('IMNMX.U32 pc_index, tid, 0x28, PT');e('MOV32I pc_id, 0x200')
for p=0,3 do
 pred(string.format('ISETP.GE.U32.AND P4, PT, pc_index, 0x%x, PT',10*p))
 pred(string.format('ISETP.LT.U32.AND P4, PT, pc_index, 0x%x, P4',10*(p+1)))
 if p==0 then e('MOV pc_temp, pc_index')
 else e(string.format('IADD pc_temp, pc_index, -0x%x',10*p))end
 e('SHL pc_temp, pc_temp, 0x2')
 e(string.format('@P4 LDS pc_id, [pc_temp+0x%x]',0x8800+p*64),'--:-:1:-:2')
end
wait('01')
for j=0,39 do
 e(string.format('LDS pc_word, [RZ+0x%x]',0x8800+(j//10)*64+(j%10)*4),'--:-:1:-:2');wait('01')
 pred(string.format('ISETP.GT.U32.AND P5, PT, pc_index, 0x%x, PT',j))
 pred('ISETP.EQ.U32.AND P5, PT, pc_id, pc_word, P5');e('@P5 MOV32I pc_id, 0xffffffff')
end
for p=0,3 do e('MOV32I pc_rank'..p..', 0xffffffff');e('MOV pc_coef'..p..', RZ')end
for rank=0,9 do for p=0,3 do
 e(string.format('LDS pc_word, [RZ+0x%x]',0x8800+p*64+rank*4),'--:-:1:-:2')
 e(string.format('LDS pc_temp, [RZ+0x%x]',0x8900+p*64+rank*4),'--:-:1:-:2');wait('01')
 pred('ISETP.EQ.U32.AND P5, PT, pc_id, pc_word, PT')
 e(string.format('@P5 MOV pc_rank%d, 0x%x',p,rank));e(string.format('@P5 MOV pc_coef%d, pc_temp',p))
end end
for p=0,3 do
 e(string.format('LDS pc_temp, [RZ+0x%x]',0x8a00+p*4),'--:-:1:-:2');wait('01')
 pred('ISETP.EQ.U32.AND P5, PT, pc_index, 0x28, PT');pred(string.format('ISETP.GT.U32.AND P5, PT, pc_count, 0x%x, P5',p))
 e(string.format('@P5 MOV pc_rank%d, 0xa',p));e(string.format('@P5 MOV pc_coef%d, pc_temp',p))
end
e('SHL pc_memoffset, pc_index, 0x6');pred('ISETP.LE.U32.AND P5, PT, tid, 0x28, PT')
e('@P5 STS [pc_memoffset+0x9000], pc_id','--:2:-:-:2')
for p=0,3 do
 e(string.format('@P5 STS [pc_memoffset+0x%x], pc_rank%d',0x9004+4*p,p),'--:2:-:-:2')
 e(string.format('@P5 STS [pc_memoffset+0x%x], pc_coef%d',0x9014+4*p,p),'--:2:-:-:2')
end
e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS.64 pc_W0, [RZ+0x8008]','--:-:1:-:2');e('LDS.64 pc_G0, [RZ+0x8010]','--:-:1:-:2')
for p=0,3 do
 e(string.format('LDS.64 pc_qbase%02d, [RZ+0x%x]',p*2,0x8400+p*32),'--:-:1:-:2')
 e(string.format('LDS.64 pc_asbase%02d, [RZ+0x%x]',p*2,0x8408+p*32),'--:-:1:-:2')
end
wait('01')
for j,v in ipairs({'0xbfad9881','0xf6eaddcf','0x26190d01','0x71594535'})do e('MOV32I pc_lut'..(j-1)..', '..v)end
e('SHL pc_rowbase, cta, 0x6');e('SHR.U32 pc_rowlocal, tid, 0x2');e('LOP32I.AND pc_lane, tid, 0x3')
e('SHR.U32 pc_block, pc_lane, 0x1');e('LOP32I.AND pc_half, pc_lane, 0x1')
l('PCOL_TILE');pred('ISETP.GE.U32.AND P6, PT, pc_rowbase, 0xa00, PT');e('@P6 BRA POSITION_END')
e('IADD pc_row, pc_rowbase, pc_rowlocal');e('MOV pc_record, RZ')
l('PCOL_RECORD');e('SHL pc_scratch, pc_record, 0x6');e('LDS pc_id, [pc_scratch+0x9000]','--:-:1:-:2')
for p=0,3 do
 e(string.format('LDS pc_rank%d, [pc_scratch+0x%x]',p,0x9004+4*p),'--:-:1:-:2')
 e(string.format('LDS pc_coef%d, [pc_scratch+0x%x]',p,0x9014+4*p),'--:-:1:-:2')
end
wait('01');pred('ISETP.EQ.U32.AND P6, PT, pc_id, -0x1, PT');e('@P6 BRA PCOL_NEXT_RECORD')
e('MOV pc_format, 0x2a');e('MOV pc_exp, pc_id');pred('ISETP.EQ.U32.AND P5, PT, pc_id, 0x200, PT')
e('@P5 LDS pc_format, [RZ+0x8024]','--:-:1:-:2');e('@P5 MOV pc_exp, RZ');wait('01')
e('MOV pc_scalebytes, 0x4');e('MOV pc_codebytes, 0x20')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x2a, PT');e('@P6 MOV pc_scalebytes, 0x2');e('@P6 MOV pc_codebytes, 0x10')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x6, PT');e('@P6 MOV pc_codebytes, 0x28')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x8, PT');e('@P6 MOV pc_codebytes, 0x40')
e('IADD pc_recbytes, pc_scalebytes, pc_codebytes');e('XMAD pc_recbytes, pc_recbytes, 0xa00, RZ')
e('XMAD pc_exp, pc_exp, pc_blocks, pc_block')
-- Some block strides exceed 65535: include the high half in address products.
e('XMAD pc_offset, pc_exp, pc_recbytes, RZ');e('XMAD.PSL pc_offset, pc_exp, pc_recbytes.H1, pc_offset')
e('@!P5 MOV pc_other0, pc_W0');e('@!P5 MOV pc_other1, pc_W1');e('@P5 MOV pc_other0, pc_G0');e('@P5 MOV pc_other1, pc_G1')
addr('pc_other','pc_other','pc_offset');e('XMAD pc_offset, pc_row, pc_scalebytes, RZ')
pred('ISETP.NE.U32.AND P6, PT, pc_format, 0x2a, PT');e('@P6 ISCADD pc_offset, pc_half, pc_offset, 0x1')
addr('pc_ptr','pc_other','pc_offset');pred('ISETP.LT.U32.AND P4, PT, pc_block, pc_blocks, PT')
e('@P4 LDG.E.CI.U16 pc_d, [pc_ptr]','--:2:1:-:2');wait('03')
e('MOV pc_tmp, pc_codebytes');pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x6, PT');e('@P6 MOV pc_tmp, 0x20')
e('XMAD pc_offset, pc_row, pc_tmp, RZ');e('XMAD pc_offset, pc_scalebytes, 0xa00, pc_offset');addr('pc_other','pc_other','pc_offset')
e('MOV pc_shift, 0x4');pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x2a, PT');e('@P6 MOV pc_shift, 0x3')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x8, PT');e('@P6 MOV pc_shift, 0x5')
e('SHL pc_offset, pc_half, pc_shift');addr('pc_ptr','pc_other','pc_offset')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x2a, PT');e('@P6 BRA PCOL_LOAD_Q2')
e('@P4 LDG.E.CI.128 pc_codes0, [pc_ptr]','--:2:1:-:2')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x8, PT');e('@!P6 BRA PCOL_LOADED')
e('@P4 LDG.E.CI.128 pc_codes4, [pc_ptr+0x10]','--:2:1:-:2');e('BRA PCOL_LOADED')
l('PCOL_LOAD_Q2');e('@P4 LDG.E.CI.64 pc_codes0, [pc_ptr]','--:2:1:-:2')
l('PCOL_LOADED');wait('03')
pred('ISETP.EQ.U32.AND P6, PT, pc_format, 0x6, PT');e('@!P6 BRA PCOL_HIGH_DONE')
e('XMAD pc_tmp, pc_row, 0x18, RZ');e('MOV32I pc_offset, 0x14000');e('IADD pc_offset, pc_offset, -pc_tmp')
e('ISCADD pc_offset, pc_half, pc_offset, 0x2');addr('pc_ptr','pc_other','pc_offset')
e('@P4 LDG.E.CI pc_hi, [pc_ptr]','--:2:1:-:2');wait('03');l('PCOL_HIGH_DONE')
for _,entry in ipairs({{42,'Q2'},{2,'Q4'},{6,'Q5'},{8,'Q8'}})do
 pred(string.format('ISETP.EQ.U32.AND P6, PT, pc_format, 0x%x, PT',entry[1]));e('@P6 BRA PCOL_DECODE_'..entry[2])
end
e('BRA PCOL_DECODE_IQ4')
for _,kind in ipairs({'Q2','Q4','Q5','Q8','IQ4'})do
 l('PCOL_DECODE_'..kind)
 for j=0,31 do
  local bits=kind=='Q2' and 2 or kind=='Q8' and 8 or 4;local elements=32//bits
  e(string.format('BFE%s pc_weight%02d, pc_codes%d, 0x%x',kind=='Q8' and '' or '.U32',j,j//elements,bits*256+(j%elements)*bits))
  if kind=='Q2' or kind=='Q4' then e(string.format('IADD pc_weight%02d, pc_weight%02d, -0x%x',j,j,kind=='Q2' and 1 or 8))
  elseif kind=='Q5' then
   e(string.format('BFE.U32 pc_tmp, pc_hi, 0x%x',0x100+j));e(string.format('ISCADD pc_weight%02d, pc_tmp, pc_weight%02d, 0x4',j,j))
   e(string.format('IADD pc_weight%02d, pc_weight%02d, -0x10',j,j))
  elseif kind=='IQ4' then
   pred(string.format('ISETP.LT.U32.AND P6, PT, pc_weight%02d, 0x8, PT',j));e(string.format('LOP32I.AND pc_tmp, pc_weight%02d, 0x7',j))
   e(string.format('@P6 PRMT pc_weight%02d, pc_lut0, pc_tmp, pc_lut1',j));e(string.format('@!P6 PRMT pc_weight%02d, pc_lut2, pc_tmp, pc_lut3',j))
   e(string.format('BFE pc_weight%02d, pc_weight%02d, 0x800',j,j))
  end
 end
 e('BRA PCOL_DECODED')
end
l('PCOL_DECODED');e('F2F.F32.F16 pc_d, pc_d.H0','--:-:6:-:2')
local function vector_address(stem,p)
 e(string.format('IADD pc_ptr0.CC, %s%02d, pc_offset',stem,p*2));e(string.format('IADD.X pc_ptr1, %s%02d, RZ',stem,p*2+1))
end
for p=0,3 do
 pred(string.format('ISETP.NE.U32.AND P%d, PT, pc_rank%d, -0x1, P4',p,p))
 e(string.format('XMAD pc_k, pc_rank%d, pc_channels, RZ',p));e('ISCADD pc_k, pc_lane, pc_k, 0x5')
 e('SHL pc_offset, pc_k, 0x1');vector_address('pc_qbase',p)
 for j=0,3 do e(string.format('@P%d LDG.E.CG.128 pc_x%02d, [pc_ptr+0x%x]',p,p*16+j*4,j*16),'--:2:1:-:2')end
 wait('02')
 e('SHR.U32 pc_offset, pc_k, 0x5');e('SHL pc_offset, pc_offset, 0x2');vector_address('pc_asbase',p)
 e(string.format('@P%d LDG.E.CG pc_sum%d, [pc_ptr]',p,p),'--:2:3:-:2');wait('02')
 e('MOV pc_dot'..p..', RZ');e('MOV pc_value'..p..', RZ')
end
wait('25')
for j=0,31 do for p=0,3 do
 e(string.format('@P%d XMAD.S16.S16 pc_dot%d, pc_weight%02d.reuse, pc_x%02d.H%d, pc_dot%d',p,p,j,p*16+j//2,j%2,p),p==3 and '--:-:-:-:3' or '--:-:-:-:1')
end end
for p=0,3 do e(string.format('I2F.F32.S32 pc_dot%d, pc_dot%d',p,p),'--:-:3:-:2')end
wait('04')
for p=0,3 do
 e(string.format('@P%d FMUL pc_dot%d, pc_d, pc_dot%d',p,p,p));e(string.format('@P%d FFMA pc_value%d, pc_sum%d, pc_dot%d, RZ',p,p,p,p))
end
for _,distance in ipairs({2,1})do
 for p=0,3 do e(string.format('SHFL.BFLY PT, pc_dot%d, pc_value%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 wait('10');for p=0,3 do e(string.format('FADD pc_value%d, pc_value%d, pc_dot%d',p,p,p))end
end
for p=0,3 do
 pred(string.format('ISETP.EQ.U32.AND P5, PT, pc_lane, RZ, P%d',p));e(string.format('FMUL pc_value%d, pc_value%d, pc_coef%d',p,p,p))
 e(string.format('SHL pc_scratch, pc_rank%d, 0x8',p));e(string.format('IADD pc_scratch, pc_scratch, 0x%x',p*11*256));e('ISCADD pc_scratch, pc_rowlocal, pc_scratch, 0x2')
 e(string.format('@P5 STS [pc_scratch], pc_value%d',p),'--:2:-:-:2');wait('02')
end
l('PCOL_NEXT_RECORD');e('IADD pc_record, pc_record, 0x1');pred('ISETP.LT.U32.AND P6, PT, pc_record, 0x29, PT');e('@P6 BRA PCOL_RECORD')
e('BAR.SYNC 0x0','3f:-:-:Y:d')
for p=0,3 do e('MOV pc_sum'..p..', RZ')end
e('SHL pc_scratch, pc_rowlocal, 0x2')
for rank=0,10 do
 for p=0,3 do e(string.format('LDS pc_value%d, [pc_scratch+0x%x]',p,(p*11+rank)*256),'--:-:1:-:2')end
 wait('01');for p=0,3 do e(string.format('FADD pc_sum%d, pc_sum%d, pc_value%d',p,p,p))end
end
for p=0,3 do
 pred(string.format('ISETP.GT.U32.AND P5, PT, pc_count, 0x%x, PT',p));pred('ISETP.EQ.U32.AND P5, PT, pc_lane, RZ, P5')
 e(string.format('@P5 STS [pc_scratch+0x%x], pc_sum%d',0x3000+p*256,p),'--:2:-:-:2')
 e(string.format('LDS.64 pc_ptr0, [RZ+0x%x]',0x8038+p*128),'--:-:1:-:2');e('SHL pc_offset, pc_row, 0x2');wait('01');addr('pc_ptr','pc_ptr','pc_offset')
 e(string.format('@P5 STG.E [pc_ptr], pc_sum%d',p),'--:2:-:-:2');wait('02')
end
e('BAR.SYNC 0x0','3f:-:-:Y:d')
o[#o+1]=dofile('unique/pair_transport.lua')(maps)
l('PCOL_PUBLISH_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d');e('IADD pc_rowbase, pc_rowbase, 0x400');e('BRA PCOL_TILE')
replace('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
replace('ISETP.GE.U32.AND P6, PT, op, 0x70, PT;','ISETP.GE.U32.AND P6, PT, op, 0x71, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA PCOL_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
