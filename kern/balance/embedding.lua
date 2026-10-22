-- Replace the production embedding body in the balanced image only. One
-- CTA/warp owns each32-column packed group; all six dies publish disjoint
-- columns directly to the six residuals. Never decode unowned columns.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local map=[=[
    16,17 : be_weight<0-1>
    18,19 : be_ids<0-1>
    20,21 : be_imap<0-1>
    22,23 : be_image<0-1>
    24,25 : be_table<0-1>
    26,27 : be_pointer<0-1>
    28,29 : be_target<0-1>
    32-35 : be_pack<0-3>
    36 : be_position
    37 : be_first
    38 : be_stride
    39 : be_dies
    40 : be_token
    41 : be_row
    42 : be_index
    43 : be_offset
    44 : be_temp
    45 : be_shift
    46 : be_bits
    47 : be_code
    48 : be_high
    49 : be_scale
    50 : be_d
    51 : be_value
    52 : be_peer
]=]
local at=assert(s:find('</REGISTER_MAPPING>',1,true))
s=s:sub(1,at-1)..map..s:sub(at)
local out={};local function emit(ins,ctl)
 out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'
end
local function label(name)out[#out+1]=name..':'end
local function wait(mask)emit('NOP',mask..':-:-:-:6')end
label('embrow_SETUP')
emit('MOV vid, cta');emit('MOV vtid, tid')
emit('ISETP.GE.U32.AND P6, PT, vid, units, PT');emit('@P6 BRA POSITION_END')
emit('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');emit('@P6 BRA POSITION_END')
label('embrow_BEGIN')
for _,x in ipairs({{'be_weight0',0xbf88},{'be_ids0',0xbfa0},{'be_imap0',0xbfa8},
                  {'be_image0',0xbfb0},{'be_table0',0xbfb8}})do
 emit(string.format('LDS.64 %s, [RZ+0x%x]',x[1],x[2]),'--:-:1:-:2');wait('01')
end
for _,x in ipairs({{'be_position',0xbf98},{'be_first',0xbfc0},
                  {'be_stride',0xbfc4},{'be_dies',0xbfcc}})do
 emit(string.format('LDS %s, [RZ+0x%x]',x[1],x[2]),'--:-:1:-:2');wait('01')
end
emit('ISCADD be_index, vid, lane, 0x5');emit('IADD be_index, be_index, be_first')
emit('SHL be_offset, be_position, 0x2')
emit('IADD be_pointer0.CC, be_ids0, be_offset');emit('IADD.X be_pointer1, be_ids1, RZ')
emit('LDG.E.CG be_token, [be_pointer]','--:-:1:-:2');wait('01')
emit('MOV32I be_row, 0xffffffff')
emit('LOP.OR be_temp, be_imap0, be_imap1')
emit('ISETP.NE.U32.AND P5, PT, be_temp, RZ, PT')
emit('IADD be_pointer0.CC, be_imap0, be_offset');emit('IADD.X be_pointer1, be_imap1, RZ')
emit('@P5 LDG.E.CG be_row, [be_pointer]','--:-:1:-:2');wait('01')
emit('IADD be_temp, be_row, 0x1')
emit('ISETP.NE.U32.AND P5, PT, be_temp, RZ, PT');emit('@P5 BRA BALANCED_EMBED_IMAGE')
emit('XMAD be_offset, be_token, be_stride, RZ')
emit('XMAD.PSL be_offset, be_token.H1, be_stride, be_offset')
emit('ISCADD be_offset, vid, be_offset, 0x4')
emit('IADD be_pointer0.CC, be_weight0, be_offset');emit('IADD.X be_pointer1, be_weight1, RZ')
emit('LDG.E.CI.128 be_pack0, [be_pointer]','--:-:1:-:2');wait('01')
emit('ISETP.GE.U32.AND P0, PT, lane, 0x10, PT')
emit('SEL be_bits, be_pack1, be_pack0, P0')
emit('LOP32I.AND be_shift, lane, 0xf');emit('SHL be_shift, be_shift, 0x1')
emit('SHR.U32 be_code, be_bits, be_shift');emit('LOP32I.AND be_code, be_code, 0x3')
emit('SHR.U32 be_high, be_pack2, lane');emit('LOP32I.AND be_high, be_high, 0x1')
emit('ISETP.EQ.U32.AND P1, PT, be_high, RZ, PT');emit('@P1 IADD be_code, be_code, -0x4')
emit('LOP32I.AND be_shift, lane, 0x10');emit('SHR.U32 be_shift, be_shift, 0x1')
emit('SHR.U32 be_scale, be_pack3, be_shift');emit('LOP32I.AND be_scale, be_scale, 0xff')
emit('IADD be_scale, be_scale, -0x20')
emit('I2F.F32.S32 be_scale, be_scale','--:-:1:-:2');wait('01')
emit('I2F.F32.S32 be_code, be_code','--:-:1:-:2');wait('01')
emit('SHR.U32 be_d, be_pack3, 0x10')
emit('F2F.F32.F16 be_d, be_d.H0','--:-:1:-:2');wait('01')
emit('FMUL be_value, be_d, be_scale');emit('FMUL be_value, be_value, be_code')
emit('BRA BALANCED_EMBED_PUBLISH')
label('BALANCED_EMBED_IMAGE')
emit('XMAD be_offset, be_row, 0xa00, RZ')
emit('XMAD.PSL be_offset, be_row.H1, 0xa00, be_offset')
emit('IADD be_offset, be_offset, be_index');emit('SHL be_offset, be_offset, 0x1')
emit('IADD be_pointer0.CC, be_image0, be_offset');emit('IADD.X be_pointer1, be_image1, RZ')
emit('LDG.E.CG.U16 be_d, [be_pointer]','--:-:1:-:2');wait('01')
emit('F2F.F32.F16 be_value, be_d.H0','--:-:1:-:2');wait('01')
label('BALANCED_EMBED_PUBLISH')
emit('SHL be_offset, be_index, 0x2');emit('MOV be_peer, RZ')
label('BALANCED_EMBED_PEER')
emit('LDG.E.CG.64 be_target0, [be_table]','--:-:1:-:2');wait('01')
emit('IADD be_target0.CC, be_target0, be_offset');emit('IADD.X be_target1, be_target1, RZ')
for _,offset in ipairs({0,0x2800,0x5000,0x7800})do
 emit(string.format('STG.E.WT [be_target+0x%x], be_value',offset),'--:2:-:-:2')
end
wait('02')
emit('IADD be_table0.CC, be_table0, 0x8');emit('IADD.X be_table1, be_table1, RZ')
emit('IADD be_peer, be_peer, 0x1')
emit('ISETP.LT.U32.AND P0, PT, be_peer, be_dies, PT');emit('@P0 BRA BALANCED_EMBED_PEER')
label('embrow_END');emit('BRA POSITION_END')
local a,b=assert(s:find('embrow_SETUP:.-\nplegate_SETUP:'))
s=s:sub(1,a-1)..table.concat(out,'\n')..'\n\nplegate_SETUP:'..s:sub(b+1)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
