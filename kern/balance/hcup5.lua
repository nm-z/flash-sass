-- Four/five-position GR up projection with one weight stream. Preserve the native
-- SiLU, eight-lane dot association, gate/mix, quantization and peer publication.
-- Matrix output ownership matches hcquant's32-column (128 interleaved row) CTA.
-- Five warps then mix/quantize one position each, computing each column's four
-- gates once instead of duplicating each gate across eight dot-product lanes.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local positions=tonumber(arg[3] or '5')
assert(positions==4 or positions==5,'native GR-up width must be four or five')
local function rep(text,old,new)
 local p,q=assert(text:find(old,1,true));assert(not text:find(old,q+1,true),'ambiguous '..old)
 return text:sub(1,p-1)..new..text:sub(q+1)
end
local a=assert(s:find('hcquant_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
local native=s:sub(a,b-1)
local maps={}
maps[#maps+1]=[=[
    16-20 : s5_raw<0-4>
    21-25 : s5_scaled<0-4>
    32-36 : s5_t1_<0-4>
    37-41 : s5_t2_<0-4>
    42-46 : s5_t3_<0-4>
    47-51 : s5_t4_<0-4>
    52-56 : s5_t5_<0-4>
    80-89 : s5_src<00-09>
    90-99 : s5_dst<00-09>
    100 : s5_tn
    101 : s5_ix
    102 : s5_sh
    16-55 : h5_act<00-39>
    160-164 : h5_acc<0-4>
    168-171 : h5_packed<0-3>
    177-184 : h5_float<0-7>
    186-190 : h5_shuffle<0-4>
    192 : h5_rows
    193 : h5_cols
    194 : h5_column_start
    195 : h5_limit
    196 : h5_row
    197 : h5_weight_row
    198 : h5_dot_lane
    199 : h5_group
    200,201 : h5_weights<0-1>
    202,203 : h5_wp<0-1>
    204 : h5_tile
    205 : h5_xs
    206 : h5_tmp
    210-219 : h5_out<00-09>
    240 : h5_position
    241 : h5_sap
    242,243 : h5_descriptor<0-1>
    244 : h5_offset
    245 : h5_word
    246 : h5_tmp2
    237 : h5_wiremode
    16-19 : g5_up<0-3>
    20-23 : g5_x<0-3>
    24-27 : g5_v<0-3>
    32-35 : g5_t1_<0-3>
    36-39 : g5_t2_<0-3>
    40-43 : g5_t3_<0-3>
    44-47 : g5_t4_<0-3>
    56,57 : g5_yp<0-1>
    58,59 : g5_xp<0-1>
    60,61 : g5_fptab<0-1>
    62 : g5_xoff
    63 : g5_fpoff
    64,65 : g5_dst<0-1>
    66 : g5_macc
    67 : g5_tmp
    68,69 : g5_address<0-1>
    70 : g5_col
    71 : g5_index
    72 : g5_stride
    16-20 : p5_value<0-4>
    21-25 : p5_code<0-4>
    26-30 : p5_s16<0-4>
    31-35 : p5_s32<0-4>
    36-40 : p5_scale<0-4>
    136 : p5_quant_peer
    137 : p5_float_peer
    138 : p5_xoff
    139 : p5_index
    140 : p5_qoff
    141 : p5_fpoff
    142 : p5_off16
    143 : p5_off32
    144 : p5_temp
    146,147 : p5_fp_table<0-1>
    148,149 : p5_q_table<0-1>
    150 : p5_data
    151 : p5_aux
]=]
for p=0,positions-1 do
 for i,name in ipairs({'fp','qp','s32p','s16p','ap'})do
  local r=80+10*p+2*(i-1)
  maps[#maps+1]=string.format('    %d,%d : p5_%s%d<0-1>\n',r,r+1,name,p)
 end
end
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)out[#out+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
l('hcup5_SETUP');e('NOP','3f:-:-:Y:d')
e('LDS h5_wiremode, [RZ+0xbf98]','--:-:1:-:2');e('NOP','01:-:-:-:6')
-- Five warps fetch five complete descriptors concurrently. All following
-- stages read this immutable shared copy; the outer framework bank is intact.
pred(string.format('ISETP.LT.U32.AND P5, PT, warp, 0x%x, PT',positions))
e('SHL h5_sap, lane, 0x2')
e('IADD h5_tmp2, pidx, 0x1');e('IADD h5_tmp2, h5_tmp2, warp');e('SHL h5_tmp2, h5_tmp2, 0x7')
e('IADD h5_descriptor0.CC, prog0, h5_tmp2');e('IADD.X h5_descriptor1, prog1, RZ')
e('IADD h5_descriptor0.CC, h5_descriptor0, h5_sap');e('IADD.X h5_descriptor1, h5_descriptor1, RZ')
e('@P5 LDG.E.CG h5_word, [h5_descriptor]','--:-:1:-:2')
e('ISCADD h5_offset, warp, h5_sap, 0x7');e('IADD h5_offset, h5_offset, 0x8000')
e('@P5 STS [h5_offset], h5_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
for line in native:gmatch('[^\n]+')do
 if line:match('MOV32I hcquant_[%w]+,') and not line:find('hcquant_xl',1,true) and
    not line:find('hcquant_rowstep',1,true) and not line:find('hcquant_stepw',1,true) then
  local name=line:match('MOV32I (hcquant_[%w]+),')
  if name:match('hcquant_c') or name:match('hcquant_ln2') or name=='hcquant_log2e' or name=='hcquant_mneg1' then out[#out+1]=line end
 end
end
for p=0,positions-1 do
 e(string.format('LDS.64 s5_src%02d, [RZ+0x%x]',p*2,0x8000+p*128+16),'--:-:1:-:2')
 e(string.format('LDS.64 s5_dst%02d, [RZ+0x%x]',p*2,0x8000+p*128+72),'--:-:1:-:2')
end
e('LDS s5_tn, [RZ+0x8050]','--:-:1:-:2')
e('SHL h5_offset, tid, 0x2')
e('NOP','01:-:-:-:6')
for p=0,positions-1 do for _,name in ipairs({'src','dst'})do
 e(string.format('IADD s5_%s%02d.CC, s5_%s%02d, h5_offset',name,p*2,name,p*2))
 e(string.format('IADD.X s5_%s%02d, s5_%s%02d, RZ',name,p*2+1,name,p*2+1))
end end
e('MOV s5_ix, tid');pred('ISETP.EQ.U32.AND P6, PT, cta, RZ, PT')
l('H5_SILU')
for p=0,positions-1 do e(string.format('LDG.E.CG s5_raw%d, [s5_src%02d]',p,p*2),'--:6:1:-:2')end
e('NOP','21:-:-:-:6')
local silu_first=assert(native:find('FMUL hcquant_gg, hcquant_su0, 0.25;',1,true))
silu_first=(native:sub(1,silu_first-1):match('.*()\n') or 0)+1
local _,silu_last=assert(native:find('FMUL hcquant_sy0, hcquant_gg, hcquant_ft5;',silu_first,true))
local silu_ops=0
for line in native:sub(silu_first,silu_last):gmatch('[^\n]+')do
 local ins=assert(line:match('^%S+%s+(.+);$'));silu_ops=silu_ops+1
 local variable=ins:match('^F2I%.') or ins:match('^MUFU%.')
 for p=0,positions-1 do
  local x=ins:gsub('hcquant_su0','s5_raw'..p):gsub('hcquant_sy0','s5_raw'..p):gsub('hcquant_gg','s5_scaled'..p)
  x=x:gsub('hcquant_ft([1-5])',function(k)return 's5_t'..k..'_'..p end)
  e(x,variable and '--:-:3:-:2' or (p==positions-1 and '--:-:-:-:2' or '--:-:-:-:1'))
 end
 if variable then e('NOP','04:-:-:-:6')end
end
assert(silu_ops==25,'native SiLU polynomial shape changed '..silu_ops)
e('SHL s5_sh, s5_ix, 0x2');e('IADD s5_sh, s5_sh, 0x2000')
for p=0,positions-1 do
 e(string.format('STS [s5_sh+0x%x], s5_raw%d',p*0x600,p),'--:2:-:-:2')
 e(string.format('@P6 STG.E [s5_dst%02d], s5_raw%d',p*2,p),'--:2:-:-:2')
end
e('NOP','02:-:-:-:6')
pred('ISETP.GE.U32.AND P0, PT, s5_ix, 0x100, PT');e('@P0 BRA H5_SILU_DONE','--:-:-:Y:d')
for p=0,positions-1 do for _,name in ipairs({'src','dst'})do
 e(string.format('IADD s5_%s%02d.CC, s5_%s%02d, 0x400',name,p*2,name,p*2))
 e(string.format('IADD.X s5_%s%02d, s5_%s%02d, RZ',name,p*2+1,name,p*2+1))
end end
e('IADD s5_ix, s5_ix, 0x100')
pred('ISETP.LT.U32.AND P0, PT, s5_ix, 0x140, PT');e('@P0 BRA H5_SILU','--:-:-:Y:d')
-- The optional four injection values are copied without SiLU. The cutoff
-- above is warp-uniform; this final copy only predicates its memory accesses.
pred('ISETP.LT.U32.AND P4, PT, s5_ix, s5_tn, PT')
pred('PSETP.AND.AND P5, PT, P4, P6, PT')
for p=0,positions-1 do e(string.format('@P4 LDG.E.CG s5_raw%d, [s5_src%02d]',p,p*2),'--:-:1:-:2')end
e('SHL s5_sh, s5_ix, 0x2');e('IADD s5_sh, s5_sh, 0x2000')
e('NOP','01:-:-:-:6')
for p=0,positions-1 do
 e(string.format('@P4 STS [s5_sh+0x%x], s5_raw%d',p*0x600,p),'--:2:-:-:2')
 e(string.format('@P5 STG.E [s5_dst%02d], s5_raw%d',p*2,p),'--:2:-:-:2')
end
l('H5_SILU_DONE');e('BAR.SYNC 0x0','3f:-:-:Y:d')
e('LDS h5_rows, [RZ+0x8020]','--:-:1:-:2')
e('LDS.64 h5_weights0, [RZ+0x8008]','--:-:1:-:2')
for p=0,positions-1 do e(string.format('LDS.64 h5_out%02d, [RZ+0x%x]',p*2,0x8000+p*128+24),'--:-:1:-:2')end
e('SHR.U32 h5_cols, h5_rows, 0x2','01:-:-:-:6')
e('SHL h5_column_start, cta, 0x5')
pred('ISETP.GE.U32.AND P6, PT, h5_column_start, h5_cols, PT')
e('@P6 BRA H5_MATRIX_DONE','--:-:-:Y:d')
e('IADD h5_limit, h5_column_start, 0x20');e('IMNMX.U32 h5_limit, h5_limit, h5_cols, PT')
e('SHL h5_limit, h5_limit, 0x2');e('SHR.U32 h5_group, tid, 0x3')
e('ISCADD h5_row, h5_column_start, h5_group, 0x2')
e('LOP32I.AND h5_dot_lane, tid, 0x7')
l('H5_MATRIX_ROW')
e('SHR.U32 h5_tmp, h5_row, 0x2');e('LOP32I.AND h5_weight_row, h5_row, 0x3')
e('XMAD h5_weight_row, h5_weight_row, h5_cols, h5_tmp','--:-:-:Y:d')
e('SHL h5_tmp, h5_weight_row, 0x7');e('ISCADD h5_tmp, h5_weight_row, h5_tmp, 0x9')
e('ISCADD h5_tmp, h5_dot_lane, h5_tmp, 0x4')
e('IADD h5_wp0.CC, h5_weights0, h5_tmp');e('IADD.X h5_wp1, h5_weights1, RZ')
e('SHL h5_xs, h5_dot_lane, 0x5');e('IADD h5_xs, h5_xs, 0x2000')
for p=0,positions-1 do e('MOV h5_acc'..p..', RZ','--:-:-:-:1')end
e('MOV h5_tile, RZ')
l('H5_MATRIX_K')
e('LDG.E.CI.128 h5_packed0, [h5_wp]','--:2:1:-:2')
for p=0,positions-1 do for half=0,1 do
 e(string.format('LDS.128 h5_act%02d, [h5_xs+0x%x]',p*8+half*4,p*0x600+half*16),'--:-:3:-:2')
end end
e('NOP','05:-:-:-:6')
for j=0,7 do
 if j%2==0 then e(string.format('SHL h5_float%d, h5_packed%d, 0x10',j,j//2),'--:-:-:-:1')
 else e(string.format('LOP32I.AND h5_float%d, h5_packed%d, 0xffff0000',j,j//2),'--:-:-:-:1')end
end
for j=0,7 do for p=0,positions-1 do
 e(string.format('FFMA h5_acc%d, h5_float%d.reuse, h5_act%02d, h5_acc%d',p,j,p*8+j,p),
   p==positions-1 and '--:-:-:-:2' or '--:-:-:-:1')
end end
e('IADD h5_wp0.CC, h5_wp0, 0x80','02:-:-:-:6');e('IADD.X h5_wp1, h5_wp1, RZ')
e('IADD h5_xs, h5_xs, 0x100');e('IADD h5_tile, h5_tile, 0x1')
pred('ISETP.LT.U32.AND P6, PT, h5_tile, 0x5, PT');e('@P6 BRA H5_MATRIX_K','--:-:-:Y:d')
for _,distance in ipairs({4,2,1})do
 for p=0,positions-1 do e(string.format('SHFL.BFLY PT, h5_shuffle%d, h5_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 e('NOP','10:-:-:-:6')
 for p=0,positions-1 do e(string.format('FADD h5_acc%d, h5_acc%d, h5_shuffle%d',p,p,p),p==positions-1 and '--:-:-:-:6' or '--:-:-:-:1')end
end
e('SHL h5_offset, h5_weight_row, 0x2');pred('ISETP.EQ.U32.AND P0, PT, h5_dot_lane, RZ, PT')
for p=0,positions-1 do
 e(string.format('IADD h5_descriptor0.CC, h5_out%02d, h5_offset',p*2))
 e(string.format('IADD.X h5_descriptor1, h5_out%02d, RZ',p*2+1))
 e(string.format('@P0 STG.E [h5_descriptor], h5_acc%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
end
e('IADD h5_row, h5_row, 0x20');pred('ISETP.LT.U32.AND P6, PT, h5_row, h5_limit, PT')
e('@P6 BRA H5_MATRIX_ROW','--:-:-:Y:d')
l('H5_MATRIX_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
pred(string.format('ISETP.GE.U32.AND P6, PT, warp, 0x%x, PT',positions));e('@P6 BRA H5_PUBLISH_JOIN','--:-:-:Y:d')
pred('ISETP.GE.U32.AND P6, PT, h5_column_start, h5_cols, PT');e('@P6 BRA H5_PUBLISH_JOIN','--:-:-:Y:d')
e('ISCADD h5_sap, warp, RZ, 0x7');e('IADD h5_sap, h5_sap, 0x8000')
e('LDS.64 g5_yp0, [h5_sap+0x18]','--:-:1:-:2')
e('LDS.64 g5_xp0, [h5_sap+0x40]','--:-:1:-:2')
e('LDS.64 g5_fptab0, [h5_sap+0x38]','--:-:1:-:2')
e('LDS g5_xoff, [h5_sap+0x30]','--:-:1:-:2')
e('IADD g5_col, h5_column_start, lane')
e('SHL g5_tmp, g5_col, 0x2')
e('IADD g5_fpoff, g5_tmp, g5_xoff','01:-:-:-:6')
e('LOP32I.AND g5_index, g5_xoff, 0xffff');e('SHR.U32 g5_index, g5_index, 0x2')
e('IADD g5_index, g5_index, g5_col')
e('IADD g5_yp0.CC, g5_yp0, g5_tmp');e('IADD.X g5_yp1, g5_yp1, RZ')
e('IADD g5_xp0.CC, g5_xp0, g5_tmp');e('IADD.X g5_xp1, g5_xp1, RZ')
e('SHL g5_stride, h5_cols, 0x2')
for j=0,3 do
 e(string.format('LDG.E.CG g5_up%d, [g5_yp]',j),'--:2:1:-:2')
 e(string.format('LDG.E.CG g5_x%d, [g5_xp+0x%x]',j,j*0x2800),'--:-:1:-:2')
 e('NOP','02:-:-:-:6')
 if j<3 then e('IADD g5_yp0.CC, g5_yp0, g5_stride');e('IADD.X g5_yp1, g5_yp1, RZ')end
end
e('NOP','01:-:-:-:6')
local first=assert(native:find('FADD hcquant_ft4, -hcquant_accf, RZ;',1,true))
first=(native:sub(1,first-1):match('.*()\n') or 0)+1
local _,last=assert(native:find('FFMA hcquant_gg, hcquant_gg, hcquant_ft1, hcquant_gg;',first,true))
local n=0
for line in native:sub(first,last):gmatch('[^\n]+')do
 local ins=assert(line:match('^%S+%s+(.+);$'));n=n+1
 local variable=ins:match('^F2I%.') or ins:match('^MUFU%.')
 for j=0,3 do
  local x=ins:gsub('hcquant_accf','g5_up'..j):gsub('hcquant_gg','g5_v'..j)
  x=x:gsub('hcquant_ft([1-4])',function(k)return 'g5_t'..k..'_'..j end)
  e(x,variable and '--:-:3:-:2' or '--:-:-:-:2')
 end
 if variable then e('NOP','04:-:-:-:6')end
end
assert(n==23,'native gate polynomial shape changed '..n)
e('FMUL g5_macc, g5_v0, g5_x0')
for j=1,3 do e(string.format('FFMA g5_macc, g5_v%d, g5_x%d, g5_macc',j,j))end
e('FMUL g5_macc, g5_macc, 0.25')
pred('ISETP.NE.U32.AND P6, PT, h5_wiremode, RZ, PT');e('@P6 BRA H5_FP_RECORD','--:-:-:Y:d')
local quant=dofile('balance/../phase/quant_emit.lua')(maps)
local q=quant('h5q',1,0xbfd8,
 '--:-:-:-:6 MOV %_v0, g5_macc;\n--:-:-:-:6 MOV %_index, g5_index;',
 '--:-:-:-:6 BRA H5_PUBLISH_JOIN;')
local send=assert(q:find('01:-:-:-:6 MOV h5q_peer, RZ;',1,true))
q=q:sub(1,send-1)
local count;q,count=q:gsub('[^\n]*LDS%.64 h5q_table0, %[RZ%+0xbfd8%];\n','');assert(count==1)
out[#out+1]=q
-- Each producing warp writes one complete512-byte record. The padding keeps
-- all fields naturally aligned; the values/quantization are unchanged.
e('SHL h5_offset, warp, 0x9');e('IADD h5_offset, h5_offset, 0x5000')
e('ISCADD h5_sap, lane, h5_offset, 0x2')
e('STS [h5_sap], h5q_v0','--:2:-:-:2')
e('STS [h5_sap+0x80], h5q_q0','--:2:-:-:2');e('NOP','02:-:-:-:6')
e('SHR.U32 h5_tmp2, lane, 0x4');e('ISCADD h5_sap, h5_tmp2, h5_offset, 0x2')
e('@P1 STS [h5_sap+0x100], h5q_s16','--:2:-:-:2')
e('@P0 STS [h5_offset+0x108], h5q_s32','--:2:-:-:2')
e('@P0 STS [h5_offset+0x10c], h5q_scale','--:2:-:-:2');e('NOP','02:-:-:-:6')
e('BRA H5_PUBLISH_JOIN','--:-:-:Y:d')
l('H5_FP_RECORD')
e('SHL h5_offset, warp, 0x9');e('IADD h5_offset, h5_offset, 0x5000')
e('ISCADD h5_sap, lane, h5_offset, 0x2')
e('STS [h5_sap], g5_macc','--:2:-:-:2');e('NOP','02:-:-:-:6')
l('H5_PUBLISH_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d')
pred('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT');e('@P6 BRA H5_PUBLISH_DONE','--:-:-:Y:d')
pred('ISETP.GE.U32.AND P6, PT, h5_column_start, h5_cols, PT');e('@P6 BRA H5_PUBLISH_DONE','--:-:-:Y:d')
e('LDS p5_xoff, [RZ+0x8030]','--:-:3:-:2')
e('SHL p5_float_peer, warp, 0x3');e('SHL p5_quant_peer, warp, 0x5')
e('IADD p5_index, h5_column_start, lane');e('SHL p5_fpoff, p5_index, 0x2')
e('IADD p5_fpoff, p5_fpoff, p5_xoff','04:-:-:-:6')
e('LOP32I.AND p5_temp, p5_xoff, 0xffff');e('SHR.U32 p5_temp, p5_temp, 0x2')
e('IADD p5_index, p5_index, p5_temp');e('SHL p5_qoff, p5_index, 0x1')
e('SHR.U32 p5_off16, p5_index, 0x4');e('SHL p5_off16, p5_off16, 0x2')
e('SHR.U32 p5_off32, p5_index, 0x5');e('SHL p5_off32, p5_off32, 0x2')
pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT')
e('LOP32I.AND p5_temp, lane, 0xf');pred('ISETP.EQ.U32.AND P1, PT, p5_temp, RZ, PT')
pred('ISETP.EQ.U32.AND P5, PT, h5_wiremode, RZ, PT')
pred('PSETP.AND.AND P0, PT, P0, P5, PT');pred('PSETP.AND.AND P1, PT, P1, P5, PT')
for p=0,positions-1 do
 e(string.format('LDS.64 p5_fp_table0, [RZ+0x%x]',0x8000+p*128+56),'--:-:3:-:2')
 e(string.format('LDS.64 p5_q_table0, [RZ+0x%x]',0x8000+p*128+88),'--:-:3:-:2')
 e('IADD p5_fp_table0.CC, p5_fp_table0, p5_float_peer','04:-:-:-:6')
 e('IADD.X p5_fp_table1, p5_fp_table1, RZ')
 e('IADD p5_q_table0.CC, p5_q_table0, p5_quant_peer');e('IADD.X p5_q_table1, p5_q_table1, RZ')
 e(string.format('LDG.E.CG.64 p5_fp%d0, [p5_fp_table]',p),'--:4:1:-:2')
 for i,name in ipairs({'qp','s32p','s16p','ap'})do
  e(string.format('@P5 LDG.E.CG.64 p5_%s%d0, [p5_q_table+0x%x]',name,p,(i-1)*8),'--:4:1:-:2')
 end
 e('NOP','08:-:-:-:6') -- pointer-table operands may now be reused
end
for p=0,positions-1 do
 e(string.format('MOV32I h5_offset, 0x%x',0x5000+p*512))
 e('ISCADD p5_data, lane, h5_offset, 0x2')
 e('SHR.U32 p5_aux, lane, 0x4');e('ISCADD p5_aux, p5_aux, h5_offset, 0x2')
 e(string.format('LDS p5_value%d, [p5_data]',p),'--:4:3:-:2')
 e(string.format('@P5 LDS p5_code%d, [p5_data+0x80]',p),'--:4:3:-:2')
 e(string.format('@P5 LDS p5_s16%d, [p5_aux+0x100]',p),'--:4:3:-:2')
 e(string.format('@P5 LDS p5_s32%d, [h5_offset+0x108]',p),'--:4:3:-:2')
 e(string.format('@P5 LDS p5_scale%d, [h5_offset+0x10c]',p),'--:4:3:-:2')
 e('NOP','08:-:-:-:6')
end
e('NOP','05:-:-:-:6')
for p=0,positions-1 do
 for _,field in ipairs({{'fp','p5_fpoff'},{'qp','p5_qoff'},{'s32p','p5_off32'},{'s16p','p5_off16'},{'ap','p5_off32'}})do
  e(string.format('IADD p5_%s%d0.CC, p5_%s%d0, %s',field[1],p,field[1],p,field[2]))
  e(string.format('IADD.X p5_%s%d1, p5_%s%d1, RZ',field[1],p,field[1],p))
 end
 -- No per-position read wait: addresses and payload values stay in distinct
 -- register sets until the entire five-position publication burst retires.
 e(string.format('STG.E.WT [p5_fp%d], p5_value%d',p,p),'--:2:-:-:2')
 e(string.format('@P5 STG.E.WT.U16 [p5_qp%d], p5_code%d',p,p),'--:2:-:-:2')
 e(string.format('@P1 STG.E.WT [p5_s16p%d], p5_s16%d',p,p),'--:2:-:-:2')
 e(string.format('@P0 STG.E.WT [p5_s32p%d], p5_s32%d',p,p),'--:2:-:-:2')
 e(string.format('@P0 STG.E.WT [p5_ap%d], p5_scale%d',p,p),'--:2:-:-:2')
end
e('NOP','02:-:-:-:6')
l('H5_PUBLISH_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
e('BRA POSITION_END','--:-:-:Y:d')
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=rep(s,'ISETP.GE.U32.AND P6, PT, op, 0x6d, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6e, PT;')
s=rep(s,'--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',[=[--:-:-:-:6 BRA hcup5_SETUP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6      BRA POSITION_END;

embrow_SETUP:]=])
s=rep(s,'\nPOSITION_END:','\n'..table.concat(out,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write(string.format('native_hcup positions=%d weight_passes=1 gate_evaluations_per_column=4 peer_warps=6 shared_input_bytes=%d\n',positions,positions*1536))
