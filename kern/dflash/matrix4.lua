-- Quantized B4 matrices in the existing packed target layouts. One unpacked
-- group supplies four integer dots and four independent FP accumulation
-- chains. Keep the scalar group/warp reduction association for each output.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2])
local function replace(a,b)
 local i,j=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local map=[=[
    16-79 : dqm_x<00-63>
    80-83 : dqm_dot<0-3>
    84-87 : dqm_acc<0-3>
    88-91 : dqm_as<0-3>
    92-95 : dqm_sum<0-3>
    96-99 : dqm_value<0-3>
    100-103 : dqm_codes<0-3>
    104-107 : dqm_header<0-3>
    108 : dqm_high
    109 : dqm_code
    110 : dqm_sc
    111 : dqm_min
    112 : dqm_d
    113 : dqm_dm
    114 : dqm_scfloat
    115 : dqm_temp
    116 : dqm_shift
    117 : dqm_aux
    118 : dqm_row
    119 : dqm_rows
    120 : dqm_rowstep
    121 : dqm_group
    122 : dqm_lane
    123 : dqm_lanes
    124 : dqm_groups
    125 : dqm_stride
    126 : dqm_header_bytes
    127 : dqm_high_base
    128,129 : dqm_weight<0-1>
    130,131 : dqm_rowbase<0-1>
    132,133 : dqm_pointer<0-1>
    134,135 : dqm_other<0-1>
    136-143 : dqm_xbase<00-07>
    144-151 : dqm_sbase<00-07>
    152-159 : dqm_abase<00-07>
    160-167 : dqm_ybase<00-07>
    168 : dqm_offset
    169 : dqm_block
    170 : dqm_subgroup
    171 : dqm_flags
    172 : dqm_global_row
    173 : dqm_die
    174 : dqm_pub_offset
    175 : dqm_position
    176,177 : dqm_table<0-1>
    178,179 : dqm_dest<0-1>
    180-183 : dqm_max<0-3>
    184-187 : dqm_best<0-3>
    188 : dqm_loglanes
    189 : dqm_logrows
    190 : dqm_head
    191 : dqm_add
    192-195 : dqm_lut<0-3>
    196 : dqm_tilelog
]=]
local head_first=assert(s:find('\nq5kk2560_END:',1,true))
local head_last=assert(s:find('HEAD_PRODUCER_DONE:',head_first,true))+#'HEAD_PRODUCER_DONE:'-1
local head_tail=s:sub(head_first+#'\nq5kk2560_END:',head_last)
local bodies={}
for _,bits in ipairs({4,5,2,3,6,8})do
 local scaled=bits==3 or bits==6 or bits==8
 local affine=bits==4 or bits==5
 local prefix='DQM'..bits
 local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function l(x)o[#o+1]=prefix..'_'..x..':'end
 local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
 l('SETUP');e('NOP','3f:-:-:Y:d')
 -- Host supplies K, packed row stride and the scalar lane geometry.
 e('LDS dqm_groups, [RZ+0xbfa0]','--:-:1:-:2')
 e('LDS dqm_stride, [RZ+0xbfa4]','--:-:1:-:2')
 e('LDS dqm_loglanes, [RZ+0xbfa8]','--:-:1:-:2')
 e('LDS dqm_add, [RZ+0xbfac]','--:-:1:-:2')
 e('LDS dqm_tilelog, [RZ+0xbfb0]','--:-:1:-:2');e('NOP','01:-:-:-:6')
 if bits==2 then
  e('SHR.U32 dqm_header_bytes, dqm_groups, 0x5');e('IADD dqm_header_bytes, dqm_header_bytes, 0xf')
  e('LOP32I.AND dqm_header_bytes, dqm_header_bytes, 0xfffffff0')
 else e('SHR.U32 dqm_header_bytes, dqm_groups, 0x4')end -- 16*(K/256)
 e('SHR.U32 dqm_groups, dqm_groups, 0x5')
 e('XMAD dqm_high_base, dqm_header_bytes, 0x9, RZ')
 e('MOV dqm_lanes, 0x1');e('SHL dqm_lanes, dqm_lanes, dqm_loglanes')
 if scaled then e('SHL dqm_header_bytes, dqm_lanes, 0x1')end
 if bits==8 then
  for j,value in ipairs({'0xbfad9881','0xf6eaddcf','0x26190d01','0x71594535'})do e('MOV32I dqm_lut'..(j-1)..', '..value)end
 end
 e('IADD dqm_temp, dqm_lanes, -0x1');e('LOP.AND dqm_lane, tid, dqm_temp')
 e('MOV dqm_logrows, 0x8');e('IADD dqm_logrows, dqm_logrows, -dqm_loglanes')
 pred('ISETP.NE.U32.AND P6, PT, dqm_tilelog, RZ, PT');e('@P6 MOV dqm_logrows, dqm_tilelog')
 e('MOV dqm_rowstep, 0x10');e('SHL dqm_rowstep, dqm_rowstep, dqm_logrows')
 pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
 e('IADD dqm_offset, pidx, 0x1');e('IADD dqm_offset, dqm_offset, warp');e('SHL dqm_offset, dqm_offset, 0x7')
 e('IADD dqm_pointer0.CC, prog0, dqm_offset');e('IADD.X dqm_pointer1, prog1, RZ')
 e('SHL dqm_offset, lane, 0x2');e('IADD dqm_pointer0.CC, dqm_pointer0, dqm_offset');e('IADD.X dqm_pointer1, dqm_pointer1, RZ')
 e('@P5 LDG.E.CG dqm_temp, [dqm_pointer]','--:-:1:-:2')
 e('ISCADD dqm_offset, warp, dqm_offset, 0x7');e('IADD dqm_offset, dqm_offset, 0x8000')
 e('@P5 STS [dqm_offset], dqm_temp','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
 e('LDS.64 dqm_weight0, [RZ+0x8008]','--:-:1:-:2');e('LDS dqm_rows, [RZ+0x8028]','--:-:1:-:2')
 e('LDS dqm_flags, [RZ+0x8078]','--:-:1:-:2')
 for p=0,3 do
  for _,field in ipairs({{'xbase',0x10},{'sbase',0x18},{'abase',0x38},{'ybase',0x20}})do
   e(string.format('LDS.64 dqm_%s%02d, [RZ+0x%x]',field[1],p*2,0x8000+p*128+field[2]),'--:-:1:-:2')
  end
  e('MOV32I dqm_max'..p..', 0xff800000');e('MOV32I dqm_best'..p..', 0xffffffff')
 end
 e('NOP','01:-:-:-:6');e('LOP32I.AND dqm_head, dqm_flags, 0x1000000')
 e('IADD dqm_temp, dqm_logrows, dqm_loglanes');e('MOV dqm_offset, 0x1');e('SHL dqm_temp, dqm_offset, dqm_temp')
 pred('ISETP.GE.U32.AND P6, PT, tid, dqm_temp, PT');e('@P6 BRA '..prefix..'_ROWS_DONE','--:-:-:Y:d')
 e('SHL dqm_row, cta, dqm_logrows');e('SHR.U32 dqm_temp, tid, dqm_loglanes');e('IADD dqm_row, dqm_row, dqm_temp')
 l('ROW')
 -- All lanes of a warp take the same row-loop branch, including a final
 -- partial pair of sixteen-lane rows. Invalid rows predicate memory accesses.
 e('SHR.U32 dqm_temp, lane, dqm_loglanes');e('IADD dqm_temp, dqm_row, -dqm_temp')
 pred('ISETP.GE.U32.AND P6, PT, dqm_temp, dqm_rows, PT');e('@P6 BRA '..prefix..'_ROWS_DONE','--:-:-:Y:d')
 pred('ISETP.LT.U32.AND P1, PT, dqm_row, dqm_rows, PT')
 e('XMAD dqm_offset, dqm_row, dqm_stride, RZ')
 e('IADD dqm_rowbase0.CC, dqm_weight0, dqm_offset');e('IADD.X dqm_rowbase1, dqm_weight1, RZ')
 for p=0,3 do e('MOV dqm_acc'..p..', RZ')end
 e('MOV dqm_group, dqm_lane')
 l('GROUP')
 if scaled then
  e('SHR.U32 dqm_block, dqm_group, 0x3');e('LOP32I.AND dqm_subgroup, dqm_group, 0x7')
  e('SHR.U32 dqm_offset, dqm_group, dqm_loglanes');e('SHL dqm_offset, dqm_offset, 0x1')
  e('SHR.U32 dqm_temp, dqm_lane, 0x3');e('ISCADD dqm_offset, dqm_temp, dqm_offset, 0x4')
  e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
  e('@P1 LDG.E.CI.U16 dqm_header0, [dqm_pointer]','--:2:1:-:2');e('NOP','03:-:-:-:6')
  if bits==8 then
   e('ISCADD dqm_offset, dqm_group, dqm_header_bytes, 0x4')
   e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
   e('@P1 LDG.E.CI.128 dqm_codes0, [dqm_pointer]','--:2:1:-:2');e('NOP','03:-:-:-:6')
   e('SHL dqm_offset, dqm_groups, 0x4');e('IADD dqm_offset, dqm_offset, dqm_header_bytes');e('IADD dqm_offset, dqm_offset, dqm_group')
   e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
   e('@P1 LDG.E.CI.S8 dqm_header1, [dqm_pointer]','--:2:1:-:2');e('NOP','02:-:-:-:6')
  else
   local record=bits==3 and 112 or 208
   e(string.format('XMAD dqm_offset, dqm_block, 0x%x, dqm_header_bytes',record))
   e('IADD dqm_other0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_other1, dqm_rowbase1, RZ')
   e('SHL dqm_offset, dqm_subgroup, '..(bits==3 and '0x3' or '0x4'))
   e('IADD dqm_pointer0.CC, dqm_other0, dqm_offset');e('IADD.X dqm_pointer1, dqm_other1, RZ')
   e('@P1 LDG.E.CI.'..(bits==3 and '64' or '128')..' dqm_codes0, [dqm_pointer]','--:2:1:-:2');e('NOP','03:-:-:-:6')
   e('SHL dqm_offset, dqm_subgroup, '..(bits==3 and '0x2' or '0x3'))
   e('IADD dqm_offset, dqm_offset, '..(bits==3 and '0x40' or '0x80'))
   e('IADD dqm_pointer0.CC, dqm_other0, dqm_offset');e('IADD.X dqm_pointer1, dqm_other1, RZ')
   e('@P1 LDG.E.CI dqm_high, [dqm_pointer]','--:2:1:-:2')
   if bits==6 then e('@P1 LDG.E.CI dqm_aux, [dqm_pointer+0x4]','--:2:1:-:2')end
   e('NOP','03:-:-:-:6')
   e('SHL dqm_offset, dqm_subgroup, 0x1');e('IADD dqm_offset, dqm_offset, '..(bits==3 and '0x60' or '0xc0'))
   e('IADD dqm_pointer0.CC, dqm_other0, dqm_offset');e('IADD.X dqm_pointer1, dqm_other1, RZ')
   e('@P1 LDG.E.CI.U16 dqm_header1, [dqm_pointer]','--:2:1:-:2');e('NOP','02:-:-:-:6')
  end
 elseif bits==2 then
  e('SHR.U32 dqm_block, dqm_group, 0x1');e('SHL dqm_offset, dqm_block, 0x1')
 else
  e('SHR.U32 dqm_block, dqm_group, 0x3');e('LOP32I.AND dqm_subgroup, dqm_group, 0x7')
  e('SHL dqm_offset, dqm_block, 0x4')
 end
 if not scaled then
 e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
 e('@P1 LDG.E.CI.'..(bits==2 and 'U16' or '128')..' dqm_header0, [dqm_pointer]','--:2:1:-:2');e('NOP',bits==2 and '03:-:-:-:6' or '02:-:-:-:6')
 e('ISCADD dqm_offset, dqm_group, dqm_header_bytes, '..(bits==2 and '0x3' or '0x4'))
 e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
 e('@P1 LDG.E.CI.'..(bits==2 and '64' or '128')..' dqm_codes0, [dqm_pointer]','--:2:1:-:2');e('NOP','02:-:-:-:6')
 if bits==5 then
  e('NOP','01:-:-:-:6')
  e('ISCADD dqm_offset, dqm_group, dqm_high_base, 0x2')
  e('IADD dqm_pointer0.CC, dqm_rowbase0, dqm_offset');e('IADD.X dqm_pointer1, dqm_rowbase1, RZ')
  e('@P1 LDG.E.CI dqm_high, [dqm_pointer]','--:2:1:-:2');e('NOP','02:-:-:-:6')
 end
 end
 e('SHL dqm_offset, dqm_group, 0x6')
 for p=0,3 do
  e(string.format('IADD dqm_pointer0.CC, dqm_xbase%02d, dqm_offset',p*2));e(string.format('IADD.X dqm_pointer1, dqm_xbase%02d, RZ',p*2+1))
  for j=0,3 do e(string.format('LDG.E.CG.128 dqm_x%02d, [dqm_pointer+0x%x]',p*16+j*4,j*16),'--:3:4:-:2')end
  e('NOP','04:-:-:-:6')
 end
 e('NOP','09:-:-:-:6') -- retire vector loads before the scalar AS/sum group
 e('SHL dqm_offset, dqm_group, 0x2')
 for p=0,3 do
  if not scaled then
  e(string.format('IADD dqm_pointer0.CC, dqm_sbase%02d, dqm_offset',p*2));e(string.format('IADD.X dqm_pointer1, dqm_sbase%02d, RZ',p*2+1))
  e(string.format('LDG.E.CG dqm_sum%d, [dqm_pointer]',p),'--:3:4:-:2');e('NOP','04:-:-:-:6')
  end
  e(string.format('IADD dqm_pointer0.CC, dqm_abase%02d, dqm_offset',p*2));e(string.format('IADD.X dqm_pointer1, dqm_abase%02d, RZ',p*2+1))
  e(string.format('LDG.E.CG dqm_as%d, [dqm_pointer]',p),'--:3:4:-:2');e('NOP','04:-:-:-:6')
 end
 e('NOP','09:-:-:-:6')
 if affine then
 e('LOP32I.AND dqm_shift, dqm_subgroup, 0x3');e('SHL dqm_shift, dqm_shift, 0x3')
 e('SHR.U32 dqm_sc, dqm_header1, dqm_shift');e('SHR.U32 dqm_min, dqm_header2, dqm_shift')
 e('SHR.U32 dqm_aux, dqm_header3, dqm_shift')
 pred('ISETP.LT.U32.AND P4, PT, dqm_subgroup, 0x4, PT')
 e('@!P4 SHR.U32 dqm_sc, dqm_sc, 0x6');e('@!P4 LOP32I.AND dqm_sc, dqm_sc, 0x3');e('@!P4 SHL dqm_sc, dqm_sc, 0x4')
 e('LOP32I.AND dqm_temp, dqm_aux, 0xf');e('@!P4 LOP.OR dqm_sc, dqm_sc, dqm_temp')
 e('@!P4 SHR.U32 dqm_min, dqm_min, 0x6');e('@!P4 LOP32I.AND dqm_min, dqm_min, 0x3');e('@!P4 SHL dqm_min, dqm_min, 0x4')
 e('SHR.U32 dqm_temp, dqm_aux, 0x4');e('LOP32I.AND dqm_temp, dqm_temp, 0xf');e('@!P4 LOP.OR dqm_min, dqm_min, dqm_temp')
 e('@P4 LOP32I.AND dqm_sc, dqm_sc, 0x3f');e('@P4 LOP32I.AND dqm_min, dqm_min, 0x3f')
 e('F2F.F32.F16 dqm_d, dqm_header0.H0','--:-:5:-:2');e('F2F.F32.F16 dqm_dm, dqm_header0.H1','--:-:5:-:2')
 e('I2F.F32.U32 dqm_scfloat, dqm_sc','--:-:5:-:2')
 else e('F2F.F32.F16 dqm_d, dqm_header0.H0','--:-:5:-:2')end
 if scaled and bits~=8 then
  e('BFE dqm_sc, dqm_header1, 0x800');e('BFE dqm_min, dqm_header1, 0x808')
 end
 for p=0,3 do e('MOV dqm_dot'..p..', RZ')end
 for j=0,31 do
  local width=(bits==2 or bits==3) and 2 or 4;local elements=32//width
  e(string.format('BFE.U32 dqm_code, dqm_codes%d, 0x%x',j//elements,width*256+(j%elements)*width))
  if bits==5 then
   e(string.format('BFE.U32 dqm_temp, dqm_high, 0x%x',0x100+j));e('ISCADD dqm_code, dqm_temp, dqm_code, 0x4')
  end
  if bits==3 then
   e(string.format('BFE.U32 dqm_temp, dqm_high, 0x%x',0x100+j));e('SHL dqm_temp, dqm_temp, 0x2');e('IADD dqm_code, dqm_code, -dqm_temp')
  elseif bits==6 then
   e(string.format('BFE.U32 dqm_temp, %s, 0x%x',j<16 and 'dqm_high' or 'dqm_aux',0x200+(j%16)*2))
   e('ISCADD dqm_code, dqm_temp, dqm_code, 0x4');e('IADD dqm_code, dqm_code, -0x20')
  elseif bits==8 then
   pred('ISETP.LT.U32.AND P4, PT, dqm_code, 0x8, PT')
   e('LOP32I.AND dqm_temp, dqm_code, 0x7')
   e('@P4 PRMT dqm_code, dqm_lut0, dqm_lut1, dqm_temp')
   e('@!P4 PRMT dqm_code, dqm_lut2, dqm_lut3, dqm_temp')
   e('BFE dqm_code, dqm_code, 0x800')
  end
  if scaled then e('XMAD.S16.S16 dqm_code, dqm_code, '..(bits==8 and 'dqm_header1' or j<16 and 'dqm_sc' or 'dqm_min')..', RZ')end
  for p=0,3 do e(string.format('XMAD.%s.S16 dqm_dot%d, dqm_code, dqm_x%02d.H%d, dqm_dot%d',scaled and 'S16' or 'U16',p,p*16+j//2,j%2,p),p==3 and '--:-:-:-:3' or '--:-:-:-:1')end
 end
 e('NOP','10:-:-:-:6')
 for p=0,3 do
  if bits==2 then e(string.format('IADD dqm_dot%d, dqm_dot%d, -dqm_sum%d',p,p,p))
  elseif affine then e(string.format('XMAD.U16.S16 dqm_sum%d, dqm_min, dqm_sum%d, RZ',p,p))end
  e(string.format('I2F.F32.S32 dqm_value%d, dqm_dot%d',p,p),'--:-:5:-:2')
  if affine then e(string.format('I2F.F32.S32 dqm_sum%d, dqm_sum%d',p,p),'--:-:5:-:2')end
 end
 e('NOP','10:-:-:-:6')
 for p=0,3 do
  if affine then e(string.format('FMUL dqm_value%d, dqm_value%d, dqm_scfloat',p,p))end
  e(string.format('FMUL dqm_value%d, dqm_d, dqm_value%d',p,p))
  if affine then e(string.format('FFMA dqm_value%d, -dqm_dm, dqm_sum%d, dqm_value%d',p,p,p))end
  e(string.format('FFMA dqm_acc%d, dqm_as%d, dqm_value%d, dqm_acc%d',p,p,p,p))
 end
 e('IADD dqm_group, dqm_group, dqm_lanes');pred('ISETP.LT.U32.AND P6, PT, dqm_group, dqm_groups, PT')
 e('@P6 BRA '..prefix..'_GROUP','--:-:-:Y:d')
 for _,distance in ipairs({16,8,4,2,1})do
  pred(string.format('ISETP.LE.U32.AND P6, PT, dqm_lanes, 0x%x, PT',distance));e('@P6 BRA '..prefix..'_REDUCE'..distance..'_DONE','--:-:-:Y:d')
  for p=0,3 do e(string.format('SHFL.BFLY PT, dqm_value%d, dqm_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
  e('NOP','10:-:-:-:6')
  for p=0,3 do e(string.format('FADD dqm_acc%d, dqm_acc%d, dqm_value%d',p,p,p))end
  l('REDUCE'..distance..'_DONE')
 end
 pred('ISETP.EQ.U32.AND P2, PT, dqm_lane, RZ, P1')
 e('SHL dqm_offset, dqm_row, 0x2')
 for p=0,3 do
  e(string.format('IADD dqm_pointer0.CC, dqm_ybase%02d, dqm_offset',p*2));e(string.format('IADD.X dqm_pointer1, dqm_ybase%02d, RZ',p*2+1))
  pred('ISETP.NE.U32.AND P5, PT, dqm_add, RZ, P2')
  e('@P5 LDG.E.CG dqm_value0, [dqm_pointer]','--:-:1:-:2');e('NOP','01:-:-:-:6')
  e(string.format('@P5 FADD dqm_acc%d, dqm_acc%d, dqm_value0',p,p))
  e(string.format('@P2 STG.E [dqm_pointer], dqm_acc%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
  pred(string.format('FSETP.GT.AND P4, PT, dqm_acc%d, dqm_max%d, P2',p,p))
  e(string.format('@P4 MOV dqm_max%d, dqm_acc%d',p,p));e(string.format('@P4 MOV dqm_best%d, dqm_row',p))
 end
 -- Row-striped shared matrices and ordinary contiguous output publications.
 e('LOP32I.AND dqm_temp, dqm_flags, 0x100080')
 pred('ISETP.EQ.U32.AND P6, PT, dqm_temp, RZ, PT');e('@P6 BRA '..prefix..'_NEXT_ROW','--:-:-:Y:d')
 e('LOP32I.AND dqm_temp, dqm_flags, 0x100000')
 pred('ISETP.NE.U32.AND P5, PT, dqm_temp, RZ, PT')
 for p=0,3 do
  e(string.format('@P5 LDS.64 dqm_table0, [RZ+0x%x]',0x8050+p*128),'--:-:1:-:2')
  e(string.format('@!P5 LDS.64 dqm_table0, [RZ+0x%x]',0x8058+p*128),'--:-:1:-:2')
  e(string.format('LDS dqm_die, [RZ+0x%x]',0x8060+p*128),'--:-:1:-:2')
  e(string.format('LDS dqm_pub_offset, [RZ+0x%x]',0x8064+p*128),'--:-:1:-:2');e('NOP','01:-:-:-:6')
  e('SHR.U32 dqm_global_row, dqm_row, 0x3');e('XMAD dqm_global_row, dqm_global_row, 0x6, dqm_die')
  e('LOP32I.AND dqm_temp, dqm_row, 0x7');e('ISCADD dqm_global_row, dqm_global_row, dqm_temp, 0x3')
  e('IADD dqm_global_row, dqm_global_row, dqm_pub_offset');e('SHL dqm_global_row, dqm_global_row, 0x2')
  e('@!P5 ISCADD dqm_global_row, dqm_row, dqm_die, 0x2')
  for peer=0,5 do
   e(string.format('@P2 LDG.E.CG.64 dqm_dest0, [dqm_table+0x%x]',peer*8),'--:-:1:-:2');e('NOP','01:-:-:-:6')
   e('IADD dqm_dest0.CC, dqm_dest0, dqm_global_row');e('IADD.X dqm_dest1, dqm_dest1, RZ')
   e(string.format('@P2 STG.E.WT [dqm_dest], dqm_acc%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
  end
 end
 l('NEXT_ROW');e('IADD dqm_row, dqm_row, dqm_rowstep');e('BRA '..prefix..'_ROW','--:-:-:Y:d')
 l('ROWS_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
 pred('ISETP.EQ.U32.AND P6, PT, dqm_head, RZ, PT');e('@P6 BRA '..prefix..'_RETURN','--:-:-:Y:d')
 -- Reuse the production head producer's warp/CTA/die winner reduction.
 for p=0,3 do
  pred('ISETP.EQ.U32.AND P5, PT, warp, RZ, PT');e('SHL dqm_offset, lane, 0x2')
  e(string.format('@P5 LDS dqm_temp, [dqm_offset+0x%x]',0x8000+p*128),'--:-:1:-:2')
  e('@P5 STS [dqm_offset+0xbf80], dqm_temp','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
  e('MOV hp_mode, dqm_head');e('MOV hp_max, dqm_max'..p);e('MOV hp_idx, dqm_best'..p)
  o[#o+1]=head_tail:gsub('HEAD_PRODUCER_DONE',prefix..'_HEAD_DONE_'..p)
  e('BAR.SYNC 0x0','3f:-:-:Y:d')
 end
 l('RETURN');e('BRA POSITION_END','--:-:-:Y:d')
 bodies[#bodies+1]=table.concat(o,'\n')
end
replace('</REGISTER_MAPPING>',map..'\n</REGISTER_MAPPING>')
replace('ISETP.GE.U32.AND P6, PT, op, 0x74, PT;','ISETP.GE.U32.AND P6, PT, op, 0x7a, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA DQM4_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA DQM5_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA DQM2_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA DQM3_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA DQM6_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA DQM8_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(bodies,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'w'));f:write(s);f:close()
