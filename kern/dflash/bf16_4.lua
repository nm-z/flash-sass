-- Native B4 ordinary BF16 projections (including alpha/beta and indexer).
-- Match the scalar lanes: 32 for K2560, eight for K320.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i,j=s:find(a,1,true);assert(i,'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
-- Reuse the DQM register map, with float activations and BF16 packed weights.
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]='DBF_'..x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
l('SETUP');e('NOP','3f:-:-:Y:d')
e('LDS dqm_groups, [RZ+0xbfa0]','--:-:1:-:2')
e('LDS dqm_stride, [RZ+0xbfa4]','--:-:1:-:2')
e('LDS dqm_head, [RZ+0xbfa8]','--:-:1:-:2') -- optional bias, not vocabulary head
e('LDS dqm_logrows, [RZ+0xbfac]','--:-:1:-:2');e('LDS dqm_loglanes, [RZ+0xbfb0]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('IADD dqm_temp, dqm_loglanes, 0x3');e('SHR.U32 dqm_groups, dqm_groups, dqm_temp')
e('MOV dqm_lanes, 0x1');e('SHL dqm_lanes, dqm_lanes, dqm_loglanes')
e('IADD dqm_temp, dqm_lanes, -0x1');e('LOP.AND dqm_lane, tid, dqm_temp')
e('SHL dqm_header_bytes, dqm_lanes, 0x5');e('SHL dqm_high_base, dqm_lanes, 0x4')
e('MOV dqm_rowstep, 0x10');e('SHL dqm_rowstep, dqm_rowstep, dqm_logrows')
pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
e('IADD dqm_offset, pidx, 0x1');e('IADD dqm_offset, dqm_offset, warp');e('SHL dqm_offset, dqm_offset, 0x7')
e('IADD dqm_pointer0.CC, prog0, dqm_offset');e('IADD.X dqm_pointer1, prog1, RZ')
e('SHL dqm_offset, lane, 0x2');e('IADD dqm_pointer0.CC, dqm_pointer0, dqm_offset');e('IADD.X dqm_pointer1, dqm_pointer1, RZ')
e('@P5 LDG.E.CG dqm_temp, [dqm_pointer]','--:-:1:-:2')
e('ISCADD dqm_offset, warp, dqm_offset, 0x7');e('IADD dqm_offset, dqm_offset, 0x8000')
e('@P5 STS [dqm_offset], dqm_temp','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS.64 dqm_weight0, [RZ+0x8008]','--:-:1:-:2')
e('LDS.64 dqm_abase00, [RZ+0x8028]','--:-:1:-:2') -- shared bias pointer
e('LDS dqm_rows, [RZ+0x8020]','--:-:1:-:2');e('LDS dqm_flags, [RZ+0x8078]','--:-:1:-:2')
for p=0,3 do
 e(string.format('LDS.64 dqm_xbase%02d, [RZ+0x%x]',p*2,0x8010+p*128),'--:-:1:-:2')
 e(string.format('LDS.64 dqm_ybase%02d, [RZ+0x%x]',p*2,0x8018+p*128),'--:-:1:-:2')
end
e('NOP','01:-:-:-:6')
e('MOV dqm_temp, 0x1');e('SHL dqm_temp, dqm_temp, dqm_logrows');e('SHL dqm_temp, dqm_temp, dqm_loglanes')
pred('ISETP.GE.U32.AND P6, PT, tid, dqm_temp, PT');e('@P6 BRA DBF_ROWS_DONE','--:-:-:Y:d')
e('SHL dqm_row, cta, dqm_logrows');e('SHR.U32 dqm_temp, tid, dqm_loglanes');e('IADD dqm_row, dqm_row, dqm_temp')
l('ROW');e('SHR.U32 dqm_temp, lane, dqm_loglanes');e('IADD dqm_temp, dqm_row, -dqm_temp')
pred('ISETP.GE.U32.AND P6, PT, dqm_temp, dqm_rows, PT');e('@P6 BRA DBF_ROWS_DONE','--:-:-:Y:d')
pred('ISETP.LT.U32.AND P1, PT, dqm_row, dqm_rows, PT')
e('XMAD dqm_offset, dqm_row, dqm_stride, RZ');e('ISCADD dqm_offset, dqm_lane, dqm_offset, 0x4')
e('IADD dqm_rowbase0.CC, dqm_weight0, dqm_offset');e('IADD.X dqm_rowbase1, dqm_weight1, RZ')
e('SHL dqm_offset, dqm_lane, 0x5')
for p=0,3 do
 e(string.format('IADD dqm_sbase%02d.CC, dqm_xbase%02d, dqm_offset',p*2,p*2))
 e(string.format('IADD.X dqm_sbase%02d, dqm_xbase%02d, RZ',p*2+1,p*2+1))
 e('MOV dqm_acc'..p..', RZ')
end
e('MOV dqm_group, RZ');l('K')
e('@P1 LDG.E.CI.128 dqm_codes0, [dqm_rowbase]','--:2:1:-:2')
for p=0,3 do for h=0,1 do
 e(string.format('LDG.E.CG.128 dqm_x%02d, [dqm_sbase%02d+0x%x]',p*8+h*4,p*2,h*16),'--:3:4:-:2')
end end
e('NOP','09:-:-:-:6')
for j=0,7 do
 if j%2==0 then e(string.format('SHL dqm_code, dqm_codes%d, 0x10',j//2))
 else e(string.format('LOP32I.AND dqm_code, dqm_codes%d, 0xffff0000',j//2))end
 for p=0,3 do e(string.format('FFMA dqm_acc%d, dqm_code.reuse, dqm_x%02d, dqm_acc%d',p,p*8+j,p),p==3 and '--:-:-:-:3' or '--:-:-:-:1')end
end
e('IADD dqm_rowbase0.CC, dqm_rowbase0, dqm_high_base','02:-:-:-:6');e('IADD.X dqm_rowbase1, dqm_rowbase1, RZ');e('NOP','04:-:-:-:6')
for p=0,3 do
 e(string.format('IADD dqm_sbase%02d.CC, dqm_sbase%02d, dqm_header_bytes',p*2,p*2))
 e(string.format('IADD.X dqm_sbase%02d, dqm_sbase%02d, RZ',p*2+1,p*2+1))
end
e('IADD dqm_group, dqm_group, 0x1');pred('ISETP.LT.U32.AND P6, PT, dqm_group, dqm_groups, PT');e('@P6 BRA DBF_K','--:-:-:Y:d')
for _,distance in ipairs({16,8,4,2,1})do
 pred(string.format('ISETP.LE.U32.AND P6, PT, dqm_lanes, 0x%x, PT',distance));e('@P6 BRA DBF_REDUCE_DONE_'..distance)
 for p=0,3 do e(string.format('SHFL.BFLY PT, dqm_value%d, dqm_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 e('NOP','10:-:-:-:6')
 for p=0,3 do e(string.format('FADD dqm_acc%d, dqm_acc%d, dqm_value%d',p,p,p))end
 l('REDUCE_DONE_'..distance)
end
pred('ISETP.EQ.U32.AND P2, PT, dqm_lane, RZ, P1')
pred('ISETP.NE.U32.AND P5, PT, dqm_head, RZ, P2')
e('SHL dqm_offset, dqm_row, 0x2')
e('IADD dqm_pointer0.CC, dqm_abase00, dqm_offset');e('IADD.X dqm_pointer1, dqm_abase01, RZ')
e('@P5 LDG.E.CI dqm_value0, [dqm_pointer]','--:-:1:-:2');e('NOP','01:-:-:-:6')
for p=0,3 do
 e(string.format('@P5 FADD dqm_acc%d, dqm_acc%d, dqm_value0',p,p))
 e(string.format('IADD dqm_pointer0.CC, dqm_ybase%02d, dqm_offset',p*2));e(string.format('IADD.X dqm_pointer1, dqm_ybase%02d, RZ',p*2+1))
 e(string.format('@P2 STG.E [dqm_pointer], dqm_acc%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
end
e('LOP32I.AND dqm_temp, dqm_flags, 0x100080');pred('ISETP.EQ.U32.AND P6, PT, dqm_temp, RZ, PT');e('@P6 BRA DBF_NEXT_ROW','--:-:-:Y:d')
e('LOP32I.AND dqm_temp, dqm_flags, 0x100000');pred('ISETP.NE.U32.AND P5, PT, dqm_temp, RZ, PT')
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
l('NEXT_ROW');e('IADD dqm_row, dqm_row, dqm_rowstep');e('BRA DBF_ROW','--:-:-:Y:d')
l('ROWS_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0');e('BRA POSITION_END','--:-:-:Y:d')
replace('ISETP.GE.U32.AND P6, PT, op, 0x7a, PT;','ISETP.GE.U32.AND P6, PT, op, 0x7b, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA DBF_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
