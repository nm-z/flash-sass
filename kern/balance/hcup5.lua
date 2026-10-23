-- Five-position GR up projection with one weight stream. Preserve the native
-- SiLU, eight-lane dot association, gate/mix, quantization and peer publication.
-- Matrix output ownership matches hcquant's32-column (128 interleaved row) CTA.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(text,old,new)
 local p,q=assert(text:find(old,1,true));assert(not text:find(old,q+1,true),'ambiguous '..old)
 return text:sub(1,p-1)..new..text:sub(q+1)
end
local a=assert(s:find('hcquant_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
local native=s:sub(a,b-1)
local begin=assert(native:find('hcquant_BEGIN:',1,true))
local prep_end=assert(native:find('MOV hcquant_yp0, hcquant_param_TAB0;',begin,true))
local prep_last=assert(native:sub(1,prep_end-1):match('.*()\n'))
local prep=native:sub(begin,prep_last):gsub('hcquant_BEGIN:','H5_PREP_BODY:')
local stage=[=[--:-:-:-:6 SHL h5_sap, h5_position, 0x9;
--:-:-:-:6 ISCADD h5_sap, h5_position, h5_sap, 0xa;
--:-:-:-:6 IADD h5_sap, h5_sap, 0x2000;
--:-:-:-:6 IADD h5_sap, h5_sap, hcquant_sa;
]=]
prep=rep(prep,'--:1:-:-:1      STS [hcquant_sa], hcquant_sy0;',
         stage..'--:1:-:-:1 STS [h5_sap], hcquant_sy0;')
prep=rep(prep,'STS [hcquant_sa+0x400], hcquant_sy1;', 'STS [h5_sap+0x400], hcquant_sy1;')
-- Keep parameter/constants setup for the native finish, removing its redundant
-- SiLU and dot work. The precomputed raw dot is precisely the lane0 value used
-- by the original gate/mix shuffle; other dot lanes never contribute to it.
local finish=native
local p=assert(finish:find('SHL hcquant_sa, hcquant_tid, 0x2;',1,true))
local first=(finish:sub(1,p-1):match('.*()\n') or 0)+1
finish=finish:sub(1,first-1)..finish:sub(prep_last+1)
p=assert(finish:find('SHL hcquant_t1, hcquant_lane, 0x5;',1,true))
first=(finish:sub(1,p-1):match('.*()\n') or 0)+1
local _,last=assert(finish:find('LDS.128 hcquant_act36, [hcquant_t1+0x410];',p,true))
finish=finish:sub(1,first-1)..finish:sub(last+1)
for j=0,4 do
 local ins=string.format('LDG.E.CI.128 hcquant_w%02d, [hcquant_wp+0x%x];',j*4,j*128)
 p=assert(finish:find(ins,1,true));first=(finish:sub(1,p-1):match('.*()\n') or 0)+1
 last=assert(finish:find('\n',p,true));finish=finish:sub(1,first-1)..finish:sub(last+1)
end
finish=rep(finish,'MOV hcquant_accf, RZ;', [=[MOV hcquant_accf, RZ;
--:2:1:-:2 @P3 LDG.E.CG hcquant_accf, [hcquant_yp];]=])
p=assert(finish:find('SHL hcquant_a0, hcquant_w00, 0x10;',1,true))
first=(finish:sub(1,p-1):match('.*()\n') or 0)+1
_,last=assert(finish:find('STG.E [hcquant_yp], hcquant_accf;',p,true))
finish=finish:sub(1,first-1)..'03:-:-:-:6 NOP; # native batched dot and residual input ready'..finish:sub(last+1)
finish=finish:gsub('hcquant_','h5finish_'):gsub('hcq_','h5quant_')
finish=finish:gsub('BRA POSITION_END;','BRA H5_FINISH_NEXT;')
local maps={}
for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find(': hcquant_',1,true)then maps[#maps+1]=line:gsub('hcquant_','h5finish_')end
 if line:find(': hcq_',1,true)then maps[#maps+1]=line:gsub('hcq_','h5quant_')end
end
maps[#maps+1]=[=[
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
]=]
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)out[#out+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
local function install(cache)
 pred('ISETP.EQ.U32.AND P5, PT, warp, RZ, PT')
 e('SHL h5_sap, lane, 0x2')
 e('SHL h5_offset, h5_position, 0x7');e('IADD h5_offset, h5_offset, 0x8000')
 e('IADD h5_offset, h5_offset, h5_sap')
 if cache then
  e('@P5 LDS h5_word, [h5_offset]','--:-:1:-:2')
 else
  e('IADD h5_tmp2, pidx, 0x1');e('IADD h5_tmp2, h5_tmp2, h5_position');e('SHL h5_tmp2, h5_tmp2, 0x7')
  e('IADD h5_descriptor0.CC, prog0, h5_tmp2');e('IADD.X h5_descriptor1, prog1, RZ')
  e('IADD h5_descriptor0.CC, h5_descriptor0, h5_sap');e('IADD.X h5_descriptor1, h5_descriptor1, RZ')
  e('@P5 LDG.E.CG h5_word, [h5_descriptor]','--:-:1:-:2')
 end
 e('@P5 STS [h5_sap+0xbf80], h5_word','01:2:-:-:2')
 if not cache then e('@P5 STS [h5_offset], h5_word','--:2:-:-:2')end
 e('BAR.SYNC 0x0','02:-:-:Y:d')
 e('MOV vid, cta');e('MOV vtid, tid');e('MOV units, 0x10')
end
l('hcup5_SETUP');e('NOP','3f:-:-:Y:d');e('MOV h5_position, RZ')
l('H5_PREP_NEXT');install(false);out[#out+1]=prep
e('IADD h5_position, h5_position, 0x1');pred('ISETP.LT.U32.AND P6, PT, h5_position, 0x5, PT')
e('@P6 BRA H5_PREP_NEXT','--:-:-:Y:d')
e('LDS h5_rows, [RZ+0xbfa0]','--:-:1:-:2')
e('LDS.64 h5_weights0, [RZ+0xbf88]','--:-:1:-:2')
for p=0,4 do e(string.format('LDS.64 h5_out%02d, [RZ+0x%x]',p*2,0x8000+p*128+24),'--:-:1:-:2')end
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
for p=0,4 do e('MOV h5_acc'..p..', RZ','--:-:-:-:1')end
e('MOV h5_tile, RZ')
l('H5_MATRIX_K')
e('LDG.E.CI.128 h5_packed0, [h5_wp]','--:2:1:-:2')
for p=0,4 do for half=0,1 do
 e(string.format('LDS.128 h5_act%02d, [h5_xs+0x%x]',p*8+half*4,p*0x600+half*16),'--:-:3:-:2')
end end
e('NOP','05:-:-:-:6')
for j=0,7 do
 if j%2==0 then e(string.format('SHL h5_float%d, h5_packed%d, 0x10',j,j//2),'--:-:-:-:1')
 else e(string.format('LOP32I.AND h5_float%d, h5_packed%d, 0xffff0000',j,j//2),'--:-:-:-:1')end
end
for j=0,7 do for p=0,4 do
 e(string.format('FFMA h5_acc%d, h5_float%d.reuse, h5_act%02d, h5_acc%d',p,j,p*8+j,p),
   p==4 and '--:-:-:-:2' or '--:-:-:-:1')
end end
e('IADD h5_wp0.CC, h5_wp0, 0x80','02:-:-:-:6');e('IADD.X h5_wp1, h5_wp1, RZ')
e('IADD h5_xs, h5_xs, 0x100');e('IADD h5_tile, h5_tile, 0x1')
pred('ISETP.LT.U32.AND P6, PT, h5_tile, 0x5, PT');e('@P6 BRA H5_MATRIX_K','--:-:-:Y:d')
for _,distance in ipairs({4,2,1})do
 for p=0,4 do e(string.format('SHFL.BFLY PT, h5_shuffle%d, h5_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 e('NOP','10:-:-:-:6')
 for p=0,4 do e(string.format('FADD h5_acc%d, h5_acc%d, h5_shuffle%d',p,p,p),p==4 and '--:-:-:-:6' or '--:-:-:-:1')end
end
e('SHL h5_offset, h5_weight_row, 0x2');pred('ISETP.EQ.U32.AND P0, PT, h5_dot_lane, RZ, PT')
for p=0,4 do
 e(string.format('IADD h5_descriptor0.CC, h5_out%02d, h5_offset',p*2))
 e(string.format('IADD.X h5_descriptor1, h5_out%02d, RZ',p*2+1))
 e(string.format('@P0 STG.E [h5_descriptor], h5_acc%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
end
e('IADD h5_row, h5_row, 0x20');pred('ISETP.LT.U32.AND P6, PT, h5_row, h5_limit, PT')
e('@P6 BRA H5_MATRIX_ROW','--:-:-:Y:d')
l('H5_MATRIX_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
e('MOV h5_position, RZ')
l('H5_FINISH_POSITION');install(true);e('BRA h5finish_SETUP','--:-:-:Y:d')
out[#out+1]=finish
l('H5_FINISH_NEXT');e('BAR.SYNC 0x0','3f:-:-:Y:d');e('MEMBAR.GL.IVALLD')
e('IADD h5_position, h5_position, 0x1');pred('ISETP.LT.U32.AND P6, PT, h5_position, 0x5, PT')
e('@P6 BRA H5_FINISH_POSITION','--:-:-:Y:d');e('BRA POSITION_END','--:-:-:Y:d')
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=rep(s,'ISETP.GE.U32.AND P6, PT, op, 0x6d, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6e, PT;')
s=rep(s,'--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',[=[--:-:-:-:6 BRA hcup5_SETUP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6      BRA POSITION_END;

embrow_SETUP:]=])
s=rep(s,'\nPOSITION_END:','\n'..table.concat(out,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('native_hcup5 positions=5 weight_passes=1 native_mixing_quantization=retained shared_input_bytes=7680\n')
