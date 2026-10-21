-- Drafter-only FP32 glue. The activation polynomial follows ref/fmath.c.
local tag=assert(arg[1]);local out={}
local function emit(s) out[#out+1]=s..'\n' end
local function ins(s) emit('--:-:-:-:6 '..s) end
local function params(list)
 emit('<CONSTANT_MAPPING>')
 for _,p in ipairs(list) do emit(string.format('%s : c[0x0][0x%x]',p[1],p[2])) end
 emit('</CONSTANT_MAPPING>')
end
local function ptr(dst,lo,hi,off)
 ins('IADD '..dst..'0.CC, '..lo..', '..off..';');ins('IADD.X '..dst..'1, '..hi..', RZ;')
end
emit('# Kernel: '..tag)
if tag=='eh_concat' then
 params{{'param_E0',0x140},{'param_E1',0x144},{'param_H0',0x148},{'param_H1',0x14c},{'param_Y0',0x150},{'param_Y1',0x154}}
 emit('<REGISTER_MAPPING>\n0 : tid\n1 : bid\n2 : off\n3 : t\n4,5 : ep<0-1>\n6,7 : hp<0-1>\n8,9 : yp<0-1>\n10 : e\n11 : h\n</REGISTER_MAPPING>')
 emit('--:-:6:-:1 S2R tid, SR_TID.X;\n--:-:6:-:2 S2R bid, SR_CTAID.X;\n20:-:-:-:6 SHL t, tid, 0x2;')
 ptr('ep','param_E0','param_E1','t')
 ins('XMAD off, bid, 0x2800, t;');ptr('hp','param_H0','param_H1','off')
 ins('XMAD off, bid, 0x5000, t;');ptr('yp','param_Y0','param_Y1','off')
 for j=0,9 do
  emit(string.format('--:-:1:-:1 LDG.E.CI e, [ep+0x%x];\n--:-:1:-:2 LDG.E.CI h, [hp+0x%x];',j*1024,j*1024))
  emit(string.format('01:2:-:-:6 STG.E [yp+0x%x], e;\n--:2:-:-:6 STG.E [yp+0x%x], h;\n02:-:-:-:6 NOP;',j*1024,10240+j*1024))
 end
elseif tag=='swiglu' then
 params{{'param_G0',0x140},{'param_G1',0x144},{'param_U0',0x148},{'param_U1',0x14c},{'param_Y0',0x150},{'param_Y1',0x154},{'param_slot',0x158},{'param_NE0',0x160},{'param_NE1',0x164}}
 emit('<REGISTER_MAPPING>\n0 : tid\n1 : bid\n2 : i\n3 : slot\n4 : off\n5 : n\n6,7 : gp<0-1>\n8,9 : up<0-1>\n10,11 : yp<0-1>\n12,13 : np<0-1>\n14 : g\n15 : u\n16 : v\n17 : a\n18 : b\n19 : c\n20 : r\n21 : e\n22 : log2e\n23 : ln2hi\n24 : ln2lo\n25 : c720\n26 : c120\n27 : c24\n28 : c6\n29 : half\n30 : one\n31 : negone\n</REGISTER_MAPPING>')
 emit('--:-:6:-:1 S2R tid, SR_TID.X;\n--:-:6:-:2 S2R bid, SR_CTAID.X;\n20:-:-:-:6 ISCADD i, bid, tid, 0x8;')
 ins('MOV slot, param_slot;');ins('MOV np0, param_NE0;');ins('MOV np1, param_NE1;')
 ins('LOP.OR n, np0, np1;');ins('ISETP.NE.U32.AND P0, PT, n, RZ, PT;');ins('MOV32I n, 0xffffffff;')
 ins('NOP;');ins('NOP;');ins('NOP;')
 emit('--:-:1:-:2 @P0 LDG.E n, [np];\n01:-:-:-:6 ISETP.GE.U32.AND P5, PT, slot, n, PT;')
 ins('NOP;');ins('NOP;');ins('NOP;')
 ins('ISETP.GE.U32.OR P5, PT, i, 0x280, P5;');ins('NOP;');ins('NOP;');ins('NOP;');ins('@P5 EXIT;')
 ins('XMAD off, slot, 0x280, i;');ins('SHL off, off, 0x2;')
 ptr('gp','param_G0','param_G1','off');ptr('up','param_U0','param_U1','off');ptr('yp','param_Y0','param_Y1','off')
 emit('--:-:1:-:1 LDG.E.CI g, [gp];\n--:-:1:-:2 LDG.E.CI u, [up];')
 local constants={{'log2e',0x3fb8aa3b},{'ln2hi',0x3f317200},{'ln2lo',0x35bfbe8e},{'c720',0x3ab60b61},{'c120',0x3c088889},{'c24',0x3d2aaaab},{'c6',0x3e2aaaab},{'half',0x3f000000},{'one',0x3f800000},{'negone',0xbf800000}}
 for _,p in ipairs(constants) do ins(string.format('MOV32I %s, 0x%08x;',p[1],p[2])) end
 emit('01:-:-:-:6 FADD e, -g, RZ;')
 for _,s in ipairs{'FMNMX e, e, 87, PT;','FMNMX e, e, -87, !PT;','FMUL a, e, log2e;','FADD b, a, 12582912;','FADD b, b, -12582912;','FFMA c, b, -ln2hi, e;','FFMA c, b, -ln2lo, c;','FFMA a, c720, c, c120;','FFMA a, a, c, c24;','FFMA a, a, c, c6;','FFMA a, a, c, half;','FFMA a, a, c, one;','FFMA a, a, c, one;'} do ins(s) end
 emit('--:-:3:-:2 F2I.S32.F32 b, b;\n04:-:-:-:6 IADD b, b, 0x7f;')
 ins('SHL b, b, 0x17;');ins('FMUL e, a, b;');ins('FADD e, e, 1;')
 emit('--:-:3:-:2 MUFU.RCP r, e;\n04:-:-:-:6 FFMA a, e, r, negone;')
 ins('FADD.FTZ a, -a, -RZ;');ins('FFMA r, r, a, r;');ins('FMUL v, g, r;');ins('FMUL v, v, u;')
 emit('--:2:-:-:6 STG.E [yp], v;\n02:-:-:-:6 NOP;')
elseif tag=='weighted_sum' then
 params{{'param_X0',0x140},{'param_X1',0x144},{'param_W0',0x148},{'param_W1',0x14c},{'param_NE0',0x150},{'param_NE1',0x154},{'param_SH0',0x158},{'param_SH1',0x15c},{'param_SW0',0x160},{'param_SW1',0x164},{'param_Y0',0x168},{'param_Y1',0x16c}}
 emit('<REGISTER_MAPPING>\n0 : tid\n1 : bid\n2 : i\n3 : off\n4 : n\n5 : j\n6,7 : xp<0-1>\n8,9 : wp<0-1>\n10,11 : np<0-1>\n12,13 : sp<0-1>\n14,15 : ap<0-1>\n16,17 : yp<0-1>\n18 : x\n19 : w\n20 : sum\n21 : tmp\n</REGISTER_MAPPING>')
 emit('--:-:6:-:1 S2R tid, SR_TID.X;\n--:-:6:-:2 S2R bid, SR_CTAID.X;\n20:-:-:-:6 ISCADD i, bid, tid, 0x8;')
 ins('ISETP.GE.U32.AND P5, PT, i, 0xa00, PT;');ins('NOP;');ins('NOP;');ins('NOP;');ins('@P5 EXIT;')
 ins('SHL off, i, 0x2;');ptr('xp','param_X0','param_X1','off');ptr('yp','param_Y0','param_Y1','off')
 ins('MOV wp0, param_W0;');ins('MOV wp1, param_W1;');ins('MOV np0, param_NE0;');ins('MOV np1, param_NE1;')
 emit('--:-:1:-:2 LDG.E n, [np];\n01:-:-:-:6 ISETP.EQ.U32.AND P0, PT, n, RZ, PT;')
 ins('MOV sum, RZ;');ins('MOV j, RZ;');ins('NOP;');ins('@P0 BRA SHARED;');emit('LOOP:')
 emit('--:-:1:-:1 LDG.E.CI x, [xp];\n--:-:1:-:2 LDG.E.CI w, [wp];\n01:-:-:-:6 FFMA sum, w, x, sum;')
 ins('IADD xp0.CC, xp0, 0x2800;');ins('IADD.X xp1, xp1, RZ;');ins('IADD wp0.CC, wp0, 0x4;');ins('IADD.X wp1, wp1, RZ;')
 ins('IADD j, j, 0x1;');ins('ISETP.LT.U32.AND P0, PT, j, n, PT;');ins('NOP;');ins('NOP;');ins('NOP;');ins('@P0 BRA LOOP;')
 emit('SHARED:');ins('MOV sp0, param_SH0;');ins('MOV sp1, param_SH1;');ins('LOP.OR tmp, sp0, sp1;')
 ins('ISETP.EQ.U32.AND P0, PT, tmp, RZ, PT;');ins('NOP;');ins('NOP;');ins('NOP;');ins('@P0 BRA STORE;')
 ptr('sp','param_SH0','param_SH1','off');ins('MOV ap0, param_SW0;');ins('MOV ap1, param_SW1;')
 emit('--:-:1:-:1 LDG.E.CI x, [sp];\n--:-:1:-:2 LDG.E.CI w, [ap];\n01:-:-:-:6 FFMA sum, w, x, sum;')
 emit('STORE:\n--:2:-:-:6 STG.E [yp], sum;\n02:-:-:-:6 NOP;')
else error('unknown MTP helper '..tag) end
emit('--:-:-:-:f EXIT;')
io.write(table.concat(out))
