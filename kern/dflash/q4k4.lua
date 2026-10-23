-- Emit PTX for the draft's four-position Q4_K matrix.
--   lua dflash/q4k4.lua OUT.ptx
-- One warp owns four rows and shares every activation load across them. Lane
-- l covers one 32-bit word of each 256-weight block's 128 code bytes: low
-- nibbles of sub-block 2j and high nibbles of sub-block 2j+1 (j = l/8), four
-- elements each. Per sub-block the integer codes are dotted with x in FP32 and
-- the affine minimum uses that sub-block's sum of x, as in ggml's Q4_K:
--   y = sum_sub d*sc*dot(code,x) - dmin*m*sum(x)
-- Accumulation order differs from the per-element FP32 reference kernel.
local out=assert(arg[1],'output path')
local o={}
local function e(s)o[#o+1]=s end
e('.version 6.0')
e('.target sm_52')
e('.address_size 64')
e('')
e('.visible .entry df4_q4k_matrix(.param .u64 pw,.param .u64 px,.param .u64 py,')
e('    .param .u32 prows,.param .u32 pk,.param .u32 pstride)')
e('{')
e('  .reg .pred %valid<4>,%more,%lead;')
e('  .reg .b32 %rtid,%lane,%rwarp,%rcta,%rows,%k,%stride,%r0,%blocks,%b,%j,%e0,%sub;')
e('  .reg .b16 %hh<2>;\n  .reg .b32 %t<8>,%h<4>,%q,%sa,%ma,%sb,%mb,%s0,%s1,%s2,%dm;')
e('  .reg .b64 %w,%x,%y,%rowp<4>,%xa,%xb,%addr,%off;')
e('  .reg .f32 %d,%dmin,%fsa,%fma,%fsb,%fmb,%lo<4>,%hi<4>,%da,%db,%dd;')
for p=0,3 do
 e(string.format('  .reg .f32 %%xa%d_<4>,%%xb%d_<4>,%%sumA%d,%%sumB%d,%%dotA%d,%%dotB%d;',p,p,p,p,p,p))
end
for r=0,3 do e(string.format('  .reg .f32 %%acc%d_<4>;',r))end
e('  ld.param.u64 %w,[pw];ld.param.u64 %x,[px];ld.param.u64 %y,[py];')
e('  ld.param.u32 %rows,[prows];ld.param.u32 %k,[pk];ld.param.u32 %stride,[pstride];')
e('  cvta.to.global.u64 %w,%w;cvta.to.global.u64 %x,%x;cvta.to.global.u64 %y,%y;')
e('  mov.u32 %rtid,%tid.x;and.b32 %lane,%rtid,31;shr.u32 %rwarp,%rtid,5;mov.u32 %rcta,%ctaid.x;')
e('  shl.b32 %r0,%rcta,3;add.u32 %r0,%r0,%rwarp;shl.b32 %r0,%r0,2;')
e('  setp.ge.u32 %more,%r0,%rows;@%more bra DONE;')
for r=0,3 do
 e(string.format('  add.u32 %%t0,%%r0,%d;setp.lt.u32 %%valid%d,%%t0,%%rows;',r,r))
 e(string.format('  @%%valid%d mul.wide.u32 %%off,%%t0,%%stride;@!%%valid%d mul.wide.u32 %%off,%%r0,%%stride;',r,r))
 e(string.format('  add.u64 %%rowp%d,%%w,%%off;',r))
 for p=0,3 do e(string.format('  mov.f32 %%acc%d_%d,0f00000000;',r,p))end
end
e('  shr.u32 %j,%lane,3;and.b32 %e0,%lane,7;shl.b32 %e0,%e0,2;')
-- Sub-block scale/min decode depends only on j: j<2 uses the direct six-bit
-- fields, j>=2 the split fields. sub = 2j and 2j+1.
e('  shl.b32 %sub,%j,1;')
e('  shr.u32 %blocks,%k,8;mov.u32 %b,0;')
e('LOOP:')
-- Activations: columns b*256 + 64j + e0 (sub 2j) and +32 (sub 2j+1).
e('  shl.b32 %t0,%b,8;shl.b32 %t1,%j,6;add.u32 %t0,%t0,%t1;add.u32 %t0,%t0,%e0;')
for p=0,3 do
 e(string.format('  mad.lo.u32 %%t1,%%k,%d,%%t0;mul.wide.u32 %%off,%%t1,4;add.u64 %%xa,%%x,%%off;',p))
 e(string.format('  ld.global.nc.v4.f32 {%%xa%d_0,%%xa%d_1,%%xa%d_2,%%xa%d_3},[%%xa];',p,p,p,p))
 e(string.format('  ld.global.nc.v4.f32 {%%xb%d_0,%%xb%d_1,%%xb%d_2,%%xb%d_3},[%%xa+128];',p,p,p,p))
 e(string.format('  add.f32 %%sumA%d,%%xa%d_0,%%xa%d_1;add.f32 %%sumA%d,%%sumA%d,%%xa%d_2;add.f32 %%sumA%d,%%sumA%d,%%xa%d_3;',p,p,p,p,p,p,p,p,p))
 e(string.format('  add.f32 %%sumB%d,%%xb%d_0,%%xb%d_1;add.f32 %%sumB%d,%%sumB%d,%%xb%d_2;add.f32 %%sumB%d,%%sumB%d,%%xb%d_3;',p,p,p,p,p,p,p,p,p))
end
e('  mul.lo.u32 %t2,%b,144;cvt.u64.u32 %off,%t2;')
for r=0,3 do
 e(string.format('  @!%%valid%d bra SKIP%d;',r,r))
 e(string.format('  add.u64 %%addr,%%rowp%d,%%off;',r))
 e('  ld.global.nc.v4.u32 {%dm,%s0,%s1,%s2},[%addr];')
 e('  shl.b32 %t3,%lane,2;cvt.u64.u32 %xb,%t3;add.u64 %xb,%addr,%xb;ld.global.nc.u32 %q,[%xb+16];')
 e('  mov.b32 {%hh0,%hh1},%dm;cvt.f32.f16 %d,%hh0;cvt.f32.f16 %dmin,%hh1;')
 -- scales: s[i] bytes 0..11 in s0,s1,s2
 e('  setp.lt.u32 %lead,%j,2;')
 e('  @%lead shl.b32 %t4,%sub,3;')            -- direct: byte sub of s0 (sc), of s1 (m)
 e('  @%lead bfe.u32 %sa,%s0,%t4,6;@%lead bfe.u32 %ma,%s1,%t4,6;')
 e('  @%lead add.u32 %t5,%t4,8;@%lead bfe.u32 %sb,%s0,%t5,6;@%lead bfe.u32 %mb,%s1,%t5,6;')
 -- split: t = sub-4; sc=(s2.byte t &15)|((s0.byte t>>6)<<4); m=(s2.byte t>>4)|((s1.byte t>>6)<<4)
 e('  @!%lead add.u32 %t4,%sub,-4;@!%lead shl.b32 %t4,%t4,3;')
 for _,pair in ipairs({{'sa','ma','%t4'},{'sb','mb','%t5'}})do
  if pair[3]=='%t5' then e('  @!%lead add.u32 %t5,%t4,8;')end
  e(string.format('  @!%%lead bfe.u32 %%t6,%%s2,%s,4;@!%%lead add.u32 %%t7,%s,6;@!%%lead bfe.u32 %%h2,%%s0,%%t7,2;',pair[3],pair[3]))
  e(string.format('  @!%%lead shl.b32 %%h2,%%h2,4;@!%%lead or.b32 %%%s,%%t6,%%h2;',pair[1]))
  e(string.format('  @!%%lead add.u32 %%t6,%s,4;@!%%lead bfe.u32 %%t6,%%s2,%%t6,4;@!%%lead bfe.u32 %%h2,%%s1,%%t7,2;',pair[3]))
  e(string.format('  @!%%lead shl.b32 %%h2,%%h2,4;@!%%lead or.b32 %%%s,%%t6,%%h2;',pair[2]))
 end
 e('  cvt.rn.f32.u32 %fsa,%sa;cvt.rn.f32.u32 %fma,%ma;cvt.rn.f32.u32 %fsb,%sb;cvt.rn.f32.u32 %fmb,%mb;')
 e('  mul.f32 %da,%d,%fsa;mul.f32 %db,%d,%fsb;mul.f32 %fma,%dmin,%fma;mul.f32 %fmb,%dmin,%fmb;')
 for i=0,3 do
  e(string.format('  bfe.u32 %%t6,%%q,%d,4;cvt.rn.f32.u32 %%lo%d,%%t6;bfe.u32 %%t6,%%q,%d,4;cvt.rn.f32.u32 %%hi%d,%%t6;',8*i,i,8*i+4,i))
 end
 for p=0,3 do
  e(string.format('  mul.f32 %%dotA%d,%%lo0,%%xa%d_0;fma.rn.f32 %%dotA%d,%%lo1,%%xa%d_1,%%dotA%d;fma.rn.f32 %%dotA%d,%%lo2,%%xa%d_2,%%dotA%d;fma.rn.f32 %%dotA%d,%%lo3,%%xa%d_3,%%dotA%d;',p,p,p,p,p,p,p,p,p,p,p))
  e(string.format('  mul.f32 %%dotB%d,%%hi0,%%xb%d_0;fma.rn.f32 %%dotB%d,%%hi1,%%xb%d_1,%%dotB%d;fma.rn.f32 %%dotB%d,%%hi2,%%xb%d_2,%%dotB%d;fma.rn.f32 %%dotB%d,%%hi3,%%xb%d_3,%%dotB%d;',p,p,p,p,p,p,p,p,p,p,p))
  e(string.format('  fma.rn.f32 %%acc%d_%d,%%da,%%dotA%d,%%acc%d_%d;neg.f32 %%dd,%%fma;fma.rn.f32 %%acc%d_%d,%%dd,%%sumA%d,%%acc%d_%d;',r,p,p,r,p,r,p,p,r,p))
  e(string.format('  fma.rn.f32 %%acc%d_%d,%%db,%%dotB%d,%%acc%d_%d;neg.f32 %%dd,%%fmb;fma.rn.f32 %%acc%d_%d,%%dd,%%sumB%d,%%acc%d_%d;',r,p,p,r,p,r,p,p,r,p))
 end
 e(string.format('SKIP%d:',r))
end
e('  add.u32 %b,%b,1;setp.lt.u32 %more,%b,%blocks;@%more bra LOOP;')
-- Warp reduction and stores: y[p*rows+row].
for r=0,3 do for p=0,3 do
 for _,delta in ipairs({16,8,4,2,1})do
  e(string.format('  shfl.sync.down.b32 %%dd,%%acc%d_%d,%d,31,0xffffffff;add.f32 %%acc%d_%d,%%acc%d_%d,%%dd;',r,p,delta,r,p,r,p))
 end
end end
e('  setp.eq.u32 %lead,%lane,0;@!%lead bra DONE;')
for r=0,3 do
 e(string.format('  @!%%valid%d bra STORED%d;',r,r))
 for p=0,3 do
  e(string.format('  add.u32 %%t0,%%r0,%d;mad.lo.u32 %%t0,%%rows,%d,%%t0;mul.wide.u32 %%off,%%t0,4;add.u64 %%addr,%%y,%%off;st.global.f32 [%%addr],%%acc%d_%d;',r,p,r,p))
 end
 e(string.format('STORED%d:',r))
end
e('DONE:')
e('  ret;')
e('}')
local f=assert(io.open(out,'w'));f:write(table.concat(o,'\n')..'\n');f:close()
