-- Preserve the scalar per-32 activation quantization before a float GEMM.
-- QACT_LAYOUT=s: [channel][ldx]; p: packed consecutive vectors of 32 values.
-- One thread owns a block and retains it while computing max/scale/codes.
-- q*scale is written as F32; the following GEMM can reuse its existing tile.
local layout=os.getenv("QACT_LAYOUT") or "s"
assert(layout=="s" or layout=="p")
local function emit(s) io.write(s,"\n") end
emit([[# Kernel: qact
<CONSTANT_MAPPING>
    param_X0 : c[0x0][0x140]
    param_X1 : c[0x0][0x144]
    param_Y0 : c[0x0][0x148]
    param_Y1 : c[0x0][0x14c]
    param_n  : c[0x0][0x150]
    param_lg : c[0x0][0x154]
</CONSTANT_MAPPING>
<REGISTER_MAPPING>
    0-31 : x<00-31>
    32 : tid
    33 : bid
    34 : t
    35 : stride
    36 : g
    37 : r
    38 : err
    39 : mneg1
    40 : c127
    41 : offset
    42,43 : ip<0-1>
    44,45 : op<0-1>
</REGISTER_MAPPING>
--:-:1:-:1      S2R tid, SR_TID.X;
--:-:2:-:2      S2R bid, SR_CTAID.X;
03:-:-:-:6      ISCADD bid, bid, tid, 0x8;
--:-:-:-:6      MOV t, param_n;
--:-:-:-:6      ISETP.LT.U32.AND P0, PT, bid, t, PT;
--:-:-:-:6      MOV32I mneg1, 0xbf800000;
--:-:-:-:6      MOV32I c127, 0x3c010204;
--:-:-:-:6      MOV32I g, 0x0d800000;
--:-:-:-:6      SEL bid, bid, RZ, P0;
]])
if layout=="s" then
  emit([[--:-:-:-:6      MOV t, param_lg;
--:-:-:-:6      MOV32I stride, 0x1;
--:-:-:-:6      SHL stride, stride, t;
--:-:-:-:6      IADD t, stride, -0x1;
--:-:-:-:6      LOP.AND t, bid, t;
--:-:-:-:6      IADD offset, bid, -t;
--:-:-:-:6      ISCADD offset, offset, t, 0x5;
--:-:-:-:6      SHL offset, offset, 0x2;
--:-:-:-:6      SHL stride, stride, 0x2;
]])
else emit([[--:-:-:-:6      SHL offset, bid, 0x7;
--:-:-:-:6      MOV32I stride, 0x4;
]]) end
emit([[--:-:-:-:6      IADD ip0.CC, offset, param_X0;
--:-:-:-:6      IADD.X ip1, RZ, param_X1;
--:-:-:-:6      IADD op0.CC, offset, param_Y0;
--:-:-:-:6      IADD.X op1, RZ, param_Y1;
]])
-- Keep the 64-bit address registers immutable across the whole memory group.
-- The four legal slab strides use immediate offsets; packed blocks need one.
local function memory_group(load)
  local prefix=load and "LOAD" or "STORE"
  local function body(step)
    for i=0,31 do
      if load then emit(string.format("--:-:2:-:1      LDG.E x%02d, [ip+0x%x];",i,step*i))
      else emit(string.format("--:1:-:-:1  @P0 STG.E [op+0x%x], x%02d;",step*i,i)) end
    end
  end
  if layout=="p" then body(4); return end
  emit("--:-:-:-:6      MOV t, param_lg;")
  for lg=7,9 do
    emit(string.format("--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, t, 0x%x, PT;",lg))
    emit(string.rep("--:-:-:-:6      NOP;\n",3).."--:-:-:-:6  @P1 BRA "..prefix..lg..";")
  end
  body(4096);emit("--:-:-:-:6      BRA "..prefix.."_DONE;")
  for lg=7,9 do
    emit(prefix..lg..":");body(4*(1<<lg));emit("--:-:-:-:6      BRA "..prefix.."_DONE;")
  end
  emit(prefix.."_DONE:")
end
memory_group(true)
emit("02:-:-:-:6      NOP;")
for i=0,31 do emit(string.format("--:-:-:-:6      FMNMX g, g, |x%02d|, !PT;",i)) end
emit([[--:-:3:-:2      MUFU.RCP r, g;
04:-:-:-:6      FFMA err, g, r, mneg1;
--:-:-:-:6      FADD.FTZ err, -err, -RZ;
--:-:-:-:6      FFMA r, r, err, r;
--:-:-:-:6      FMUL r, r, 127;
--:-:-:-:6      FMUL g, g, c127;
]])
for i=0,31 do emit(string.format("--:-:-:-:1      FMUL x%02d, x%02d, r;",i,i)) end
for i=0,31 do emit(string.format("--:-:3:-:1      F2I.S32.F32 x%02d, x%02d;",i,i)) end
emit("04:-:-:-:6      NOP;")
for i=0,31 do emit(string.format("--:-:3:-:1      I2F.F32.S32 x%02d, x%02d;",i,i)) end
emit("04:-:-:-:6      NOP;")
for i=0,31 do emit(string.format("--:-:-:-:1      FMUL x%02d, x%02d, g;",i,i)) end
memory_group(false)
emit("01:-:-:-:6      MEMBAR.GL;\n--:-:-:-:f      EXIT;")
