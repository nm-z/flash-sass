#!/usr/bin/env lua5.4
-- CA_OP=kv|idx|sel|app|attn lua5.4 cattn.lua > cattn_<op>.sass: the attention side of a prefill chunk (QSA, ref/qsa.h) as die ops
-- (die.pl, 'cta' units of 256 threads), on the chunk program's k-major slabs X[row][ldx] (a position per column). The
-- attention layer's owner die (the one with its decode KV cache) projects K / V / the indexer (GEMMs) and runs kv, idx and
-- sel; every die with query heads gets the owner's chunk K / V and selections through one exchange, appends the K / V of its
-- heads' KV head to its own copy of the prompt's cache (app) and runs attn for its heads. Not bit-exact with the decode ops
-- (MUFU.RSQ / RCP / EX2, its own summation orders).
--
-- Chunk tables: CS / SN hold the RoPE cos / sin rows of positions t0 - 3 .. t0 + npos - 1 (row p + 3 = position t0 + p).
-- The chunk exchange of the owner, per position p: CKV [p][2][512] (KV head h: K codes 256 then V codes 256, u8 = int8 + 128),
-- CKS [p][2][16] f32 (h: the 8 K scales then the 8 V scales), CSEL [p][512] u16 (the selected blocks, ascending; written only
-- where nv > nblk). A die's prompt cache of one KV head (app): R [T][512] (K codes then V codes, u8 = int8 + 128) and
-- RS [T][16] f32 (K scales then V scales), T the position.
--
--   kv    CTA p: the two KV heads' K rows (LK [512][ldx]) RMS-normalised (KN, eps), RoPE (dims < 64, pairs (j, j + 32)), and the
--         V rows (LV) as projected, quantised in blocks of 32 (gmax = max(max |x|, 2^-100), q = rint(x 127 / gmax), scale =
--         gmax / 127) into the decode cache (KQ / VQ int8 [T][2][256], KS / VS fp16 [T][2][8], T = t0 + p) and CKV / CKS.
--         0x140 LK  0x148 LV  0x150 KN  0x158 CS  0x160 SN  0x168 KQ  0x170 KS  0x178 VQ  0x180 VS  0x188 CKV  0x190 CKS
--         0x198 t0  0x19c lgx  0x1a0 eps.  units = npos
--   idx   CTA p: the indexer of position t = t0 + p from LI [640][ldx] (iq 4 x 128 rows, then the raw key's 128): warps 0..3
--         iq head w (RMS128 * QN, RoPE) into IQ [p][512] f32; warp 4, when t completes block b = t >> 2 inside the chunk
--         (p >= 3, t & 3 == 3): the pooled key (mean of the 4 raw keys, RMS128 * KN, RoPE at 4 b) into the decode pool PKT
--         (idxqk.sass's transposed fp16 layout) and POOL [128][8192] f32 (dim-major); warp 5 of unit 0: the block that
--         straddles t0 (its first raw keys from RING [4][128], slot pos & 3), then RING <- the chunk's last four raw keys.
--         0x140 LI  0x148 QN  0x150 KN  0x158 CS  0x160 SN  0x168 IQ  0x170 RING  0x178 PKT  0x180 POOL  0x188 t0  0x18c npos
--         0x190 lgx  0x194 eps.  units = npos
--   sel   CTA p, only where nv = (t + 1) >> 2 > nblk: score[b] = sum_h relu(iq_h . pool_b) for b < nv into shared memory, the
--         nblk best (radix select over the score bits, 8 + 8 + 8 + 7 bits; among equal scores the lower blocks) written in
--         ascending order to SEL [p][512] u16. nv <= 8192. 0x140 IQ  0x148 POOL  0x150 SEL  0x158 t0  0x15c nblk.  units = npos
--   app   the chunk's K / V of KV head hk from the exchange into the die's prompt cache: R[t0 + p] <- CKV[p][hk],
--         RS[t0 + p] <- CKS[p][hk]. 0x140 CKV  0x148 CKS  0x150 R  0x158 RS  0x160 hk  0x164 t0  0x168 npos.  units = npos / 4
--   attn  CTA p: query heads hq0 .. hq0 + nhh - 1 (nhh <= 6, one KV head, local rows of LQ [nh 512][ldx]: q 256 then gate 256
--         per head) of position t over the keys of qsa_gather_gpu (the blocks 0 .. nv - 1, or SEL [p] when nv > nblk, then the
--         tail block nv): q RMS-normalised (QN, eps), RoPE, times 1/16; tiles of 64 keys, 4 threads per key (64 dims each),
--         online softmax; out = o / den sigmoid(gate) into LAO [nh 256][ldx] rows (hq0 + h) 256 + d.
--         0x140 LQ  0x148 QN  0x150 CS  0x158 SN  0x160 R  0x168 RS  0x170 SEL  0x178 LAO  0x180 hq0  0x184 nhh  0x188 t0
--         0x18c nblk  0x190 lgx  0x194 eps.  units = npos
--
-- (ported from the Ruby generator cattn.rb, output byte-identical apart from the header line)

local OP = os.getenv("CA_OP") or "attn"   -- ENV.fetch('CA_OP', 'attn')

-- Ruby's abort: the message and a newline on stderr, then exit status 1
local function abort(msg)
  io.stderr:write(msg, "\n")
  os.exit(1)
end

if OP ~= "kv" and OP ~= "idx" and OP ~= "sel" and OP ~= "app" and OP ~= "attn" then
  abort("CA_OP must be kv, idx, sel, app or attn")
end

local out = {}   -- Ruby's out (a list of the emitted pieces; joined where the text is edited)
local function emit(s)   -- Ruby's emit = ->(s) { out << s << "\n" }
  out[#out + 1] = s .. "\n"
end

-- Ruby's out.gsub! / out.sub! on the whole text so far: fn maps the text to the new text
local function edit_out(fn)
  out = { fn(table.concat(out)) }
end

-- Ruby's str.sub!(old, new) (the first occurrence) or str.gsub!(old, new) (all of them) for a plain string old
local function plain_sub(text, old, new, all)
  local parts = {}
  local pos = 1
  while true do
    local s, e = text:find(old, pos, true)
    if not s then break end
    parts[#parts + 1] = text:sub(pos, s - 1)
    parts[#parts + 1] = new
    pos = e + 1
    if not all then break end
  end
  parts[#parts + 1] = text:sub(pos)
  return table.concat(parts)
end

local function out_sub(old, new)
  edit_out(function(text) return plain_sub(text, old, new, false) end)
end

local function out_gsub(old, new)
  edit_out(function(text) return plain_sub(text, old, new, true) end)
end

-- Ruby's <<~ heredoc: the least indentation of the non-blank lines is removed from every line
local function dedent(text)
  local lines = {}
  for line in text:gmatch("([^\n]*)\n") do
    lines[#lines + 1] = line
  end
  local width = nil
  for _, line in ipairs(lines) do
    if line:find("%S") then
      local ws = #line:match("^ *")
      if width == nil or ws < width then width = ws end
    end
  end
  width = width or 0
  local res = {}
  for i, line in ipairs(lines) do
    local ws = #line:match("^ *")
    res[i] = line:sub(math.min(width, ws) + 1) .. "\n"
  end
  return table.concat(res)
end

-- Ruby's #{...} in a heredoc: the caller computes each value (Ruby evaluates it inline) and names it in the table
local function interp(text, vals)
  return (text:gsub("#{([%w_]+)}", function(name)
    local v = vals[name]
    assert(v ~= nil, "cattn: no value for #{" .. name .. "}")
    return v
  end))
end

local function heredoc(text, vals)
  return interp(dedent(text), vals or {})
end

-- Ruby's x.to_s(16) and format('%x') for a non-negative integer
local function hx(n)
  return string.format("%x", n)
end

-- the names prefix0 .. prefix(n - 1) (Ruby's (0...n).map { |h| "#{prefix}#{h}" })
local function names(prefix, n)
  local t = {}
  for i = 0, n - 1 do t[#t + 1] = prefix .. i end
  return t
end

local L2E = 0x3fb8aa3b
local POOL_STRIDE = 8192 * 4   -- bytes per dim row of POOL
local NH = 6   -- the attn op's query heads per CTA (Ruby: NH = 6)

-- d = a * b + c (32 x 32 -> low 32); x is a temporary, d must differ from a (the same three XMADs as gen_gemm's xmad_lo)
local function mul(d, a, b, c, x)
  return string.format("--:-:-:-:6      XMAD.MRG %s, %s, %s.H1, RZ;\n", x, a, b)
      .. string.format("--:-:-:-:6      XMAD %s, %s, %s, %s;\n", d, a, b, c)
      .. string.format("--:-:-:-:6      XMAD.PSL.CBCC %s, %s.H1, %s.H1, %s;\n", d, a, x, d)
end

-- warp butterfly over xor offs of each register (op FADD / FMNMX); temps distinct, barrier bar
local function bfly(regs, temps, offs, op, bar)
  bar = bar or 6
  local s = {}
  for _, x in ipairs(offs) do
    for i, r in ipairs(regs) do
      local wr = 1
      if i == #regs then wr = 2 end
      s[#s + 1] = string.format("--:-:%d:-:%d      SHFL.BFLY PT, %s, %s, 0x%x, 0x1f;\n", bar, wr, temps[i], r, x)
    end
    for i, r in ipairs(regs) do
      local w = "--"
      if i == 1 then w = string.format("%02x", 1 << (bar - 1)) end
      local st = 1
      if i == #regs then st = 6 end
      if op == "max" then
        s[#s + 1] = string.format("%s:-:-:-:%d      FMNMX %s, %s, %s, !PT;\n", w, st, r, r, temps[i])
      else
        s[#s + 1] = string.format("%s:-:-:-:%d      FADD %s, %s, %s;\n", w, st, r, r, temps[i])
      end
    end
  end
  return table.concat(s)
end

-- 2^(x log2 e) in place for the registers xs (MUFU.EX2 on barrier bar)
local function ex2(xs, l2e, bar)
  bar = bar or 3
  local s = {}
  for i, x in ipairs(xs) do
    local st = 1
    if i == #xs then st = 6 end
    s[#s + 1] = string.format("--:-:-:-:%d      FMUL %s, %s, %s;\n", st, x, x, l2e)
  end
  for i, x in ipairs(xs) do
    local st = 1
    if i == #xs then st = 6 end
    s[#s + 1] = string.format("--:-:-:-:%d      RRO.EX2 %s, %s;\n", st, x, x)
  end
  for _, x in ipairs(xs) do
    s[#s + 1] = string.format("--:-:%d:-:1      MUFU.EX2 %s, %s;\n", bar, x, x)
  end
  s[#s + 1] = string.format("%02x:-:-:-:6      NOP;\n", 1 << (bar - 1))
  return table.concat(s)
end

local PARAMS = ({
  kv   = { "LK0", "LK1", "LV0", "LV1", "KN0", "KN1", "CS0", "CS1", "SN0", "SN1", "KQ0", "KQ1", "KS0", "KS1", "VQ0", "VQ1", "VS0", "VS1", "CKV0", "CKV1", "CKS0", "CKS1", "t0", "lgx", "eps" },
  idx  = { "LI0", "LI1", "QN0", "QN1", "KN0", "KN1", "CS0", "CS1", "SN0", "SN1", "IQ0", "IQ1", "RING0", "RING1", "PKT0", "PKT1", "POOL0", "POOL1", "t0", "npos", "lgx", "eps" },
  sel  = { "IQ0", "IQ1", "POOL0", "POOL1", "SEL0", "SEL1", "t0", "nblk" },
  app  = { "CKV0", "CKV1", "CKS0", "CKS1", "R0", "R1", "RS0", "RS1", "hk", "t0", "npos" },
  attn = { "LQ0", "LQ1", "QN0", "QN1", "CS0", "CS1", "SN0", "SN1", "R0", "R1", "RS0", "RS1", "SEL0", "SEL1", "LAO0", "LAO1", "hq0", "nhh", "t0", "nblk", "lgx", "eps" },
})[OP]

emit("# Kernel: cattn_" .. OP .. "  (generated by CA_OP=" .. OP .. " cattn.lua; do not edit)")
emit("<CONSTANT_MAPPING>")
for i, p in ipairs(PARAMS) do
  emit(string.format("    param_%-6s : c[0x0][0x%x]", p, 0x140 + 4 * (i - 1)))
end
emit("</CONSTANT_MAPPING>")

-- ---------------------------------------------------------------------------------------------------------------------------
if OP == "kv" then
emit(heredoc([==[
    <REGISTER_MAPPING>
        0     : tid
        1     : p
        2     : lane
        3     : w
        4     : lgx
        5     : t
        6     : st
        7     : tt
        8-11  : x<0-3>
        12-15 : y<0-3>
        16-19 : ss<0-3>
        20-23 : tm<0-3>
        24-31 : a<0-7>
        32    : kn
        33    : cs
        34    : sn
        35    : one
        36,37 : k0p<0-1>
        38,39 : k1p<0-1>
        40,41 : v0p<0-1>
        42,43 : v1p<0-1>
        44,45 : cp<0-1>
        46,47 : qp<0-1>
        48,49 : sp<0-1>
        50,51 : bp<0-1>
        52,53 : bsp<0-1>
        54-57 : g<0-3>
        58-61 : qi<0-3>
        62-65 : sc<0-3>
        66-69 : sh<0-3>
        70    : tiny
        71    : big
        72    : r127
        73    : T
        74    : u
        75    : yp
    </REGISTER_MAPPING>
]==]))
-- thread d = tid: x0 / x1 = K rows d / 256 + d (KV heads 0 / 1), x2 / x3 = V rows d / 256 + d, column p
emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R p, SR_CTAID.X;
    --:-:-:-:6      MOV lgx, param_lgx;
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 w, tid, 0x5;
    --:-:-:-:6      SHL t, tid, lgx;
    --:-:-:-:6      IADD t, t, p;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      MOV32I one, 0x1;
    --:-:-:-:6      IADD st, lgx, 0xa;
    --:-:-:-:6      SHL st, one, st;
    --:-:-:-:6      IADD k0p0.CC, t, param_LK0;
    --:-:-:-:6      IADD.X k0p1, RZ, param_LK1;
    --:-:-:-:6      IADD k1p0.CC, k0p0, st;
    --:-:-:-:6      IADD.X k1p1, k0p1, RZ;
    --:-:-:-:6      IADD v0p0.CC, t, param_LV0;
    --:-:-:-:6      IADD.X v0p1, RZ, param_LV1;
    --:-:-:-:6      IADD v1p0.CC, v0p0, st;
    --:-:-:-:6      IADD.X v1p1, v0p1, RZ;
    --:-:-:-:6      SHL tt, tid, 0x2;
    --:-:-:-:6      IADD cp0.CC, tt, param_KN0;
    --:-:-:-:6      IADD.X cp1, RZ, param_KN1;
    --:-:2:-:1      LDG.E x0, [k0p];
    --:-:2:-:1      LDG.E x1, [k1p];
    --:-:2:-:1      LDG.E x2, [v0p];
    --:-:2:-:1      LDG.E x3, [v1p];
    --:-:2:-:1      LDG.E kn, [cp];
    --:-:-:-:6      LOP32I.AND tt, tid, 0x1f;
    --:-:-:-:6      ISCADD tt, p, tt, 0x5;
    --:-:-:-:6      IADD tt, tt, 0x60;
    --:-:-:-:6      SHL tt, tt, 0x2;
    --:-:-:-:6      IADD sp0.CC, tt, param_CS0;
    --:-:-:-:6      IADD.X sp1, RZ, param_CS1;
    --:-:-:-:6      IADD bp0.CC, tt, param_SN0;
    --:-:-:-:6      IADD.X bp1, RZ, param_SN1;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, tid, 0x40, PT;
    --:-:-:-:6      ISETP.LT.U32.AND P2, PT, tid, 0x20, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, lane, RZ, PT;
    --:-:-:-:6      MOV cs, RZ;
    --:-:-:-:6      MOV sn, RZ;
    --:-:2:-:1  @P1 LDG.E cs, [sp];
    --:-:2:-:2  @P1 LDG.E sn, [bp];
    02:-:-:-:6      FMUL ss0, x0, x0;
    --:-:-:-:6      FMUL ss1, x1, x1;
]==]))
emit(bfly({ "ss0", "ss1" }, { "tm0", "tm1" }, { 16, 8, 4, 2, 1 }, "sum"))
emit(heredoc([==[
    --:-:-:-:6      SHL tt, w, 0x2;
    --:1:-:-:1  @P0 STS [tt], ss0;
    --:1:-:-:1  @P0 STS [tt+0x20], ss1;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:2:-:1      LDS.U.128 a0, [RZ];
    --:-:2:-:1      LDS.U.128 a4, [RZ+0x10];
    --:-:2:-:1      LDS.U.128 ss0, [RZ+0x20];
    --:-:2:-:2      LDS.U.128 tm0, [RZ+0x30];
    02:-:-:-:6      FADD a0, a0, a1;
    --:-:-:-:6      FADD a0, a0, a2;
    --:-:-:-:6      FADD a0, a0, a3;
    --:-:-:-:6      FADD a0, a0, a4;
    --:-:-:-:6      FADD a0, a0, a5;
    --:-:-:-:6      FADD a0, a0, a6;
    --:-:-:-:6      FADD a0, a0, a7;
    --:-:-:-:6      FADD ss0, ss0, ss1;
    --:-:-:-:6      FADD ss0, ss0, ss2;
    --:-:-:-:6      FADD ss0, ss0, ss3;
    --:-:-:-:6      FADD ss0, ss0, tm0;
    --:-:-:-:6      FADD ss0, ss0, tm1;
    --:-:-:-:6      FADD ss0, ss0, tm2;
    --:-:-:-:6      FADD ss0, ss0, tm3;
    --:-:-:-:6      FMUL a0, a0, 0.00390625;
    --:-:-:-:6      FMUL ss0, ss0, 0.00390625;
    --:-:-:-:6      FADD a0, a0, param_eps;
    --:-:-:-:6      FADD ss0, ss0, param_eps;
    --:-:3:-:1      MUFU.RSQ a0, a0;
    --:-:3:-:1      MUFU.RSQ ss0, ss0;
    04:-:-:-:6      FMUL y0, x0, a0;
    --:-:-:-:6      FMUL y1, x1, ss0;
    --:-:-:-:6      FMUL y0, y0, kn;
    --:-:-:-:6      FMUL y1, y1, kn;
    --:-:-:-:6      SHL tt, tid, 0x2;
    --:1:-:-:1      STS [tt+0x100], y0;
    --:1:-:-:1      STS [tt+0x500], y1;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      LOP32I.XOR u, tid, 0x20;
    --:-:-:-:6      SHL u, u, 0x2;
    --:-:2:-:1  @P1 LDS.U a0, [u+0x100];
    --:-:2:-:1  @P1 LDS.U a1, [u+0x500];
    02:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6  @P2 FADD sn, RZ, -sn;
    --:-:-:-:6      FMUL tm0, y0, cs;
    --:-:-:-:6      FMUL tm1, y1, cs;
    --:-:-:-:6  @P1 FFMA y0, a0, sn, tm0;
    --:-:-:-:6  @P1 FFMA y1, a1, sn, tm1;
    --:-:-:-:6      MOV y2, x2;
    --:-:-:-:6      MOV y3, x3;
    --:-:-:-:6      MOV32I tiny, 0x0d800000;
    --:-:-:-:6      MOV32I big, 0x42fe0000;
    --:-:-:-:6      MOV32I r127, 0x3c010204;
    --:-:-:-:6      FADD g0, |y0|, -RZ;
    --:-:-:-:6      FADD g1, |y1|, -RZ;
    --:-:-:-:6      FADD g2, |y2|, -RZ;
    --:-:-:-:6      FADD g3, |y3|, -RZ;
]==]))
emit(bfly({ "g0", "g1", "g2", "g3" }, { "tm0", "tm1", "tm2", "tm3" }, { 16, 8, 4, 2, 1 }, "max"))
for j = 0, 3 do emit(string.format("--:-:-:-:6      FMNMX g%d, g%d, tiny, !PT;", j, j)) end
for j = 0, 3 do emit(string.format("--:-:3:-:1      MUFU.RCP tm%d, g%d;", j, j)) end
emit("04:-:-:-:6      NOP;")
for j = 0, 3 do emit(string.format("--:-:-:-:6      FMUL tm%d, tm%d, big;", j, j)) end
for j = 0, 3 do emit(string.format("--:-:-:-:6      FMUL tm%d, y%d, tm%d;", j, j, j)) end
for j = 0, 3 do emit(string.format("--:-:4:-:1      F2I.S32.F32 qi%d, tm%d;", j, j)) end
for j = 0, 3 do emit(string.format("--:-:-:-:6      FMUL sc%d, g%d, r127;", j, j)) end
for j = 0, 3 do emit(string.format("--:-:5:-:1      F2F.F16.F32 sh%d, sc%d;", j, j)) end
-- decode cache: KQ + T 512 + 256 h + d (VQ likewise), KS + T 32 + 16 h + 2 w; chunk: CKV + p 1024 + 512 h (+ 256 V) + d,
-- CKS + p 128 + 64 h (+ 32 V) + 4 w
emit(heredoc([==[
    --:-:-:-:6      IADD T, p, param_t0;
    --:-:-:-:6      SHL tt, T, 0x9;
    --:-:-:-:6      IADD tt, tt, tid;
    --:-:-:-:6      IADD k0p0.CC, tt, param_KQ0;
    --:-:-:-:6      IADD.X k0p1, RZ, param_KQ1;
    --:-:-:-:6      IADD v0p0.CC, tt, param_VQ0;
    --:-:-:-:6      IADD.X v0p1, RZ, param_VQ1;
    --:-:-:-:6      SHL tt, T, 0x5;
    --:-:-:-:6      ISCADD tt, w, tt, 0x1;
    --:-:-:-:6      IADD k1p0.CC, tt, param_KS0;
    --:-:-:-:6      IADD.X k1p1, RZ, param_KS1;
    --:-:-:-:6      IADD v1p0.CC, tt, param_VS0;
    --:-:-:-:6      IADD.X v1p1, RZ, param_VS1;
    --:-:-:-:6      SHL tt, p, 0xa;
    --:-:-:-:6      IADD tt, tt, tid;
    --:-:-:-:6      IADD bp0.CC, tt, param_CKV0;
    --:-:-:-:6      IADD.X bp1, RZ, param_CKV1;
    --:-:-:-:6      SHL tt, p, 0x7;
    --:-:-:-:6      ISCADD tt, w, tt, 0x2;
    --:-:-:-:6      IADD bsp0.CC, tt, param_CKS0;
    --:-:-:-:6      IADD.X bsp1, RZ, param_CKS1;
    08:-:-:-:6      IADD a0, qi0, 0x80;
    --:-:-:-:6      IADD a1, qi1, 0x80;
    --:-:-:-:6      IADD a2, qi2, 0x80;
    --:-:-:-:6      IADD a3, qi3, 0x80;
    --:1:-:-:1      STG.E.U8 [k0p], qi0;
    --:1:-:-:1      STG.E.U8 [k0p+0x100], qi1;
    --:1:-:-:1      STG.E.U8 [v0p], qi2;
    --:1:-:-:1      STG.E.U8 [v0p+0x100], qi3;
    --:1:-:-:1      STG.E.U8 [bp], a0;
    --:1:-:-:1      STG.E.U8 [bp+0x200], a1;
    --:1:-:-:1      STG.E.U8 [bp+0x100], a2;
    --:1:-:-:1      STG.E.U8 [bp+0x300], a3;
    10:1:-:-:1  @P0 STG.E.U16 [k1p], sh0;
    --:1:-:-:1  @P0 STG.E.U16 [k1p+0x10], sh1;
    --:1:-:-:1  @P0 STG.E.U16 [v1p], sh2;
    --:1:-:-:1  @P0 STG.E.U16 [v1p+0x10], sh3;
    --:1:-:-:1  @P0 STG.E [bsp], sc0;
    --:1:-:-:1  @P0 STG.E [bsp+0x40], sc1;
    --:1:-:-:1  @P0 STG.E [bsp+0x20], sc2;
    --:1:-:-:1  @P0 STG.E [bsp+0x60], sc3;
    01:-:-:-:f      EXIT;
]==]))
end

-- ---------------------------------------------------------------------------------------------------------------------------
if OP == "idx" then
emit(heredoc([==[
    <REGISTER_MAPPING>
        0     : tid
        1     : p
        2     : lane
        3     : w
        4     : lgx
        5     : t
        6     : tt
        7     : ss
        8-11  : x<0-3>
        12-15 : y<0-3>
        16-19 : c<0-3>
        20-23 : s<0-3>
        24-27 : wn<0-3>
        28-31 : yp<0-3>
        32-47 : r<00-15>
        48,49 : ap<0-1>
        50,51 : bp<0-1>
        52,53 : cp<0-1>
        54,55 : dp<0-1>
        56,57 : ep<0-1>
        58    : tm
        59    : col
        60    : b
        61    : pos
        62    : tb
        63    : k
        64,65 : pk<0-1>
        66-69 : h<0-3>
        70    : row
        71    : l7
        72    : bt
    </REGISTER_MAPPING>
]==]))
-- rows row + j (j < 4) at column col: pointers ap, bp, cp, dp (each its own: a load reads its address late)
local function rows4(rowreg, colreg)
  local s = {}
  for j, pp in ipairs({ "ap", "bp", "cp", "dp" }) do
    s[#s + 1] = string.format("--:-:-:-:6      IADD tt, %s, 0x%x;\n--:-:-:-:6      SHL tt, tt, lgx;\n--:-:-:-:6      IADD tt, tt, %s;\n--:-:-:-:6      SHL tt, tt, 0x2;\n"
      .. "--:-:-:-:6      IADD %s0.CC, tt, param_LI0;\n--:-:-:-:6      IADD.X %s1, RZ, param_LI1;\n", rowreg, j - 1, colreg, pp, pp)
  end
  return table.concat(s)
end

-- y = RMS128(x) * wn (lane: dims 4 lane .. + 3; the warp's 128), then RoPE (dims < 64: lanes < 16, partner lane ^ 8) with the
-- cos / sin row crow (a register: the chunk table's row)
local function normrope(wnp, crow)
  return heredoc([==[
      --:-:-:-:6      FMUL ss, x0, x0;
      --:-:-:-:6      FFMA ss, x1, x1, ss;
      --:-:-:-:6      FFMA ss, x2, x2, ss;
      --:-:-:-:6      FFMA ss, x3, x3, ss;
      #{bfly_ss}
      --:-:-:-:6      FMUL ss, ss, 0.0078125;
      --:-:-:-:6      FADD ss, ss, param_eps;
      --:-:3:-:1      MUFU.RSQ ss, ss;
      --:-:-:-:6      SHL tt, lane, 0x4;
      --:-:-:-:6      IADD ep0.CC, tt, param_#{wnp}0;
      --:-:-:-:6      IADD.X ep1, RZ, param_#{wnp}1;
      --:-:2:-:2      LDG.E.128 wn0, [ep];
      --:-:-:-:6      LOP32I.AND l7, lane, 0x7;
      --:-:-:-:6      ISCADD tt, #{crow}, l7, 0x3;
      --:-:-:-:6      SHL tt, tt, 0x4;
      --:-:-:-:6      ISETP.LT.U32.AND P1, PT, lane, 0x10, PT;
      --:-:-:-:6      ISETP.LT.U32.AND P2, PT, lane, 0x8, PT;
      --:-:-:-:6      IADD ap0.CC, tt, param_CS0;
      --:-:-:-:6      IADD.X ap1, RZ, param_CS1;
      --:-:-:-:6      MOV c0, RZ;
      --:-:-:-:6      MOV c1, RZ;
      --:-:-:-:6      MOV c2, RZ;
      --:-:-:-:6      MOV c3, RZ;
      --:-:2:-:2  @P1 LDG.E.128 c0, [ap];
      --:-:-:-:6      IADD bp0.CC, tt, param_SN0;
      --:-:-:-:6      IADD.X bp1, RZ, param_SN1;
      --:-:-:-:6      MOV s0, RZ;
      --:-:-:-:6      MOV s1, RZ;
      --:-:-:-:6      MOV s2, RZ;
      --:-:-:-:6      MOV s3, RZ;
      --:-:2:-:2  @P1 LDG.E.128 s0, [bp];
      06:-:-:-:6      FMUL y0, x0, ss;
      --:-:-:-:6      FMUL y1, x1, ss;
      --:-:-:-:6      FMUL y2, x2, ss;
      --:-:-:-:6      FMUL y3, x3, ss;
      --:-:-:-:6      FMUL y0, y0, wn0;
      --:-:-:-:6      FMUL y1, y1, wn1;
      --:-:-:-:6      FMUL y2, y2, wn2;
      --:-:-:-:6      FMUL y3, y3, wn3;
      --:-:6:-:1      SHFL.BFLY PT, yp0, y0, 0x8, 0x1f;
      --:-:6:-:1      SHFL.BFLY PT, yp1, y1, 0x8, 0x1f;
      --:-:6:-:1      SHFL.BFLY PT, yp2, y2, 0x8, 0x1f;
      --:-:6:-:2      SHFL.BFLY PT, yp3, y3, 0x8, 0x1f;
      --:-:-:-:6  @P2 FADD s0, RZ, -s0;
      --:-:-:-:6  @P2 FADD s1, RZ, -s1;
      --:-:-:-:6  @P2 FADD s2, RZ, -s2;
      --:-:-:-:6  @P2 FADD s3, RZ, -s3;
      --:-:-:-:6      FMUL c0, y0, c0;
      --:-:-:-:6      FMUL c1, y1, c1;
      --:-:-:-:6      FMUL c2, y2, c2;
      --:-:-:-:6      FMUL c3, y3, c3;
      20:-:-:-:6  @P1 FFMA y0, yp0, s0, c0;
      --:-:-:-:6  @P1 FFMA y1, yp1, s1, c1;
      --:-:-:-:6  @P1 FFMA y2, yp2, s2, c2;
      --:-:-:-:6  @P1 FFMA y3, yp3, s3, c3;
  ]==], {
    wnp = wnp,
    crow = crow,
    bfly_ss = bfly({ "ss" }, { "bt" }, { 16, 8, 4, 2, 1 }, "sum"),
  })
end

-- the pooled key of block b from the 4 raw keys in r00..r15 (r[4 i + j]: position 4 b + i, dim 4 lane + j): PKT and POOL
local function pool(crow)
  return heredoc([==[
      --:-:-:-:6      FADD x0, r00, r04;
      --:-:-:-:6      FADD x1, r01, r05;
      --:-:-:-:6      FADD x2, r02, r06;
      --:-:-:-:6      FADD x3, r03, r07;
      --:-:-:-:6      FADD x0, x0, r08;
      --:-:-:-:6      FADD x1, x1, r09;
      --:-:-:-:6      FADD x2, x2, r10;
      --:-:-:-:6      FADD x3, x3, r11;
      --:-:-:-:6      FADD x0, x0, r12;
      --:-:-:-:6      FADD x1, x1, r13;
      --:-:-:-:6      FADD x2, x2, r14;
      --:-:-:-:6      FADD x3, x3, r15;
      --:-:-:-:6      FMUL x0, x0, 0.25;
      --:-:-:-:6      FMUL x1, x1, 0.25;
      --:-:-:-:6      FMUL x2, x2, 0.25;
      --:-:-:-:6      FMUL x3, x3, 0.25;
      #{normrope_kn}
      --:-:4:-:1      F2F.F16.F32 h0, y0;
      --:-:4:-:1      F2F.F16.F32 h1, y1;
      --:-:4:-:1      F2F.F16.F32 h2, y2;
      --:-:4:-:1      F2F.F16.F32 h3, y3;
      --:-:-:-:6      SHR.U32 tt, b, 0x5;
      --:-:-:-:6      SHL tt, tt, 0xd;
      --:-:-:-:6      ISCADD tt, lane, tt, 0x8;
      --:-:-:-:6      LOP32I.AND tm, b, 0x1f;
      --:-:-:-:6      ISCADD tt, tm, tt, 0x3;
      --:-:-:-:6      IADD ep0.CC, tt, param_PKT0;
      --:-:-:-:6      IADD.X ep1, RZ, param_PKT1;
      08:-:-:-:6      PRMT pk0, h0, 0x5410, h1;
      --:-:-:-:6      PRMT pk1, h2, 0x5410, h3;
      --:1:-:-:1      STG.E.64 [ep], pk0;
      --:-:-:-:6      SHL tt, lane, 0xf;
      --:-:-:-:6      SHL tt, tt, 0x2;
      --:-:-:-:6      ISCADD tt, b, tt, 0x2;
      --:-:-:-:6      IADD ap0.CC, tt, param_POOL0;
      --:-:-:-:6      IADD.X ap1, RZ, param_POOL1;
      --:1:-:-:1      STG.E [ap], y0;
      --:1:-:-:1      STG.E [ap+0x#{ps1}], y1;
      --:1:-:-:1      STG.E [ap+0x#{ps2}], y2;
      --:1:-:-:1      STG.E [ap+0x#{ps3}], y3;
      01:-:-:-:6      NOP;
  ]==], {
    normrope_kn = normrope("KN", crow),
    ps1 = hx(POOL_STRIDE),
    ps2 = hx(2 * POOL_STRIDE),
    ps3 = hx(3 * POOL_STRIDE),
  })
end

emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R p, SR_CTAID.X;
    --:-:-:-:6      MOV lgx, param_lgx;
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 w, tid, 0x5;
    --:-:-:-:6      IADD t, p, param_t0;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, w, 0x4, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA KEY;
    --:-:-:-:6      ISCADD row, w, RZ, 0x7;
    --:-:-:-:6      ISCADD row, lane, row, 0x2;
    --:-:-:-:6      MOV col, p;
]==]))
emit(rows4("row", "col"))
emit(heredoc([==[
    --:-:2:-:1      LDG.E x0, [ap];
    --:-:2:-:1      LDG.E x1, [bp];
    --:-:2:-:1      LDG.E x2, [cp];
    --:-:2:-:2      LDG.E x3, [dp];
    --:-:-:-:6      IADD tm, p, 0x3;
    02:-:-:-:6      NOP;
    #{normrope_qn}
    --:-:-:-:6      SHL tt, p, 0xb;
    --:-:-:-:6      ISCADD tt, w, tt, 0x9;
    --:-:-:-:6      ISCADD tt, lane, tt, 0x4;
    --:-:-:-:6      IADD ep0.CC, tt, param_IQ0;
    --:-:-:-:6      IADD.X ep1, RZ, param_IQ1;
    --:1:-:-:1      STG.E.128 [ep], y0;
    01:-:-:-:f      EXIT;
    KEY:
    --:-:-:-:6      ISETP.NE.U32.AND P0, PT, w, 0x4, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA RINGW;
    --:-:-:-:6      LOP32I.AND tt, t, 0x3;
    --:-:-:-:6      ISETP.NE.U32.AND P0, PT, tt, 0x3, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      ISETP.LT.U32.OR P0, PT, p, 0x3, P0;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 EXIT;
    --:-:-:-:6      SHR.U32 b, t, 0x2;
    --:-:-:-:6      ISCADD row, lane, 0x200, 0x2;
    --:-:-:-:6      IADD col, p, -0x3;
]==], { normrope_qn = normrope("QN", "tm") }))
emit(rows4("row", "col"))
for q = 0, 15 do
  emit(string.format("--:-:2:-:1      LDG.E r%02d, [%s+0x%x];", q, ({ "ap", "bp", "cp", "dp" })[q % 4 + 1], 4 * (q // 4)))
end
emit("02:-:-:-:6      MOV tm, p;")
emit(pool("tm"))
emit("--:-:-:-:f      EXIT;")
-- warp 5 of unit 0: the straddling block tb = t0 & ~3 when t0 & 3 != 0 and tb + 3 < t0 + npos, then the ring
emit(heredoc([==[
    RINGW:
    --:-:-:-:6      ISETP.NE.U32.AND P0, PT, w, 0x5, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      ISETP.NE.U32.OR P0, PT, p, RZ, P0;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 EXIT;
    --:-:-:-:6      MOV tt, param_t0;
    --:-:-:-:6      LOP32I.AND tb, tt, 0xfffffffc;
    --:-:-:-:6      IADD tm, tb, 0x3;
    --:-:-:-:6      MOV k, param_npos;
    --:-:-:-:6      IADD k, k, tt;
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, tb, tt, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      ISETP.GE.U32.OR P0, PT, tm, k, P0;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA NEWRING;
    --:-:-:-:6      SHR.U32 b, tb, 0x2;
]==]))
-- position tb + i: from the ring (pos < t0) or the chunk's column pos - t0
for i = 0, 3 do
  emit(heredoc([==[
      --:-:-:-:6      IADD pos, tb, 0x#{hi};
      --:-:-:-:6      MOV tt, param_t0;
      --:-:-:-:6      ISETP.LT.U32.AND P1, PT, pos, tt, PT;
      --:-:-:-:6      LOP32I.AND tm, pos, 0x3;
      --:-:-:-:6      SHL tm, tm, 0x9;
      --:-:-:-:6      ISCADD tm, lane, tm, 0x4;
      --:-:-:-:6      IADD ep0.CC, tm, param_RING0;
      --:-:-:-:6      IADD.X ep1, RZ, param_RING1;
      --:-:2:-:2  @P1 LDG.E.128 r#{rr}, [ep];
      02:-:-:-:6      IADD col, pos, -tt;
      --:-:-:-:6      ISCADD row, lane, 0x200, 0x2;
  ]==], { hi = hx(i), rr = string.format("%02d", 4 * i) }))
  emit(rows4("row", "col"))
  for j = 0, 3 do
    emit(string.format("--:-:2:-:1 @!P1 LDG.E r%02d, [%s];", 4 * i + j, ({ "ap", "bp", "cp", "dp" })[j + 1]))
  end
  emit("02:-:-:-:6      NOP;")
end
emit(heredoc([==[
    --:-:-:-:6      MOV tt, param_t0;
    --:-:-:-:6      IADD tm, tb, -tt;
    --:-:-:-:6      IADD tm, tm, 0x3;
]==]))
emit(pool("tm"))
-- the ring: positions t0 + npos - 4 + k at or after t0 (chunk columns npos - 4 + k), slot pos & 3
emit("NEWRING:")
for kk = 0, 3 do
  emit(heredoc([==[
      --:-:-:-:6      MOV tt, param_npos;
      --:-:-:-:6      IADD col, tt, -0x#{neg};
      --:-:-:-:6      ISETP.LT.AND P1, PT, col, RZ, PT;
      --:-:-:-:6      ISCADD row, lane, 0x200, 0x2;
  ]==], { neg = hx(4 - kk) }))
  emit(rows4("row", "col"))
  for j = 0, 3 do
    emit(string.format("--:-:2:-:1 @!P1 LDG.E x%d, [%s];", j, ({ "ap", "bp", "cp", "dp" })[j + 1]))
  end
  emit(heredoc([==[
      --:-:-:-:6      MOV tt, param_t0;
      --:-:-:-:6      IADD pos, col, tt;
      --:-:-:-:6      LOP32I.AND tm, pos, 0x3;
      --:-:-:-:6      SHL tm, tm, 0x9;
      --:-:-:-:6      ISCADD tm, lane, tm, 0x4;
      --:-:-:-:6      IADD ep0.CC, tm, param_RING0;
      --:-:-:-:6      IADD.X ep1, RZ, param_RING1;
      02:1:-:-:1 @!P1 STG.E.128 [ep], x0;
      01:-:-:-:6      NOP;
  ]==]))
end
emit("--:-:-:-:f      EXIT;")
end

-- ---------------------------------------------------------------------------------------------------------------------------
if OP == "sel" then
emit(heredoc([==[
    <REGISTER_MAPPING>
        0     : tid
        1     : p
        2     : lane
        3     : w
        4     : t
        5     : nv
        6     : tt
        7     : b
        8-11  : acc<0-3>
        12-15 : qa<0-3>
        16-19 : qb<0-3>
        20-23 : qc<0-3>
        24-27 : qd<0-3>
        28-43 : pv<00-15>
        44-47 : accb<0-3>
        48,49 : pp<0-1>
        50    : it
        51    : nit
        52    : pre
        53    : pmask
        54    : kk
        55    : sh
        56    : dm
        57    : v
        58    : dg
        59    : one
        60    : cnt
        61    : sfx
        62    : tm
        63    : above
        64-71 : wc<0-7>
        72    : found
        73    : d8
        74,75 : sp<0-1>
        76    : bit
        77    : run
        78    : hb
        79    : dsel
        80    : knew
        81    : lt
        82    : bal
        83    : rank
        84    : base
        85    : eqb
        86    : tb
        87    : need
        88    : tot
    </REGISTER_MAPPING>
]==]))
emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R p, SR_CTAID.X;
    --:-:1:-:1      S2R lt, SR_LTMASK;
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 w, tid, 0x5;
    --:-:-:-:6      IADD t, p, param_t0;
    --:-:-:-:6      IADD nv, t, 0x1;
    --:-:-:-:6      SHR.U32 nv, nv, 0x2;
    --:-:-:-:6      ISETP.LE.U32.AND P0, PT, nv, param_nblk, PT;
    --:-:-:-:6      MOV32I one, 0x1;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 EXIT;
    --:-:-:-:6      SHL tt, p, 0xb;
    --:-:-:-:6      ISCADD tt, tid, tt, 0x3;
    --:-:-:-:6      IADD sp0.CC, tt, param_IQ0;
    --:-:-:-:6      IADD.X sp1, RZ, param_IQ1;
    --:-:2:-:2      LDG.E.64 acc0, [sp];
    --:-:-:-:6      SHL tt, tid, 0x3;
    02:1:-:-:1      STS.64 [tt], acc0;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      IADD nit, nv, 0xff;
    --:-:-:-:6      SHR.U32 nit, nit, 0x8;
    --:-:-:-:6      MOV it, RZ;
    SCORE:
    --:-:-:-:6      ISCADD b, it, tid, 0x8;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, b, nv, PT;
    --:-:-:-:6      SHL tt, b, 0x2;
    --:-:-:-:6      IADD pp0.CC, tt, param_POOL0;
    --:-:-:-:6      IADD.X pp1, RZ, param_POOL1;
    --:-:-:-:6      MOV acc0, RZ;
    --:-:-:-:6      MOV acc1, RZ;
    --:-:-:-:6      MOV acc2, RZ;
    --:-:-:-:6      MOV acc3, RZ;
]==]))
-- dims in groups of 8: the pool rows' values (LDG with the row's immediate offset) double-buffered (bank g & 1 on barrier
-- 2 / 5), q of the 4 heads by LDS.128; two accumulator sets (even / odd dims) keep each FFMA chain 8 instructions apart
local function ldg(g)
  local t = {}
  for e = 0, 7 do
    local bar = 5
    if g % 2 == 0 then bar = 2 end
    t[#t + 1] = string.format("--:-:%d:-:1  @P1 LDG.E pv%02d, [pp+0x%x];", bar, 8 * (g & 1) + e, (8 * g + e) * POOL_STRIDE)
  end
  return table.concat(t, "\n")
end
emit(ldg(0))
emit("--:-:-:-:1      MOV accb0, RZ;\n--:-:-:-:1      MOV accb1, RZ;\n--:-:-:-:1      MOV accb2, RZ;\n--:-:-:-:1      MOV accb3, RZ;")
for g = 0, 15 do
  if g < 15 then emit(ldg(g + 1)) end
  for half = 0, 1 do
    local d0 = 8 * g + 4 * half
    emit(string.format("--:-:3:-:1      LDS.U.128 qa0, [RZ+0x%x];", 4 * d0))
    emit(string.format("--:-:3:-:1      LDS.U.128 qb0, [RZ+0x%x];", 4 * (128 + d0)))
    emit(string.format("--:-:3:-:1      LDS.U.128 qc0, [RZ+0x%x];", 4 * (256 + d0)))
    emit(string.format("--:-:3:-:1      LDS.U.128 qd0, [RZ+0x%x];", 4 * (384 + d0)))
    for e = 0, 3 do
      local wait = 0
      if e == 0 then
        if half == 0 then
          if g % 2 == 0 then wait = 0x06 else wait = 0x14 end
        else
          wait = 0x04
        end
      end
      for h0, qn in ipairs({ "qa", "qb", "qc", "qd" }) do
        local h = h0 - 1
        local a
        if e % 2 == 0 then a = string.format("acc%d", h) else a = string.format("accb%d", h) end
        local w = "--"
        if wait ~= 0 and h == 0 then w = string.format("%02x", wait) end
        emit(string.format("%s:-:-:-:1      FFMA %s, %s%d, pv%02d, %s;", w, a, qn, e, 8 * (g & 1) + 4 * half + e, a))
      end
    end
  end
end
emit("--:-:-:-:6      NOP;\n--:-:-:-:6      FADD acc0, acc0, accb0;\n--:-:-:-:6      FADD acc1, acc1, accb1;\n--:-:-:-:6      FADD acc2, acc2, accb2;\n--:-:-:-:6      FADD acc3, acc3, accb3;")
emit(heredoc([==[
    --:-:-:-:6      FMNMX acc0, acc0, RZ, !PT;
    --:-:-:-:6      FMNMX acc1, acc1, RZ, !PT;
    --:-:-:-:6      FMNMX acc2, acc2, RZ, !PT;
    --:-:-:-:6      FMNMX acc3, acc3, RZ, !PT;
    --:-:-:-:6      FADD acc0, acc0, acc1;
    --:-:-:-:6      FADD acc0, acc0, acc2;
    --:-:-:-:6      FADD acc0, acc0, acc3;
    --:-:-:-:6      FADD acc0, acc0, RZ;
    --:-:-:-:6      SHL tt, b, 0x2;
    --:1:-:-:1  @P1 STS [tt+0x800], acc0;
    --:-:-:-:6      IADD it, it, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, it, nit, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    01:-:-:-:6  @P0 BRA SCORE;
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      MOV pre, RZ;
    --:-:-:-:6      MOV pmask, RZ;
    --:-:-:-:6      MOV kk, param_nblk;
]==]))
-- four digit levels: bits 30..23, 22..15, 14..7, 6..0
local LEVELS = { { 23, 0xff }, { 15, 0xff }, { 7, 0xff }, { 0, 0x7f } }
for lv = 0, 3 do
  local shift, mask = LEVELS[lv + 1][1], LEVELS[lv + 1][2]
  emit(heredoc([==[
      --:-:-:-:6      SHL tt, tid, 0x2;
      --:1:-:-:1      STS [tt+0x8800], RZ;
      01:-:-:-:5      BAR.SYNC 0x0;
      --:-:-:-:6      MOV it, RZ;
      HIST#{lv}:
      --:-:-:-:6      ISCADD b, it, tid, 0x8;
      --:-:-:-:6      ISETP.LT.U32.AND P1, PT, b, nv, PT;
      --:-:-:-:6      SHL tt, b, 0x2;
      --:-:-:-:6      NOP;
      --:-:2:-:2  @P1 LDS.U v, [tt+0x800];
      02:-:-:-:6      LOP.AND tm, v, pmask;
      --:-:-:-:6      ISETP.EQ.U32.AND P2, PT, tm, pre, P1;
      --:-:-:-:6      SHR.U32 dg, v, 0x#{shift};
      --:-:-:-:6      LOP32I.AND dg, dg, 0x#{mask};
      --:-:-:-:6      SHL dg, dg, 0x2;
      --:-:-:-:6      NOP;
      --:-:2:-:2  @P2 ATOMS.ADD tm, [dg+0x8800], one;
      02:-:-:-:6      IADD it, it, 0x1;
      --:-:-:-:6      ISETP.LT.U32.AND P0, PT, it, nit, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P0 BRA HIST#{lv};
      --:-:-:-:5      BAR.SYNC 0x0;
      --:-:-:-:6      ISETP.NE.U32.AND P0, PT, w, RZ, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P0 BRA SCANNED#{lv};
      --:-:-:-:6      SHL tt, lane, 0x5;
      --:-:2:-:1      LDS.U.128 wc0, [tt+0x8800];
      --:-:2:-:2      LDS.U.128 wc4, [tt+0x8810];
      02:-:-:-:6      IADD cnt, wc0, wc1;
      --:-:-:-:6      IADD cnt, cnt, wc2;
      --:-:-:-:6      IADD cnt, cnt, wc3;
      --:-:-:-:6      IADD cnt, cnt, wc4;
      --:-:-:-:6      IADD cnt, cnt, wc5;
      --:-:-:-:6      IADD cnt, cnt, wc6;
      --:-:-:-:6      IADD cnt, cnt, wc7;
      --:-:-:-:6      MOV sfx, cnt;
  ]==], { lv = hx(lv), shift = hx(shift), mask = hx(mask) }))
  -- inclusive suffix sums over the lanes: sfx_l = sum of cnt over lanes >= l
  for _, off in ipairs({ 1, 2, 4, 8, 16 }) do
    emit(heredoc([==[
        --:-:2:-:2      SHFL.DOWN P3, tm, sfx, 0x#{off}, 0x1f;
        02:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6  @P3 IADD sfx, sfx, tm;
    ]==], { off = hx(off) }))
  end
  -- the lane whose bins hold the kk-th largest: above < kk <= sfx; it walks its bins 8 l + 7 .. 8 l
  emit(heredoc([==[
      --:-:-:-:6      IADD above, sfx, -cnt;
      --:-:-:-:6      ISETP.LT.U32.AND P3, PT, above, kk, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      ISETP.GE.U32.AND P3, PT, sfx, kk, P3;
      --:-:-:-:6      MOV run, above;
      --:-:-:-:6      MOV found, RZ;
      --:-:-:-:6      SHL d8, lane, 0x3;
  ]==]))
  for i = 7, 0, -1 do
    emit(heredoc([==[
        --:-:-:-:6      IADD hb, run, wc#{i};
        --:-:-:-:6      ISETP.GE.U32.AND P4, PT, hb, kk, P3;
        --:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, found, RZ, P4;
        --:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6      NOP;
        --:-:-:-:6  @P4 IADD dsel, d8, 0x#{hi};
        --:-:-:-:6  @P4 IADD knew, kk, -run;
        --:-:-:-:6  @P4 MOV32I found, 0x1;
        --:-:-:-:6      MOV run, hb;
    ]==], { i = string.format("%d", i), hi = hx(i) }))
  end
  emit(heredoc([==[
      --:-:-:-:6      NOP;
      --:1:-:-:1  @P3 STS [RZ+0x8c00], dsel;
      --:1:-:-:1  @P3 STS [RZ+0x8c04], knew;
      01:-:-:-:6      NOP;
      SCANNED#{lv}:
      --:-:-:-:5      BAR.SYNC 0x0;
      --:-:2:-:1      LDS.U dsel, [RZ+0x8c00];
      --:-:2:-:2      LDS.U kk, [RZ+0x8c04];
      02:-:-:-:6      SHL tm, dsel, 0x#{shift};
      --:-:-:-:6      LOP.OR pre, pre, tm;
      --:-:-:-:6      MOV32I tm, 0x#{maskshift};
      --:-:-:-:6      LOP.OR pmask, pmask, tm;
      --:-:-:-:5      BAR.SYNC 0x0;
  ]==], { lv = hx(lv), shift = hx(shift), maskshift = hx(mask << shift) }))
end
if os.getenv("CA_DEBUG") == "1" then
  emit(heredoc([==[
      --:-:-:-:6      MOV it, RZ;
      DBGS:
      --:-:-:-:6      ISCADD b, it, tid, 0x8;
      --:-:-:-:6      ISETP.LT.U32.AND P1, PT, b, nv, PT;
      --:-:-:-:6      SHL tt, b, 0x2;
      --:-:-:-:6      NOP;
      --:-:2:-:2  @P1 LDS.U v, [tt+0x800];
      --:-:-:-:6      SHL tm, p, 0xb;
      --:-:-:-:6      IADD tt, tm, tt;
      --:-:-:-:6      IADD pp0.CC, tt, param_IQ0;
      --:-:-:-:6      IADD.X pp1, RZ, param_IQ1;
      02:1:-:-:1  @P1 STG.E [pp], v;
      01:-:-:-:6      IADD it, it, 0x1;
      --:-:-:-:6      ISETP.LT.U32.AND P0, PT, it, nit, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P0 BRA DBGS;
      --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, tid, RZ, PT;
      --:-:-:-:6      SHL tt, p, 0xa;
      --:-:-:-:6      IADD pp0.CC, tt, param_SEL0;
      --:-:-:-:6      IADD.X pp1, RZ, param_SEL1;
      --:1:-:-:1  @P0 STG.E [pp+0x3e8], pre;
      --:1:-:-:1  @P0 STG.E [pp+0x3ec], kk;
      --:1:-:-:1  @P0 STG.E [pp+0x3f0], nv;
      01:-:-:-:5      BAR.SYNC 0x0;
  ]==]))
end
-- T = pre, need = kk: in ascending block order, take v > T and the first need of v == T
emit(heredoc([==[
    --:-:-:-:6      MOV need, kk;
    --:-:-:-:6      MOV base, RZ;
    --:-:-:-:6      MOV eqb, RZ;
    --:-:-:-:6      SHL tt, p, 0xa;
    --:-:-:-:6      IADD sp0.CC, tt, param_SEL0;
    --:-:-:-:6      IADD.X sp1, RZ, param_SEL1;
    --:-:-:-:6      MOV it, RZ;
    TAKE:
    --:-:-:-:6      ISCADD b, it, tid, 0x8;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, b, nv, PT;
    --:-:-:-:6      SHL tt, b, 0x2;
    --:-:-:-:6      MOV v, RZ;
    --:-:-:-:6      NOP;
    --:-:2:-:2  @P1 LDS.U v, [tt+0x800];
    02:-:-:-:6      ISETP.EQ.U32.AND P2, PT, v, pre, P1;
    --:-:-:-:6      ISETP.GT.U32.AND P4, PT, v, pre, P1;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      VOTE.ANY bal, PT, P2;
    --:-:-:-:6      LOP.AND tm, bal, lt;
    --:-:4:-:1      POPC rank, tm;
    --:-:4:-:1      POPC tot, bal;
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, lane, RZ, PT;
    --:-:-:-:6      SHL tt, w, 0x2;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    08:1:-:-:1  @P0 STS [tt+0x8c40], tot;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:2:-:1      LDS.U.128 wc0, [RZ+0x8c40];
    --:-:2:-:2      LDS.U.128 wc4, [RZ+0x8c50];
    02:-:-:-:6      MOV tb, eqb;
]==]))
-- rank among the equal ones: eqb + those of the warps before + the lane's; the warps' total moves eqb
for i = 0, 7 do
  emit(heredoc([==[
      --:-:-:-:6      ISETP.GT.U32.AND P5, PT, w, 0x#{hi}, PT;
      --:-:-:-:6      IADD eqb, eqb, wc#{i};
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P5 IADD tb, tb, wc#{i};
  ]==], { hi = hx(i), i = string.format("%d", i) }))
end
emit(heredoc([==[
    --:-:-:-:6      IADD rank, rank, tb;
    --:-:-:-:6      ISETP.LT.U32.AND P2, PT, rank, need, P2;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      PSETP.OR.AND P4, PT, P4, P2, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      VOTE.ANY bal, PT, P4;
    --:-:-:-:6      LOP.AND tm, bal, lt;
    --:-:4:-:1      POPC rank, tm;
    --:-:4:-:1      POPC tot, bal;
    --:-:-:-:6      SHL tt, w, 0x2;
    08:1:-:-:1  @P0 STS [tt+0x8c60], tot;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:2:-:1      LDS.U.128 wc0, [RZ+0x8c60];
    --:-:2:-:2      LDS.U.128 wc4, [RZ+0x8c70];
    02:-:-:-:6      MOV tb, base;
]==]))
for i = 0, 7 do
  emit(heredoc([==[
      --:-:-:-:6      ISETP.GT.U32.AND P5, PT, w, 0x#{hi}, PT;
      --:-:-:-:6      IADD base, base, wc#{i};
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P5 IADD tb, tb, wc#{i};
  ]==], { hi = hx(i), i = string.format("%d", i) }))
end
emit(heredoc([==[
    --:-:-:-:6      IADD rank, rank, tb;
    --:-:-:-:6      SHL tt, rank, 0x1;
    --:-:-:-:6      IADD pp0.CC, sp0, tt;
    --:-:-:-:6      IADD.X pp1, sp1, RZ;
    --:-:-:-:6      NOP;
    --:1:-:-:1  @P4 STG.E.U16 [pp], b;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      IADD it, it, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, it, nit, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA TAKE;
    --:-:-:-:f      EXIT;
]==]))
end

-- ---------------------------------------------------------------------------------------------------------------------------
if OP == "app" then
emit(heredoc([==[
    <REGISTER_MAPPING>
        0     : tid
        1     : u
        2     : pl
        3     : c
        4     : p
        5     : tt
        6     : T
        7     : off
        8-11  : x<0-3>
        12,13 : sp<0-1>
        14,15 : dp<0-1>
    </REGISTER_MAPPING>
]==]))
-- thread (pl = tid >> 6, c = tid & 63): position p = 4 u + pl; c < 32: the 16 B at 16 c of the head's 512 codes; 32 <= c < 36:
-- the 16 B at 16 (c - 32) of its 64 B of scales
emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R u, SR_CTAID.X;
    01:-:-:-:6      SHR.U32 pl, tid, 0x6;
    --:-:-:-:6      LOP32I.AND c, tid, 0x3f;
    --:-:-:-:6      ISCADD p, u, pl, 0x2;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, p, param_npos, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      ISETP.GE.U32.OR P0, PT, c, 0x24, P0;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, c, 0x20, PT;
    --:-:-:-:6      IADD T, p, param_t0;
    --:-:-:-:6      SHL off, c, 0x4;
    --:-:-:-:6      SHL tt, p, 0xa;
    --:-:-:-:6      ISCADD tt, param_hk, tt, 0x9;
    --:-:-:-:6      IADD tt, tt, off;
    --:-:-:-:6      IADD sp0.CC, tt, param_CKV0;
    --:-:-:-:6      IADD.X sp1, RZ, param_CKV1;
    --:-:-:-:6      SHL tt, T, 0x9;
    --:-:-:-:6      IADD tt, tt, off;
    --:-:-:-:6      IADD dp0.CC, tt, param_R0;
    --:-:-:-:6      IADD.X dp1, RZ, param_R1;
    --:-:-:-:6      IADD off, off, -0x200;
    --:-:-:-:6      SHL tt, p, 0x7;
    --:-:-:-:6      ISCADD tt, param_hk, tt, 0x6;
    --:-:-:-:6      IADD tt, tt, off;
    --:-:-:-:6  @!P1 IADD sp0.CC, tt, param_CKS0;
    --:-:-:-:6  @!P1 IADD.X sp1, RZ, param_CKS1;
    --:-:-:-:6      SHL tt, T, 0x6;
    --:-:-:-:6      IADD tt, tt, off;
    --:-:-:-:6  @!P1 IADD dp0.CC, tt, param_RS0;
    --:-:-:-:6  @!P1 IADD.X dp1, RZ, param_RS1;
    --:-:2:-:2 @!P0 LDG.E.128 x0, [sp];
    02:1:-:-:1 @!P0 STG.E.128 [dp], x0;
    01:-:-:-:f      EXIT;
]==]))
end

-- ---------------------------------------------------------------------------------------------------------------------------
if OP == "attn" then
emit(heredoc([==[
    <REGISTER_MAPPING>
        0     : tid
        1     : p
        2     : lane
        3     : w
        4     : lgx
        5     : t
        6     : nv
        7     : tail
        8     : ns
        9     : n
        10    : ntile
        11    : tile
        12    : kk
        13    : qq
        14    : r
        15    : dg
        16    : jj
        17    : src
        18    : T
        19    : tt
        20-25 : s<0-5>
        26-31 : pt<0-5>
        32-37 : m<0-5>
        38-43 : den<0-5>
        44-49 : cr<0-5>
        52-75 : o<00-23>
        76-81 : g<0-5>
        84-99 : kc<00-15>
        100-103 : ks<0-3>
        104-127 : qa<00-23>
        128-131 : fv<0-3>
        132   : c4b
        133   : nbias
        134   : l2e
        135   : negbig
        136   : qsb
        137   : tm
        138   : qn
        139   : cs
        140   : sn
        141   : nlist
        142   : i
        143   : slot
        144,145 : ap<0-1>
        146,147 : bp<0-1>
        148-163 : xp<00-15>
        164-169 : tmp<0-5>
        170   : hh
        171   : ldxs
        172-175 : vw<0-3>
        176-179 : vs<0-3>
        180   : rk4
        181   : rp32
        182   : ta
    </REGISTER_MAPPING>
]==]))
-- head h (< 6) reads local head min(h, nhh - 1): rows (hq0 + hh) 512 + d (q) and + 256 (gate), column p
emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R p, SR_CTAID.X;
    --:-:-:-:6      MOV lgx, param_lgx;
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 w, tid, 0x5;
    --:-:-:-:6      MOV32I c4b, 0x4b000000;
    --:-:-:-:6      MOV32I nbias, 0xcb000080;
    --:-:-:-:6      MOV32I l2e, 0x#{l2e};
    --:-:-:-:6      MOV32I negbig, 0xff7fffff;
    --:-:-:-:6      MOV32I tm, 0x1;
    --:-:-:-:6      SHL ldxs, tm, lgx;
    --:-:-:-:6      SHL ldxs, ldxs, 0x2;
]==], { l2e = hx(L2E) }))
for h = 0, NH - 1 do
  emit(heredoc([==[
      --:-:-:-:6      MOV hh, param_nhh;
      --:-:-:-:6      IADD hh, hh, -0x1;
      --:-:-:-:6      IMNMX.U32 hh, hh, 0x#{hh}, PT;
      --:-:-:-:6      IADD hh, hh, param_hq0;
      --:-:-:-:6      ISCADD tt, hh, tid, 0x9;
      --:-:-:-:6      SHL tt, tt, lgx;
      --:-:-:-:6      IADD tt, tt, p;
      --:-:-:-:6      SHL tt, tt, 0x2;
      --:-:-:-:6      IADD xp#{xa}.CC, tt, param_LQ0;
      --:-:-:-:6      IADD.X xp#{xb}, RZ, param_LQ1;
      --:-:2:-:1      LDG.E s#{h}, [xp#{xa}];
  ]==], { hh = hx(h), xa = string.format("%02d", 2 * h), xb = string.format("%02d", 2 * h + 1), h = string.format("%d", h) }))
end
-- gates: 256 rows further (256 << lgx << 2 bytes); the pointers move only after their loads returned
emit("02:-:-:-:6      SHL tm, ldxs, 0x8;")
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      IADD xp%02d.CC, xp%02d, tm;\n--:-:-:-:6      IADD.X xp%02d, xp%02d, RZ;\n--:-:2:-:1      LDG.E g%d, [xp%02d];", 2 * h, 2 * h, 2 * h + 1, 2 * h + 1, h, 2 * h))
end
emit(heredoc([==[
    --:-:-:-:6      SHL tt, tid, 0x2;
    --:-:-:-:6      IADD ap0.CC, tt, param_QN0;
    --:-:-:-:6      IADD.X ap1, RZ, param_QN1;
    --:-:2:-:2      LDG.E qn, [ap];
    02:-:-:-:6      NOP;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1      FMUL pt%d, s%d, s%d;", h, h, h))
end
emit(bfly(names("pt", NH), names("tmp", NH), { 16, 8, 4, 2, 1 }, "sum"))
emit(heredoc([==[
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, lane, RZ, PT;
    --:-:-:-:6      SHL tt, w, 0x2;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:1:-:-:1  @P0 STS [tt+0x%x], pt%d;", 0x6000 + 32 * h, h))
end
emit("01:-:-:-:5      BAR.SYNC 0x0;")
for h = 0, NH - 1 do
  emit(string.format("--:-:2:-:1      LDS.U.128 kc00, [RZ+0x%x];\n--:-:2:-:2      LDS.U.128 kc04, [RZ+0x%x];", 0x6000 + 32 * h, 0x6010 + 32 * h))
  emit(string.format("02:-:-:-:6      FADD pt%d, kc00, kc01;", h))
  for _, k in ipairs({ "kc02", "kc03", "kc04", "kc05", "kc06", "kc07" }) do
    emit(string.format("--:-:-:-:6      FADD pt%d, pt%d, %s;", h, h, k))
  end
  emit(string.format("--:-:-:-:6      FMUL pt%d, pt%d, 0.00390625;\n--:-:-:-:6      FADD pt%d, pt%d, param_eps;\n--:-:3:-:1      MUFU.RSQ pt%d, pt%d;", h, h, h, h, h, h))
end
emit("04:-:-:-:6      NOP;")
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      FMUL s%d, s%d, pt%d;\n--:-:-:-:6      FMUL s%d, s%d, qn;\n--:-:-:-:6      FMUL s%d, s%d, 0.0625;", h, h, h, h, h, h, h))
end
emit("--:-:-:-:6      SHL tt, tid, 0x2;")
for h = 0, NH - 1 do
  emit(string.format("--:1:-:-:1      STS [tt+0x%x], s%d;", 1024 * h, h))
end
-- RoPE on dims < 64 (warps 0, 1): the partner d ^ 32 read before the barrier, the rotated value written after
emit(heredoc([==[
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, tid, 0x40, PT;
    --:-:-:-:6      ISETP.LT.U32.AND P2, PT, tid, 0x20, PT;
    --:-:-:-:6      LOP32I.XOR tm, tid, 0x20;
    --:-:-:-:6      SHL tm, tm, 0x2;
    --:-:-:-:6      ISCADD tt, p, lane, 0x5;
    --:-:-:-:6      IADD tt, tt, 0x60;
    --:-:-:-:6      SHL tt, tt, 0x2;
    --:-:-:-:6      IADD ap0.CC, tt, param_CS0;
    --:-:-:-:6      IADD.X ap1, RZ, param_CS1;
    --:-:-:-:6      IADD bp0.CC, tt, param_SN0;
    --:-:-:-:6      IADD.X bp1, RZ, param_SN1;
    --:-:-:-:6      MOV cs, RZ;
    --:-:-:-:6      MOV sn, RZ;
    --:-:2:-:1  @P1 LDG.E cs, [ap];
    --:-:2:-:1  @P1 LDG.E sn, [bp];
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:-:2:-:1  @P1 LDS.U tmp%d, [tm+0x%x];", h, 1024 * h))
end
emit("02:-:-:-:5      BAR.SYNC 0x0;\n--:-:-:-:6  @P2 FADD sn, RZ, -sn;")
for h = 0, NH - 1 do
  local text = string.format("--:-:-:-:6      FMUL pt%d, s%d, cs;\n--:-:-:-:6  @P1 FFMA s%d, tmp%d, sn, pt%d;\n--:1:-:-:1  @P1 STS [tt+0x%x], s%d;", h, h, h, h, h, 0, h)
  text = plain_sub(text, "[tt+0x0]", string.format("[tid4+0x%s]", hx(1024 * h)), false)
  emit(text)
end
out_gsub("tid4", "qsb")
emit("01:-:-:-:5      BAR.SYNC 0x0;")   -- (qsb holds 4 tid here, set below before the RoPE stores)
-- (the RoPE stores above address qsb; it is set from tid before them)
out_sub("--:-:-:-:6      ISETP.LT.U32.AND P1, PT, tid, 0x40, PT;\n", "--:-:-:-:6      ISETP.LT.U32.AND P1, PT, tid, 0x40, PT;\n--:-:-:-:6      SHL qsb, tid, 0x2;\n")
-- the sequence: nv complete blocks, the tail, ns attended blocks (SEL when nv > nblk), n keys, the list's 16-block tiles
emit(heredoc([==[
    --:-:-:-:6      IADD t, p, param_t0;
    --:-:-:-:6      IADD nv, t, 0x1;
    --:-:-:-:6      LOP32I.AND tail, nv, 0x3;
    --:-:-:-:6      SHR.U32 nv, nv, 0x2;
    --:-:-:-:6      MOV tm, param_nblk;
    --:-:-:-:6      IMNMX.U32 ns, nv, tm, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P6, PT, nv, tm, PT;
    --:-:-:-:6      IADD n, t, 0x1;
    --:-:-:-:6      ISCADD tt, tm, tail, 0x2;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P6 MOV n, tt;
    --:-:-:-:6      ISETP.NE.U32.AND P0, PT, tail, RZ, PT;
    --:-:-:-:6      MOV nlist, ns;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 IADD nlist, nlist, 0x1;
    --:-:-:-:6      IADD ntile, nlist, 0xf;
    --:-:-:-:6      SHR.U32 ntile, ntile, 0x4;
    --:-:-:-:6      SHR.U32 kk, tid, 0x2;
    --:-:-:-:6      LOP32I.AND qq, tid, 0x3;
    --:-:-:-:6      SHL qsb, qq, 0x6;
    --:-:-:-:6      SHR.U32 r, tid, 0x6;
    --:-:-:-:6      LOP32I.AND dg, tid, 0x3f;
    --:-:-:-:6      MOV tile, RZ;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1      MOV32I m%d, 0xf149f2ca;\n--:-:-:-:1      MOV den%d, RZ;", h, h))
end
for q = 0, 23 do
  emit(string.format("--:-:-:-:1      MOV o%02d, RZ;", q))
end
-- ---- a tile: this thread's key kk (list block jj = 16 tile + kk / 4, key i = kk & 3), dims 16 qq + 64 c + e (c, e < 4 x 16)
emit(heredoc([==[
    TILE:
    --:-:-:-:6      SHR.U32 jj, kk, 0x2;
    --:-:-:-:6      ISCADD jj, tile, jj, 0x4;
    --:-:-:-:6      LOP32I.AND i, kk, 0x3;
    --:-:-:-:6      ISCADD slot, jj, i, 0x2;
    --:-:-:-:6      ISETP.LT.U32.AND P3, PT, slot, n, PT;
    --:-:-:-:6      ISETP.LT.U32.AND P4, PT, jj, ns, P6;
    --:-:-:-:6      ISETP.GE.U32.AND P5, PT, jj, ns, PT;
    --:-:-:-:6      MOV src, jj;
    --:-:-:-:6      SHL tt, p, 0x9;
    --:-:-:-:6      IADD tt, tt, jj;
    --:-:-:-:6      SHL tt, tt, 0x1;
    --:-:-:-:6      IADD ap0.CC, tt, param_SEL0;
    --:-:-:-:6      IADD.X ap1, RZ, param_SEL1;
    --:-:2:-:2  @P4 LDG.E.U16 src, [ap];
    02:-:-:-:6  @P5 MOV src, nv;
    --:-:-:-:6      ISCADD T, src, i, 0x2;
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, qq, RZ, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @!P3 MOV T, RZ;
    --:-:-:-:6      SHL ta, kk, 0x2;
    --:-:-:-:6      NOP;
    --:1:-:-:1  @P0 STS [ta+0x6100], T;
    --:-:-:-:6      SHL tt, T, 0x9;
    --:-:-:-:6      IADD tt, tt, qsb;
    --:-:-:-:6      SHR.U32 tm, qsb, 0x2;
    --:-:-:-:6      IADD tt, tt, -tm;
    --:-:-:-:6      ISCADD tt, qq, tt, 0x4;
    --:-:-:-:6      IADD ap0.CC, tt, param_R0;
    --:-:-:-:6      IADD.X ap1, RZ, param_R1;
    --:-:2:-:1      LDG.E.128 kc00, [ap];
    --:-:2:-:1      LDG.E.128 kc04, [ap+0x40];
    --:-:2:-:1      LDG.E.128 kc08, [ap+0x80];
    --:-:2:-:1      LDG.E.128 kc12, [ap+0xc0];
    --:-:-:-:6      SHL tt, T, 0x6;
    --:-:-:-:6      SHR.U32 tm, qq, 0x1;
    --:-:-:-:6      ISCADD tt, tm, tt, 0x2;
    --:-:-:-:6      IADD bp0.CC, tt, param_RS0;
    --:-:-:-:6      IADD.X bp1, RZ, param_RS1;
    --:-:2:-:1      LDG.E ks0, [bp];
    --:-:2:-:1      LDG.E ks1, [bp+0x8];
    --:-:2:-:1      LDG.E ks2, [bp+0x10];
    --:-:2:-:2      LDG.E ks3, [bp+0x18];
]==]))
-- (the K pointer: R + 512 T + 16 qq; qsb = 64 qq is added and 16 qq taken off again: tt = 512 T + 64 qq - 16 qq + 16 qq ... )
out_sub("--:-:-:-:6      IADD tt, tt, qsb;\n--:-:-:-:6      SHR.U32 tm, qsb, 0x2;\n--:-:-:-:6      IADD tt, tt, -tm;\n--:-:-:-:6      ISCADD tt, qq, tt, 0x4;\n",
  "--:-:-:-:6      ISCADD tt, qq, tt, 0x4;\n")
-- 4 chunks c of 16 dims: q of the 6 heads from shared memory (QS[h][d] at 1024 h + 4 d; d = 64 c + 16 qq + 4 e4 + b)
for c = 0, 3 do
  for e4 = 0, 3 do
    local word = string.format("kc%02d", 4 * c + e4)
    for b = 0, 3 do
      local w = "--"
      if c == 0 and e4 == 0 and b == 0 then w = "02" end
      emit(string.format("%s:-:-:-:1      PRMT fv%d, %s, 0x%x, c4b;", w, b, word, 0x7540 + b))
    end
    for h = 0, NH - 1 do
      emit(string.format("--:-:3:-:1      LDS.U.128 qa%02d, [qsb+0x%x];", 4 * h, 1024 * h + 256 * c + 16 * e4))
    end
    -- (the first FMUL reads fv0 four instructions after its FADD: the last FADD stalls 3, so the ALU result is there after 6 cycles even
    -- when the qa loads have already returned)
    for b = 0, 3 do
      emit(string.format("--:-:-:-:%d      FADD fv%d, fv%d, nbias;", b == 3 and 3 or 1, b, b))
    end
    for b = 0, 3 do
      for h = 0, NH - 1 do
        local first = (e4 == 0 and b == 0)
        local wait = "--"
        if b == 0 and h == 0 then wait = "04" end
        if first then
          emit(string.format("%s:-:-:-:1      FMUL pt%d, qa%02d, fv%d;", wait, h, 4 * h + b, b))
        else
          emit(string.format("%s:-:-:-:1      FFMA pt%d, qa%02d, fv%d, pt%d;", wait, h, 4 * h + b, b, h))
        end
      end
    end
  end
  for h = 0, NH - 1 do
    if c == 0 then
      emit(string.format("--:-:-:-:1      FMUL s%d, ks%d, pt%d;", h, c, h))
    else
      emit(string.format("--:-:-:-:1      FFMA s%d, ks%d, pt%d, s%d;", h, c, h, h))
    end
  end
end
emit("--:-:-:-:6      NOP;")
-- the key's score: the 4 threads' parts (xor 1, 2), invalid keys far below any score
emit(bfly(names("s", NH), names("tmp", NH), { 1, 2 }, "sum"))
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1  @!P3 MOV s%d, negbig;", h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1      MOV pt%d, s%d;", h, h))
end
emit("--:-:-:-:6      NOP;")
emit(bfly(names("pt", NH), names("tmp", NH), { 4, 8, 16 }, "max"))
emit(heredoc([==[
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, lane, RZ, PT;
    --:-:-:-:6      SHL tt, w, 0x2;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:1:-:-:1  @P0 STS [tt+0x%x], pt%d;", 0x6200 + 32 * h, h))
end
emit("01:-:-:-:5      BAR.SYNC 0x0;")
for h = 0, NH - 1 do
  emit(string.format("--:-:2:-:1      LDS.U.128 kc00, [RZ+0x%x];\n--:-:2:-:2      LDS.U.128 kc04, [RZ+0x%x];", 0x6200 + 32 * h, 0x6210 + 32 * h))
  emit(string.format("02:-:-:-:6      FMNMX pt%d, kc00, kc01, !PT;", h))
  for _, k in ipairs({ "kc02", "kc03", "kc04", "kc05", "kc06", "kc07" }) do
    emit(string.format("--:-:-:-:6      FMNMX pt%d, pt%d, %s, !PT;", h, h, k))
  end
  -- pt = the new max; cr = exp(m - pt); s = exp(s - pt) (0 for invalid keys); m = pt
  emit(string.format("--:-:-:-:6      FMNMX pt%d, pt%d, m%d, !PT;\n--:-:-:-:6      FADD cr%d, m%d, -pt%d;\n--:-:-:-:6      FADD s%d, s%d, -pt%d;\n--:-:-:-:6      MOV m%d, pt%d;", h, h, h, h, h, h, h, h, h, h, h))
end
local ex2in = {}
for _, v in ipairs(names("cr", NH)) do ex2in[#ex2in + 1] = v end
for _, v in ipairs(names("s", NH)) do ex2in[#ex2in + 1] = v end
emit(ex2(ex2in, "l2e"))
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1  @!P3 MOV s%d, RZ;", h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:1      MOV pt%d, s%d;", h, h))
end
emit("--:-:-:-:6      NOP;")
emit(bfly(names("pt", NH), names("tmp", NH), { 4, 8, 16 }, "sum"))
emit(heredoc([==[
    --:-:-:-:6      ISETP.EQ.U32.AND P0, PT, lane, RZ, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P1, PT, qq, RZ, PT;
    --:-:-:-:6      SHL tt, w, 0x2;
    --:-:-:-:6      SHL tm, kk, 0x5;
    --:-:-:-:6      NOP;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:1:-:-:1  @P0 STS [tt+0x%x], pt%d;", 0x6300 + 32 * h, h))
end
emit("--:1:-:-:1  @P1 STS.128 [tm+0x6400], s0;\n--:1:-:-:1  @P1 STS.64 [tm+0x6410], s4;")
emit("01:-:-:-:5      BAR.SYNC 0x0;")
for h = 0, NH - 1 do
  emit(string.format("--:-:2:-:1      LDS.U.128 kc00, [RZ+0x%x];\n--:-:2:-:2      LDS.U.128 kc04, [RZ+0x%x];", 0x6300 + 32 * h, 0x6310 + 32 * h))
  emit(string.format("02:-:-:-:6      FMUL den%d, den%d, cr%d;", h, h, h))
  for _, k in ipairs({ "kc00", "kc01", "kc02", "kc03", "kc04", "kc05", "kc06", "kc07" }) do
    emit(string.format("--:-:-:-:6      FADD den%d, den%d, %s;", h, h, k))
  end
end
-- ---- V: thread (r, dg) over keys r + 4 mm (mm < 16) of the tile, 4 keys at a time; dims 4 dg .. + 3
for q = 0, 23 do
  emit(string.format("--:-:-:-:1      FMUL o%02d, o%02d, cr%d;", q, q, q // 4))
end
local TRS = { "i", "jj", "src", "slot" }
for batch = 0, 3 do
  for k = 0, 3 do
    local kk2 = 16 * batch + 4 * k
    emit(string.format("--:-:2:-:1      LDS.U %s, [rk+0x%x];", TRS[k + 1], 0x6100 + 4 * kk2))
  end
  emit("02:-:-:-:6      NOP;")
  for k = 0, 3 do
    local kk2 = 16 * batch + 4 * k
    local tr = TRS[k + 1]
    emit(heredoc([==[
        --:-:-:-:6      SHL tt, #{tr}, 0x9;
        --:-:-:-:6      ISCADD tt, dg, tt, 0x2;
        --:-:-:-:6      IADD tt, tt, 0x100;
        --:-:-:-:6      IADD xp#{xa}.CC, tt, param_R0;
        --:-:-:-:6      IADD.X xp#{xb}, RZ, param_R1;
        --:-:3:-:1      LDG.E vw#{k}, [xp#{xa}];
        --:-:-:-:6      SHL tt, #{tr}, 0x6;
        --:-:-:-:6      SHR.U32 tm, dg, 0x3;
        --:-:-:-:6      ISCADD tt, tm, tt, 0x2;
        --:-:-:-:6      IADD tt, tt, 0x20;
        --:-:-:-:6      IADD xp#{xc}.CC, tt, param_RS0;
        --:-:-:-:6      IADD.X xp#{xd}, RZ, param_RS1;
        --:-:3:-:1      LDG.E vs#{k}, [xp#{xc}];
    ]==], {
      tr = tr,
      k = string.format("%d", k),
      xa = string.format("%02d", 2 * k),
      xb = string.format("%02d", 2 * k + 1),
      xc = string.format("%02d", 2 * k + 8),
      xd = string.format("%02d", 2 * k + 9),
    }))
    local pb = ({ "kc", "qa" })[k // 2 + 1]
    local po = 8 * (k % 2)
    emit(string.format("--:-:2:-:1      LDS.U.128 %s%02d, [rp+0x%x];\n--:-:2:-:1      LDS.U.64 %s%02d, [rp+0x%x];", pb, po, 0x6400 + 32 * kk2, pb, po + 4, 0x6410 + 32 * kk2))
  end
  emit("06:-:-:-:6      NOP;")
  for k = 0, 3 do
    local pb = ({ "kc", "qa" })[k // 2 + 1]
    local po = 8 * (k % 2)
    for h = 0, NH - 1 do
      emit(string.format("--:-:-:-:1      FMUL %s%02d, %s%02d, vs%d;", pb, po + h, pb, po + h, k))
    end
    for b = 0, 3 do
      emit(string.format("--:-:-:-:1      PRMT fv%d, vw%d, 0x%x, c4b;", b, k, 0x7540 + b))
    end
    emit("--:-:-:-:2      NOP;")
    for b = 0, 3 do
      emit(string.format("--:-:-:-:1      FADD fv%d, fv%d, nbias;", b, b))
    end
    emit("--:-:-:-:2      NOP;")
    for b = 0, 3 do
      for h = 0, NH - 1 do
        emit(string.format("--:-:-:-:1      FFMA o%02d, %s%02d, fv%d, o%02d;", 4 * h + b, pb, po + h, b, 4 * h + b))
      end
    end
  end
end
out_gsub("[rk+", "[rk4+")
out_gsub("[rp+", "[rp32+")
emit(heredoc([==[
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      IADD tile, tile, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, tile, ntile, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA TILE;
]==]))
-- ---- the four key streams summed through shared memory OR[r][h][d] at 4 (1536 r + 256 h + d), then out = o / den sigmoid(g)
emit(heredoc([==[
    --:-:-:-:6      XMAD tt, r, 0x600, RZ;
    --:-:-:-:6      ISCADD tt, dg, tt, 0x2;
    --:-:-:-:6      SHL tt, tt, 0x2;
]==]))
for h = 0, NH - 1 do
  emit(string.format("--:1:-:-:1      STS.128 [tt+0x%x], o%02d;", 1024 * h, 4 * h))
end
emit("01:-:-:-:5      BAR.SYNC 0x0;\n--:-:-:-:6      SHL tt, tid, 0x2;")
for h = 0, NH - 1 do
  emit(string.format("--:-:2:-:1      LDS.U kc00, [tt+0x%x];\n--:-:2:-:1      LDS.U kc01, [tt+0x%x];\n--:-:2:-:1      LDS.U kc02, [tt+0x%x];\n--:-:2:-:2      LDS.U kc03, [tt+0x%x];",
    1024 * h, 6144 + 1024 * h, 12288 + 1024 * h, 18432 + 1024 * h))
  emit(string.format("02:-:-:-:6      FADD kc00, kc00, kc01;\n--:-:-:-:6      FADD kc02, kc02, kc03;\n--:-:-:-:6      FADD o%02d, kc00, kc02;", 4 * h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:3:-:1      MUFU.RCP den%d, den%d;", h, h))
end
emit("04:-:-:-:6      NOP;")
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      FMUL o%02d, o%02d, den%d;", 4 * h, 4 * h, h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      FMUL g%d, g%d, -l2e;", h, h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      RRO.EX2 g%d, g%d;", h, h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:3:-:1      MUFU.EX2 g%d, g%d;", h, h))
end
emit("04:-:-:-:6      NOP;")
for h = 0, NH - 1 do
  emit(string.format("--:-:-:-:6      FADD g%d, g%d, 1;", h, h))
end
for h = 0, NH - 1 do
  emit(string.format("--:-:3:-:1      MUFU.RCP g%d, g%d;", h, h))
end
emit("04:-:-:-:6      NOP;")
for h = 0, NH - 1 do
  emit(heredoc([==[
      --:-:-:-:6      FMUL o#{o4}, o#{o4}, g#{h};
      --:-:-:-:6      MOV hh, param_hq0;
      --:-:-:-:6      IADD hh, hh, 0x#{hh};
      --:-:-:-:6      ISCADD tt, hh, tid, 0x8;
      --:-:-:-:6      SHL tt, tt, lgx;
      --:-:-:-:6      IADD tt, tt, p;
      --:-:-:-:6      SHL tt, tt, 0x2;
      --:-:-:-:6      IADD xp#{xa}.CC, tt, param_LAO0;
      --:-:-:-:6      IADD.X xp#{xb}, RZ, param_LAO1;
      --:-:-:-:6      ISETP.GT.U32.AND P0, PT, param_nhh, 0x#{hh}, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:1:-:-:1  @P0 STG.E [xp#{xa}], o#{o4};
  ]==], {
    o4 = string.format("%02d", 4 * h),
    h = string.format("%d", h),
    hh = hx(h),
    xa = string.format("%02d", 2 * h),
    xb = string.format("%02d", 2 * h + 1),
  }))
end
emit("01:-:-:-:f      EXIT;")
end

-- the V phase's shared-memory bases: TS[kk] at 0x6100 + 4 kk, P[kk] at 0x6400 + 32 kk for kk = r + 4 mm
if OP == "attn" then
  out_sub("--:-:-:-:6      MOV tile, RZ;\n", "--:-:-:-:6      MOV tile, RZ;\n--:-:-:-:6      SHL rk4, r, 0x2;\n--:-:-:-:6      SHL rp32, r, 0x5;\n")
end

-- A barrier an instruction sets is active a cycle after it issues: an instruction that waits on it right after (stall 1) can pass
-- before it is set. Such stalls become 2.

-- Ruby's lines[i].match(/^(\h\h|--):(\d|-):(\d|-):-:1(?=\s)/): the wait field, the read and write barriers, and the position
-- just after the stall digit when the line is a stall-1 line followed by whitespace (nil otherwise)
local function stall1(line)
  local wait, rd, wr, after = line:match("^([%x%-][%x%-]):([%d%-]):([%d%-]):%-:1()")
  if not wait then return nil end
  if wait ~= "--" and not wait:match("^%x%x$") then return nil end
  if not line:sub(after, after):match("%s") then return nil end
  return { rd, wr, after }
end

local function barfix(text)
  local lines = {}
  local pos = 1
  while pos <= #text do
    local nl = text:find("\n", pos, true)
    if nl then
      lines[#lines + 1] = text:sub(pos, nl)
      pos = nl + 1
    else
      lines[#lines + 1] = text:sub(pos)
      break
    end
  end
  local n = #lines
  for i = 1, n - 1 do
    local ca = stall1(lines[i])
    if ca then
      local j = i + 1
      while j <= n and (lines[j]:match("^%s*$") or lines[j]:match("^%s*[%w_%.]+:%s*$") or lines[j]:match("^%s*#")) do
        j = j + 1
      end
      local cb = j <= n and lines[j]:match("^(%x%x):")
      if cb then
        local mask = tonumber(cb, 16)
        local hit = false
        for _, bar in ipairs({ ca[1], ca[2] }) do
          if bar ~= "-" and (mask & (1 << (tonumber(bar) - 1))) ~= 0 then hit = true end
        end
        if hit then
          local after = ca[3]
          lines[i] = lines[i]:sub(1, after - 2) .. "2" .. lines[i]:sub(after)
        end
      end
    end
  end
  return table.concat(lines)
end

-- CA_SAFE=1 (debugging): every instruction that sets a barrier is followed by a NOP waiting on all six, so no read can race its load
-- (before barfix: the setter's stall then becomes 2)
local function safe(text)
  if not os.getenv("CA_SAFE") then return text end
  -- CA_SAFE_LO / CA_SAFE_HI: only the instructions on lines lo .. hi of the generated text (bisecting)
  local lo, hi, n, o = tonumber(os.getenv("CA_SAFE_LO") or "1"), tonumber(os.getenv("CA_SAFE_HI") or "1e9"), 0, {}
  for line in text:gmatch("[^\n]*\n") do
    n = n + 1
    o[#o + 1] = line
    local rd, wr = line:match("[%x%-][%x%-]:([%d%-]):([%d%-]):%-:%x")
    if rd and (rd ~= "-" or wr ~= "-") and n >= lo and n <= hi then o[#o + 1] = "3f:-:-:-:6      NOP;\n" end
  end
  return table.concat(o)
end

-- every exit fences the op's stores (sassfence.lua)
io.write(dofile((arg[0]:match("^(.*/)") or "") .. "sassfence.lua").fence_exits(barfix(safe(table.concat(out)))))
