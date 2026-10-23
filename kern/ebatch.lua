#!/usr/bin/env lua5.4
-- EB_OP=gu lua5.4 ebatch.lua > ebatch_gu.sass: the routed experts of a prefill chunk on one die of the unit layout (kern/hold2.c): per work
-- item of slab_group (an expert with units here and routed positions), the gate or up rows of its units against those
-- positions' activations. A die op (die.pl), 'cta' units; the CTA walks items vid, vid + units, ... while < *NITEMS.
--
--   gu   G[off + c][r] = sum_k W[r][k] Zt[LP[off + c]][k] for the item's R = 64 count rows (Q2_0 unit rows of K 2560 at
--        W + first * 46080, row stride 720: 80 B of fp16 scales then 40 x 16 B of 2-bit codes, weight d (q - 1)) and its n
--        positions (Zt: the chunk's z position-major, [ldx][2560]). G is [pair][192]. Thread t: row t mod R, K quarter / half
--        t / R (NQ = 256 / R groups: 4, 2, 1); per round every group takes its next 64-k block: the 16 columns' activations
--        staged in shared memory (Xs[group][k][16]), the row's 16 B of codes and its scale, 16 accumulators; columns in groups
--        of 16; the NQ partial sums added through shared memory at the end.
--        0x140 W  0x148 ZT  0x150 G  0x158 LP  0x160 ITEMS  0x168 NITEMS.  units = 16
--
--   dn   PART[r][LP[off + c]] += LW[off + c] sum_k WD[r][k] H[off + c][k] for all 2560 rows r and the item's K = 64 count (its
--        units' down blocks: unit slot s at WD + 46080 s, 2560 x 16 B of codes, then 2560 fp16 scales at + 40960), atomically
--        (RED.ADD of int32 fixed point, value 2^20, |sums| < 2048: several experts add to a position's column, in any order to the same sum). Columns in groups of 16: Hs[k][16] = LW H staged in shared
--        memory; thread t takes rows t + 256 m, m < 10. 0x140 WD  0x148 H  0x150 PART  0x158 LP  0x160 LW  0x168 ITEMS
--        0x170 NITEMS  0x178 ldx.  units = 16
--
-- Shared (gu): Xs at 0 (NQ x 64 x 16 floats = 16 KB), the partials at 0x4000 (256 x 16 floats = 16 KB). (dn): Hs at 0 (192 x 16
-- floats = 12 KB), the 16 positions at 0x3000.
--
-- (ported from the Ruby generator ebatch.rb, output byte-identical apart from the header line)

local OP = os.getenv("EB_OP") or "gu"   -- ENV.fetch('EB_OP', 'gu')

-- Ruby's abort: the message and a newline on stderr, then exit status 1
local function abort(msg)
  io.stderr:write(msg, "\n")
  os.exit(1)
end

if OP ~= "gu" and OP ~= "dn" and OP ~= "gu2" then
  abort("EB_OP must be gu, gu2 or dn")
end

local out = {}   -- Ruby's out (a list of the emitted pieces; joined where the text is edited)
local function emit(s)   -- Ruby's emit = ->(s) { out << s << "\n" }
  out[#out + 1] = s .. "\n"
end

-- Ruby's out.gsub! on the whole text so far: fn maps the text to the new text
local function edit_out(fn)
  out = { fn(table.concat(out)) }
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
    assert(v ~= nil, "ebatch: no value for #{" .. name .. "}")
    return v
  end))
end

local function heredoc(text, vals)
  return interp(dedent(text), vals or {})
end

-- d = a * b + c (32 x 32 -> low 32); x is a temporary, d must differ from a (the same three XMADs as gen_gemm's xmad_lo)
local function mul(d, a, b, c, x)
  return string.format("--:-:-:-:6      XMAD.MRG %s, %s, %s.H1, RZ;\n", x, a, b)
      .. string.format("--:-:-:-:6      XMAD %s, %s, %s, %s;\n", d, a, b, c)
      .. string.format("--:-:-:-:6      XMAD.PSL.CBCC %s, %s.H1, %s.H1, %s;\n", d, a, x, d)
end

-- hook: code right after the BFEs (q0..q3 are free from there: dn loads the next unit's codes into them)
local function kloop(cg, hook)
  local s = {}
  local function add(str) s[#s + 1] = str end
  for k = 0, 63 do
    add(string.format("--:-:-:-:1      BFE.U32 wv%02d, q%d, 0x%x;\n", k, k // 16, (2 << 8) | (2 * (k % 16))))
  end
  add(hook or "")
  for k = 0, 63 do
    add(string.format("--:-:-:-:1      IADD wv%02d, wv%02d, mag;\n", k, k))
  end
  for k = 0, 63 do
    add(string.format("--:-:-:-:1      FADD wv%02d, wv%02d, moff;\n", k, k))
  end
  -- the activations of k: buffer k mod D of D (registers x / xb in cg-wide pieces), loaded D - 1 k ahead on barrier bars[k mod D] of
  -- its own (dn: four buffers on barriers 5, 6, 1 and 3 for cg <= 8; gu and cg 16: two on 5 and 6)
  local D, bufs, bars = 2, { { "x", 0 }, { "xb", 0 } }, { 5, 6 }
  if OP == "dn" and cg == 8 then D, bufs, bars = 4, { { "x", 0 }, { "x", 8 }, { "xb", 0 }, { "xb", 8 } }, { 5, 6, 1, 3 }
  elseif OP == "dn" and cg == 4 then D, bufs, bars = 4, { { "x", 0 }, { "x", 4 }, { "x", 8 }, { "x", 12 } }, { 5, 6, 1, 3 } end
  local function reg(k, c) local b = bufs[k % D + 1]; return string.format("%s%02d", b[1], b[2] + c) end
  local function bar(k) return bars[k % D + 1] end
  local function lds(k)
    local t = {}
    for j = 0, cg // 4 - 1 do
      t[#t + 1] = string.format("--:-:%d:-:1      LDS.U.128 %s, [sm+0x%x];\n", bar(k), reg(k, 4 * j), 64 * k + 16 * j)
    end
    return table.concat(t)
  end
  for k = 0, D - 2 do add(lds(k)) end
  for k = 0, 63 do
    if k + D - 1 <= 63 then add(lds(k + D - 1)) end
    for c = 0, cg - 1 do
      local acc
      if cg == 4 and k % 2 == 1 then acc = string.format("sb%d", c) else acc = string.format("s%02d", c) end
      local first
      if cg == 4 then first = k < 2 else first = (k == 0) end
      local w = "--"
      if c == 0 then w = string.format("%02x", 1 << (bar(k) - 1)) end
      if first then
        add(string.format("%s:-:-:-:1      FMUL %s, wv%02d, %s;\n", w, acc, k, reg(k, c)))
      else
        add(string.format("%s:-:-:-:1      FFMA %s, wv%02d, %s, %s;\n", w, acc, k, reg(k, c), acc))
      end
    end
  end
  add("--:-:-:-:6      NOP;\n")
  if cg == 4 then
    for c = 0, 3 do
      add(string.format("--:-:-:-:1      FADD s%02d, s%02d, sb%d;\n", c, c, c))
    end
  end
  if cg == 4 then add("--:-:-:-:6      NOP;\n") end
  for c = 0, cg - 1 do
    local w = "--"
    if c == 0 then w = "08" end
    add(string.format("%s:-:-:-:1      FFMA acc%02d, dsc, s%02d, acc%02d;\n", w, c, c, c))
  end
  return table.concat(s)
end

-- the three loops, chosen by cgw (uniform per CTA)
local function kloops(tag, hook)
  return "--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, cgw, 0x10, PT;\n--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, cgw, 0x8, PT;\n"
    .. "--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6  @P0 BRA K16" .. tag .. ";\n--:-:-:-:6  @P1 BRA K8" .. tag .. ";\n"
    .. kloop(4, hook) .. "--:-:-:-:6      BRA KE" .. tag .. ";\nK8" .. tag .. ":\n"
    .. kloop(8, hook) .. "--:-:-:-:6      BRA KE" .. tag .. ";\nK16" .. tag .. ":\n"
    .. kloop(16, hook) .. "KE" .. tag .. ":\n"
end

-- a column group: cgw = 16, 8 or 4 columns from c0 = cg (rem = n - c0 >= 9, >= 5, else), ncol = min(rem, cgw)
local COLW = heredoc([==[
  --:-:-:-:6      IADD rem, n, -cg;
  --:-:-:-:6      ISETP.GE.U32.AND P5, PT, rem, 0x9, PT;
  --:-:-:-:6      ISETP.GE.U32.AND P6, PT, rem, 0x5, PT;
  --:-:-:-:6      MOV32I cgw, 0x4;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P6 MOV32I cgw, 0x8;
  --:-:-:-:6  @P5 MOV32I cgw, 0x10;
  --:-:-:-:6      NOP;
  --:-:-:-:6      IMNMX.U32 ncol, rem, cgw, PT;
]==])

local PARAMS
if OP == "gu" then
  PARAMS = { "W0", "W1", "ZT0", "ZT1", "G0", "G1", "LP0", "LP1", "IT0", "IT1", "NI0", "NI1" }
elseif OP == "gu2" then
  PARAMS = { "W0", "W1", "ZT0", "ZT1", "G0", "G1", "LP0", "LP1", "IT0", "IT1", "NI0", "NI1", "WU0", "WU1", "U0", "U1" }
else
  PARAMS = { "WD0", "WD1", "H0", "H1", "PT0", "PT1", "LP0", "LP1", "LW0", "LW1", "IT0", "IT1", "NI0", "NI1", "ldx" }
end
emit("# Kernel: ebatch_" .. OP .. "  (generated by EB_OP=" .. OP .. " ebatch.lua; do not edit)")
emit("<CONSTANT_MAPPING>")
for i, p in ipairs(PARAMS) do
  emit(string.format("    param_%-4s : c[0x0][0x%x]", p, 0x140 + 4 * (i - 1)))
end
emit("    gridDimX   : c[0x0][0x14]")
emit("</CONSTANT_MAPPING>")

if OP == "gu" then
emit(heredoc([==[
  <REGISTER_MAPPING>
      0-15   : acc<00-15>
      16-31  : x<00-15>
      32-35  : q<0-3>
      36     : dsc
      37     : tid
      38     : item
      39     : nitems
      40     : count
      41     : first
      42     : off
      43     : n
      44     : R
      45     : r
      46     : kq
      47     : nq
      48     : rnd
      49     : nrnd
      50     : cg
      51     : t
      52     : t2
      53     : wq
      54     : f
      55     : w
      56     : sm
      57     : smx
      58     : lc
      59     : lk
      60     : lg
      61     : pos
      62,63  : wp<0-1>
      64,65  : sp<0-1>
      66,67  : ip<0-1>
      68,69  : zp<0-1>
      70,71  : gp<0-1>
      72-75  : z<0-3>
      76     : mag
      77     : moff
      78     : ncol
      79     : act
      80     : kb
      81     : tm
      82     : krow
      83-146 : wv<00-63>
      147-162 : s<00-15>
      163-166 : sb<0-3>
      168-183 : xb<00-15>
      184    : cgw
      185    : rem
      186    : dscn
      188-203 : y<00-15>
  </REGISTER_MAPPING>
]==]))

emit(heredoc([==[
  --:-:1:-:1      S2R tid, SR_TID.X;
  --:-:1:-:1      S2R item, SR_CTAID.X;
  --:-:-:-:6      MOV ip0, param_NI0;
  --:-:-:-:6      MOV ip1, param_NI1;
  --:-:2:-:2      LDG.E nitems, [ip];
  --:-:-:-:6      MOV32I mag, 0x4b400000;
  --:-:-:-:6      MOV32I moff, 0xcb400001;
  ITEM:
  03:-:-:-:6      ISETP.GE.U32.AND P0, PT, item, nitems, PT;
  --:-:-:-:6      SHL t, item, 0x4;
  --:-:-:-:6      IADD ip0.CC, t, param_IT0;
  --:-:-:-:6      IADD.X ip1, RZ, param_IT1;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 EXIT;
  --:-:2:-:1      LDG.E t, [ip];
  --:-:2:-:1      LDG.E off, [ip+0x4];
  --:-:2:-:2      LDG.E n, [ip+0x8];
  02:-:-:-:6      LOP32I.AND count, t, 0xff;
  --:-:-:-:6      SHR.U32 first, t, 0x8;
  --:-:-:-:6      SHL R, count, 0x6;
]==]))
-- the thread's row and K group: count 1: r = t & 63, kq = t >> 6 (nq 4); count 2: r = t & 127, kq = t >> 7 (nq 2); count 3: r = t,
-- kq = 0 (nq 1, threads 192.. idle: act = 0); nrnd = 40 / nq rounds of 64 k
emit(heredoc([==[
  --:-:-:-:6      ISETP.EQ.U32.AND P1, PT, count, 0x1, PT;
  --:-:-:-:6      ISETP.EQ.U32.AND P2, PT, count, 0x2, PT;
  --:-:-:-:6      MOV r, tid;
  --:-:-:-:6      MOV kq, RZ;
  --:-:-:-:6      MOV32I nq, 0x1;
  --:-:-:-:6      MOV32I nrnd, 0x28;
  --:-:-:-:6  @P1 LOP32I.AND r, tid, 0x3f;
  --:-:-:-:6  @P1 SHR.U32 kq, tid, 0x6;
  --:-:-:-:6  @P1 MOV32I nq, 0x4;
  --:-:-:-:6  @P1 MOV32I nrnd, 0xa;
  --:-:-:-:6  @P2 LOP32I.AND r, tid, 0x7f;
  --:-:-:-:6  @P2 SHR.U32 kq, tid, 0x7;
  --:-:-:-:6  @P2 MOV32I nq, 0x2;
  --:-:-:-:6  @P2 MOV32I nrnd, 0x14;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, r, R, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      SEL act, RZ, 0x1, !P3;
  --:-:-:-:6      MOV cg, RZ;
]==]))
-- column group cg: columns 16 cg .. min(n, 16 cg + 16)
emit("COLS:\n" .. COLW)
for i = 0, 15 do
  emit(string.format("--:-:-:-:1      MOV acc%02d, RZ;", i))
end
-- the row's codes / scales: W + (first 64 + r) 720; round rnd of group kq covers k block kb = kq nrnd + rnd: codes at wp + 80 + 16 kb,
-- the scale at wp + 2 kb (the staging thread's block is its own group lg's, wq = lg nrnd + rnd). The activations: thread (lg = tid >> 6 group, lc = (tid >> 2) & 15 column, lk = tid & 3 k quad of 16) loads
-- the 16 k of group lg's block for column lc (k = 64 kb + 16 lk .. + 15) and stores them to Xs[lg][k][lc]; groups past nq and columns
-- past ncol load nothing (their slots are never read, or read into sums nobody stores). Pipelined: round rnd + 1's activations load
-- into y during round rnd's k loop and go to the other buffer (Xs at 0x4000 (rnd & 1)) after it, round rnd + 1's codes into q inside
-- the k loop once its BFEs have read q (kloop hook): one BAR per round, no load latency between rounds.
emit(heredoc([==[
  --:-:-:-:6      ISCADD t, first, r, 0x6;
  --:-:-:-:6      MOV32I t2, 0x2d0;
  #{wptr_mul}
  --:-:-:-:6      IADD wp0.CC, tm, param_W0;
  --:-:-:-:6      IADD.X wp1, RZ, param_W1;
  --:-:-:-:6      SHR.U32 lg, tid, 0x6;
  --:-:-:-:6      BFE.U32 lc, tid, 0x402;
  --:-:-:-:6      LOP32I.AND lk, tid, 0x3;
  --:-:-:-:6      ISETP.LT.U32.AND P4, PT, lg, nq, PT;
  --:-:-:-:6      ISETP.LT.U32.AND P6, PT, lc, ncol, PT;
  --:-:-:-:6      ISETP.NE.U32.AND P5, PT, act, RZ, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      PSETP.AND.AND P4, PT, P4, P6, PT;
  --:-:-:-:6      IADD t, cg, lc;
  --:-:-:-:6      IADD t, t, off;
  --:-:-:-:6      SHL t, t, 0x2;
  --:-:-:-:6      IADD ip0.CC, t, param_LP0;
  --:-:-:-:6      IADD.X ip1, RZ, param_LP1;
  --:-:-:-:6      MOV pos, RZ;
  --:-:2:-:2  @P4 LDG.E pos, [ip];
  02:-:-:-:6      MOV32I t2, 0xa00;
  #{stage_pos}
  --:-:-:-:6      MOV krow, tm;
  --:-:-:-:6      ISCADD t, lg, lk, 0x2;
  --:-:-:-:6      SHL t, t, 0x4;
  --:-:-:-:6      SHL t, t, 0x4;
  --:-:-:-:6      IADD t, t, lc;
  --:-:-:-:6      SHL smx, t, 0x2;
  --:-:-:-:6      MOV rnd, RZ;
  #{kb0}
  #{kx0}
  --:-:-:-:6      MOV q0, RZ;
  --:-:-:-:6      MOV q1, RZ;
  --:-:-:-:6      MOV q2, RZ;
  --:-:-:-:6      MOV q3, RZ;
  --:-:-:-:6      MOV dscn, RZ;
]==], { wptr_mul = mul("tm", "t", "t2", "RZ", "f"), stage_pos = mul("tm", "pos", "t2", "RZ", "f"), kb0 = mul("kb", "kq", "nrnd", "RZ", "tm"), kx0 = mul("wq", "lg", "nrnd", "RZ", "tm") }))
-- the activations of k block kb (register) into the 16 registers dst (barrier 3), predicate pr
local function stage_load(kb, dst, pr)
  local s = heredoc([==[
    --:-:-:-:6      ISCADD t, #{kb}, krow, 0x6;
    --:-:-:-:6      ISCADD t, lk, t, 0x4;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD zp0.CC, t, param_ZT0;
    --:-:-:-:6      IADD.X zp1, RZ, param_ZT1;
  ]==], { kb = kb })
  for i = 0, 3 do s = s .. string.format("--:-:3:-:1  @%s LDG.E.128 %s%02d, [zp+0x%x];\n", pr, dst, 4 * i, 16 * i) end
  return s
end
-- the codes / scale pointers of k block kb (sp, gp)
local function wptr(kb)
  return string.format("--:-:-:-:6      ISCADD t, %s, 0x50, 0x4;\n--:-:-:-:6      IADD sp0.CC, wp0, t;\n--:-:-:-:6      IADD.X sp1, wp1, RZ;\n--:-:-:-:6      ISCADD t, %s, RZ, 0x1;\n--:-:-:-:6      IADD gp0.CC, wp0, t;\n--:-:-:-:6      IADD.X gp1, wp1, RZ;\n", kb, kb)
end
-- round 0: its activations to Xs[0], its codes into q
for i = 0, 15 do emit(string.format("--:-:-:-:1      MOV x%02d, RZ;", i)) end
for i = 0, 15 do emit(string.format("--:-:-:-:1      MOV y%02d, RZ;", i)) end
emit(stage_load("wq", "x", "P4"))
emit(wptr("kb"))
emit("--:-:2:-:1  @P5 LDG.E.128 q0, [sp];\n--:-:2:-:2  @P5 LDG.E.U16 dscn, [gp];\n04:-:-:-:6      NOP;")
for i = 0, 15 do emit(string.format("--:1:-:-:1      STS [smx+0x%x], x%02d;", 64 * i, i)) end
emit("01:-:-:-:5      BAR.SYNC 0x0;\nROUND:")
-- P3: a next round (rnd + 1 < nrnd); P6 = P3 and P4 (its activations), P2 = P3 and P5 (its codes); kb + 1 is its block
emit(heredoc([==[
  --:-:-:-:6      IADD t2, rnd, 0x1;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, t2, nrnd, PT;
  --:-:-:-:6      IADD tm, kb, 0x1;
  --:-:-:-:6      IADD w, wq, 0x1;
  --:-:-:-:6      NOP;
  --:-:-:-:6      PSETP.AND.AND P6, PT, P3, P4, PT;
  --:-:-:-:6      PSETP.AND.AND P2, PT, P3, P5, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
]==]))
emit(stage_load("w", "y", "P6"))
emit("02:-:-:-:6      NOP;")   -- (round rnd's codes loaded through sp / gp, which read their address late: they move after)
emit(wptr("tm"))
emit(heredoc([==[
  02:-:4:-:6      F2F.F32.F16 dsc, dscn;
  --:-:-:-:6      LOP32I.AND t, rnd, 0x1;
  --:-:-:-:6      SHL t, t, 0xe;
  --:-:-:-:6      ISCADD sm, kq, t, 0xc;
]==]))
emit(kloops("g", "08:-:-:-:6      NOP;\n--:-:2:-:1  @P2 LDG.E.128 q0, [sp];\n--:-:2:-:1  @P2 LDG.E.U16 dscn, [gp];\n"))
-- round rnd + 1's activations to the other buffer (last read in round rnd - 1, before the BAR that ended it)
emit(heredoc([==[
  --:-:-:-:6      LOP32I.AND t, t2, 0x1;
  --:-:-:-:6      ISCADD t, t, smx, 0xe;
  04:-:-:-:6      NOP;
]==]))
for i = 0, 15 do emit(string.format("--:1:-:-:1  @P3 STS [t+0x%x], y%02d;", 64 * i, i)) end
emit(heredoc([==[
  01:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:6      MOV rnd, t2;
  --:-:-:-:6      IADD kb, kb, 0x1;
  --:-:-:-:6      IADD wq, wq, 0x1;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P3 BRA ROUND;
]==]))
-- the NQ partials: thread t's 16 sums to 0x4000 + 64 t; the group-0 thread of row r adds those of tid + R i, i < nq
emit("--:-:-:-:6      ISCADD t, tid, 0x4000, 0x6;")
for j = 0, 3 do
  emit(string.format("--:1:-:-:1      STS.128 [t+0x%x], acc%02d;", 16 * j, 4 * j))
end
emit(heredoc([==[
  01:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, tid, R, PT;
  --:-:-:-:6      MOV32I lk, 0x1;
  --:-:-:-:6      MOV krow, tid;
  --:-:-:-:6      NOP;
  --:-:-:-:6 @!P0 BRA PARTED;
  PART:
  --:-:-:-:6      ISETP.LT.U32.AND P1, PT, lk, nq, PT;
  --:-:-:-:6      IADD krow, krow, R;
  --:-:-:-:6      ISCADD t, krow, 0x4000, 0x6;
  --:-:-:-:6      NOP;
  --:-:-:-:6 @!P1 BRA PARTED;
]==]))
for j = 0, 3 do
  emit(string.format("--:-:2:-:1      LDS.U.128 x%02d, [t+0x%x];", 4 * j, 16 * j))
end
emit("02:-:-:-:6      NOP;")
for c = 0, 15 do
  emit(string.format("--:-:-:-:1      FADD acc%02d, acc%02d, x%02d;", c, c, c))
end
emit(heredoc([==[
  --:-:-:-:6      IADD lk, lk, 0x1;
  --:-:-:-:6      BRA PART;
  PARTED:
]==]))
-- G[off + 16 cg + c][r] for c < ncol (group 0, active threads)
emit(heredoc([==[
  --:-:-:-:6      IADD t, cg, off;
  --:-:-:-:6      MOV32I t2, 0xc0;
  #{st_mul}
  --:-:-:-:6      SHL t, tm, 0x2;
  --:-:-:-:6      IADD gp0.CC, t, param_G0;
  --:-:-:-:6      IADD.X gp1, RZ, param_G1;
]==], { st_mul = mul("tm", "t", "t2", "r", "f") }))
for c = 0, 15 do
  emit(string.format(dedent([==[
    --:-:-:-:6      ISETP.GT.U32.AND P2, PT, ncol, 0x%x, P0;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:1:-:-:1  @P2 STG.E [gp+0x%x], acc%02d;
  ]==]), c, 768 * c, c))
end
emit(heredoc([==[
  01:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:6      IADD cg, cg, cgw;
  --:-:-:-:6      MOV t, cg;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, t, n, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 BRA COLS;
  --:-:-:-:6      IADD item, item, gridDimX;
  --:-:-:-:6      BRA ITEM;
]==]))
end

-- gu2: gu for the gate (W, G) and the up (WU, U) rows of the units at once, one pass over the activations: thread (r, kq) streams both
-- rows' codes (dequantised 16 k at a time), every activation it reads from Xs feeds 2 cg FFMAs (gate and up sums 2 cg instructions
-- apart, so no second accumulator set); the partials of the K groups summed through shared memory, gate then up.
if OP == "gu2" then
emit(heredoc([==[
  <REGISTER_MAPPING>
      0-15   : acc<00-15>
      16-31  : x<00-15>
      32-35  : q<0-3>
      36     : dsc
      37     : tid
      38     : item
      39     : nitems
      40     : count
      41     : first
      42     : off
      43     : n
      44     : R
      45     : r
      46     : kq
      47     : nq
      48     : rnd
      49     : nrnd
      50     : cg
      51     : t
      52     : t2
      53     : wq
      54     : f
      55     : w
      56     : sm
      57     : smx
      58     : lc
      59     : lk
      60     : lg
      61     : pos
      62,63  : wp<0-1>
      64,65  : sp<0-1>
      66,67  : ip<0-1>
      68,69  : zp<0-1>
      70,71  : gp<0-1>
      72-75  : qu<0-3>
      76     : mag
      77     : moff
      78     : ncol
      79     : act
      80     : kb
      81     : tm
      82     : krow
      83-98  : wg<00-15>
      99-114 : wu<00-15>
      115-130 : s<00-15>
      131-146 : su<00-15>
      148-163 : au<00-15>
      164-179 : xb<00-15>
      180    : cgw
      181    : rem
      182    : dscn
      183    : dscu
      184    : dscnu
      186,187 : wpu<0-1>
      188-203 : y<00-15>
      204,205 : spu<0-1>
      206,207 : gpu<0-1>
  </REGISTER_MAPPING>
]==]))
emit(heredoc([==[
  --:-:1:-:1      S2R tid, SR_TID.X;
  --:-:-:-:6      MOV ip0, param_NI0;
  --:-:-:-:6      MOV ip1, param_NI1;
  --:-:2:-:2      LDG.E nitems, [ip];
  --:-:-:-:6      MOV32I mag, 0x4b400000;
  --:-:-:-:6      MOV32I moff, 0xcb400001;
  ITEM:
  03:-:-:-:6      ISETP.EQ.U32.AND P1, PT, tid, RZ, PT;
  --:-:-:-:6      MOV ip0, param_NI0;
  --:-:-:-:6      MOV ip1, param_NI1;
  --:-:-:-:6      MOV32I t, 0x1;
  --:-:-:-:6      NOP;
  --:-:2:-:2  @P1 ATOM.E.ADD item, [ip+0x4], t;
  02:1:-:-:1  @P1 STS [RZ+0xa000], item;
  01:-:-:-:5      BAR.SYNC 0x0;
  --:-:2:-:2      LDS item, [RZ+0xa000];
  02:-:-:-:5      BAR.SYNC 0x0;
  03:-:-:-:6      ISETP.GE.U32.AND P0, PT, item, nitems, PT;
  --:-:-:-:6      SHL t, item, 0x4;
  --:-:-:-:6      IADD ip0.CC, t, param_IT0;
  --:-:-:-:6      IADD.X ip1, RZ, param_IT1;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 EXIT;
  --:-:2:-:1      LDG.E t, [ip];
  --:-:2:-:1      LDG.E off, [ip+0x4];
  --:-:2:-:2      LDG.E n, [ip+0x8];
  02:-:-:-:6      LOP32I.AND count, t, 0xff;
  --:-:-:-:6      SHR.U32 first, t, 0x8;
  --:-:-:-:6      SHL R, count, 0x6;
  --:-:-:-:6      ISETP.EQ.U32.AND P1, PT, count, 0x1, PT;
  --:-:-:-:6      ISETP.EQ.U32.AND P2, PT, count, 0x2, PT;
  --:-:-:-:6      MOV r, tid;
  --:-:-:-:6      MOV kq, RZ;
  --:-:-:-:6      MOV32I nq, 0x1;
  --:-:-:-:6      MOV32I nrnd, 0x28;
  --:-:-:-:6  @P1 LOP32I.AND r, tid, 0x3f;
  --:-:-:-:6  @P1 SHR.U32 kq, tid, 0x6;
  --:-:-:-:6  @P1 MOV32I nq, 0x4;
  --:-:-:-:6  @P1 MOV32I nrnd, 0xa;
  --:-:-:-:6  @P2 LOP32I.AND r, tid, 0x7f;
  --:-:-:-:6  @P2 SHR.U32 kq, tid, 0x7;
  --:-:-:-:6  @P2 MOV32I nq, 0x2;
  --:-:-:-:6  @P2 MOV32I nrnd, 0x14;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, r, R, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      SEL act, RZ, 0x1, !P3;
  --:-:-:-:6      MOV cg, RZ;
]==]))
emit("COLS:\n" .. COLW)
for i = 0, 15 do emit(string.format("--:-:-:-:1      MOV acc%02d, RZ;\n--:-:-:-:1      MOV au%02d, RZ;", i, i)) end
emit(heredoc([==[
  --:-:-:-:6      ISCADD t, first, r, 0x6;
  --:-:-:-:6      MOV32I t2, 0x2d0;
  #{wptr_mul}
  --:-:-:-:6      IADD wp0.CC, tm, param_W0;
  --:-:-:-:6      IADD.X wp1, RZ, param_W1;
  --:-:-:-:6      IADD wpu0.CC, tm, param_WU0;
  --:-:-:-:6      IADD.X wpu1, RZ, param_WU1;
  --:-:-:-:6      SHR.U32 lg, tid, 0x6;
  --:-:-:-:6      BFE.U32 lc, tid, 0x402;
  --:-:-:-:6      LOP32I.AND lk, tid, 0x3;
  --:-:-:-:6      ISETP.LT.U32.AND P4, PT, lg, nq, PT;
  --:-:-:-:6      ISETP.LT.U32.AND P6, PT, lc, ncol, PT;
  --:-:-:-:6      ISETP.NE.U32.AND P5, PT, act, RZ, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      PSETP.AND.AND P4, PT, P4, P6, PT;
  --:-:-:-:6      IADD t, cg, lc;
  --:-:-:-:6      IADD t, t, off;
  --:-:-:-:6      SHL t, t, 0x2;
  --:-:-:-:6      IADD ip0.CC, t, param_LP0;
  --:-:-:-:6      IADD.X ip1, RZ, param_LP1;
  --:-:-:-:6      MOV pos, RZ;
  --:-:2:-:2  @P4 LDG.E pos, [ip];
  02:-:-:-:6      MOV32I t2, 0xa00;
  #{stage_pos}
  --:-:-:-:6      MOV krow, tm;
  --:-:-:-:6      ISCADD t, lg, lk, 0x2;
  --:-:-:-:6      SHL t, t, 0x4;
  --:-:-:-:6      SHL t, t, 0x4;
  --:-:-:-:6      IADD t, t, lc;
  --:-:-:-:6      SHL smx, t, 0x2;
  --:-:-:-:6      MOV rnd, RZ;
  #{kb0}
  #{kx0}
]==], { wptr_mul = mul("tm", "t", "t2", "RZ", "f"), stage_pos = mul("tm", "pos", "t2", "RZ", "f"), kb0 = mul("kb", "kq", "nrnd", "RZ", "tm"), kx0 = mul("wq", "lg", "nrnd", "RZ", "tm") }))
for i = 0, 3 do emit(string.format("--:-:-:-:1      MOV q%d, RZ;\n--:-:-:-:1      MOV qu%d, RZ;", i, i)) end
emit("--:-:-:-:1      MOV dscn, RZ;\n--:-:-:-:1      MOV dscnu, RZ;")
for i = 0, 15 do emit(string.format("--:-:-:-:1      MOV x%02d, RZ;\n--:-:-:-:1      MOV y%02d, RZ;", i, i)) end
local function stage_load(kb, dst, pr)
  local s = heredoc([==[
    --:-:-:-:6      ISCADD t, #{kb}, krow, 0x6;
    --:-:-:-:6      ISCADD t, lk, t, 0x4;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD zp0.CC, t, param_ZT0;
    --:-:-:-:6      IADD.X zp1, RZ, param_ZT1;
  ]==], { kb = kb })
  for i = 0, 3 do s = s .. string.format("--:-:3:-:1  @%s LDG.E.128 %s%02d, [zp+0x%x];\n", pr, dst, 4 * i, 16 * i) end
  return s
end
-- both rows' codes / scale pointers of k block kb
local function wptr2(kb)
  return string.format("--:-:-:-:6      ISCADD t, %s, 0x50, 0x4;\n--:-:-:-:6      IADD sp0.CC, wp0, t;\n--:-:-:-:6      IADD.X sp1, wp1, RZ;\n--:-:-:-:6      IADD spu0.CC, wpu0, t;\n--:-:-:-:6      IADD.X spu1, wpu1, RZ;\n"
    .. "--:-:-:-:6      ISCADD t, %s, RZ, 0x1;\n--:-:-:-:6      IADD gp0.CC, wp0, t;\n--:-:-:-:6      IADD.X gp1, wp1, RZ;\n--:-:-:-:6      IADD gpu0.CC, wpu0, t;\n--:-:-:-:6      IADD.X gpu1, wpu1, RZ;\n", kb, kb)
end
local function wload(pr)
  return string.format("--:-:2:-:1  @%s LDG.E.128 q0, [sp];\n--:-:2:-:1  @%s LDG.E.128 qu0, [spu];\n--:-:2:-:1  @%s LDG.E.U16 dscn, [gp];\n--:-:2:-:1  @%s LDG.E.U16 dscnu, [gpu];\n", pr, pr, pr, pr)
end
emit(stage_load("wq", "x", "P4"))
emit(wptr2("kb"))
emit(wload("P5") .. "04:-:-:-:6      NOP;")
for i = 0, 15 do emit(string.format("--:1:-:-:1      STS [smx+0x%x], x%02d;", 64 * i, i)) end
emit("01:-:-:-:5      BAR.SYNC 0x0;\nROUND:")
emit(heredoc([==[
  --:-:-:-:6      IADD t2, rnd, 0x1;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, t2, nrnd, PT;
  --:-:-:-:6      IADD tm, kb, 0x1;
  --:-:-:-:6      IADD w, wq, 0x1;
  --:-:-:-:6      NOP;
  --:-:-:-:6      PSETP.AND.AND P6, PT, P3, P4, PT;
  --:-:-:-:6      PSETP.AND.AND P2, PT, P3, P5, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
]==]))
emit(stage_load("w", "y", "P6"))
emit("02:-:-:-:6      NOP;")   -- (this round's codes loaded through sp / spu / gp / gpu, which read their address late)
emit(wptr2("tm"))
emit(heredoc([==[
  --:-:4:-:1      F2F.F32.F16 dsc, dscn;
  --:-:4:-:1      F2F.F32.F16 dscu, dscnu;
  --:-:-:-:6      LOP32I.AND t, rnd, 0x1;
  --:-:-:-:6      SHL t, t, 0xe;
  --:-:-:-:6      ISCADD sm, kq, t, 0xc;
]==]))
-- the k loop of one column width: four 16-k blocks, each the gate and up codes of its word dequantised, then its 16 k
local function kloop2(cg, hook)
  local s = {}
  local function add(str) s[#s + 1] = str end
  -- the activations of k: buffer k mod D, loaded D - 1 k ahead on a barrier of its own (three on 5, 6, 1 for cg <= 8: 2 and 3
  -- carry the next round's codes and activations through the loop; two for cg 16)
  local D, bufs, bars = 2, { { "x", 0 }, { "xb", 0 } }, { 5, 6 }
  if cg == 8 then D, bufs, bars = 3, { { "x", 0 }, { "x", 8 }, { "xb", 0 } }, { 5, 6, 1 }
  elseif cg == 4 then D, bufs, bars = 3, { { "x", 0 }, { "x", 4 }, { "x", 8 } }, { 5, 6, 1 } end
  local function reg(k, c) local b = bufs[k % D + 1]; return string.format("%s%02d", b[1], b[2] + c) end
  local function bar(k) return bars[k % D + 1] end
  local function lds(k)
    local u = {}
    for j = 0, cg // 4 - 1 do u[#u + 1] = string.format("--:-:%d:-:1      LDS.U.128 %s, [sm+0x%x];\n", bar(k), reg(k, 4 * j), 64 * k + 16 * j) end
    return table.concat(u)
  end
  for k = 0, D - 2 do add(lds(k)) end
  for i = 0, 3 do
    for kk = 0, 15 do add(string.format("--:-:-:-:1      BFE.U32 wg%02d, q%d, 0x%x;\n", kk, i, (2 << 8) | (2 * kk))) end
    for kk = 0, 15 do add(string.format("--:-:-:-:1      BFE.U32 wu%02d, qu%d, 0x%x;\n", kk, i, (2 << 8) | (2 * kk))) end
    if i == 3 then add(hook) end
    for kk = 0, 15 do add(string.format("--:-:-:-:1      IADD wg%02d, wg%02d, mag;\n--:-:-:-:1      IADD wu%02d, wu%02d, mag;\n", kk, kk, kk, kk)) end
    for kk = 0, 15 do add(string.format("--:-:-:-:1      FADD wg%02d, wg%02d, moff;\n--:-:-:-:1      FADD wu%02d, wu%02d, moff;\n", kk, kk, kk, kk)) end
    for kk = 0, 15 do
      local k = 16 * i + kk
      if k + D - 1 <= 63 then add(lds(k + D - 1)) end
      for c = 0, cg - 1 do
        local w = c == 0 and string.format("%02x", 1 << (bar(k) - 1)) or "--"
        if k == 0 then
          add(string.format("%s:-:-:-:1      FMUL s%02d, wg%02d, %s;\n--:-:-:-:1      FMUL su%02d, wu%02d, %s;\n", w, c, kk, reg(k, c), c, kk, reg(k, c)))
        else
          add(string.format("%s:-:-:-:1      FFMA s%02d, wg%02d, %s, s%02d;\n--:-:-:-:1      FFMA su%02d, wu%02d, %s, su%02d;\n", w, c, kk, reg(k, c), c, c, kk, reg(k, c), c))
        end
      end
    end
  end
  add("--:-:-:-:6      NOP;\n")
  for c = 0, cg - 1 do
    add(string.format("%s:-:-:-:1      FFMA acc%02d, dsc, s%02d, acc%02d;\n--:-:-:-:1      FFMA au%02d, dscu, su%02d, au%02d;\n", c == 0 and "08" or "--", c, c, c, c, c, c))
  end
  return table.concat(s)
end
local hook = "08:-:-:-:6      NOP;\n" .. wload("P2")
emit("--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, cgw, 0x10, PT;\n--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, cgw, 0x8, PT;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n"
  .. "--:-:-:-:6  @P0 BRA K16g;\n--:-:-:-:6  @P1 BRA K8g;\n" .. kloop2(4, hook) .. "--:-:-:-:6      BRA KEg;\nK8g:\n" .. kloop2(8, hook)
  .. "--:-:-:-:6      BRA KEg;\nK16g:\n" .. kloop2(16, hook) .. "KEg:")
emit(heredoc([==[
  --:-:-:-:6      LOP32I.AND t, t2, 0x1;
  --:-:-:-:6      ISCADD t, t, smx, 0xe;
  04:-:-:-:6      NOP;
]==]))
for i = 0, 15 do emit(string.format("--:1:-:-:1  @P3 STS [t+0x%x], y%02d;", 64 * i, i)) end
emit(heredoc([==[
  01:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:6      MOV rnd, t2;
  --:-:-:-:6      IADD kb, kb, 0x1;
  --:-:-:-:6      IADD wq, wq, 0x1;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P3 BRA ROUND;
]==]))
-- the NQ partials, gate (acc) then up (au): thread t's 16 sums to 0x4000 + 64 t; the group-0 thread of row r adds those of tid + R i
for _, v in ipairs({ { "acc", "G" }, { "au", "U" } }) do
  local a, lab = v[1], v[2]
  emit("--:-:-:-:6      ISCADD t, tid, 0x4000, 0x6;")
  for j = 0, 3 do emit(string.format("--:1:-:-:1      STS.128 [t+0x%x], %s%02d;", 16 * j, a, 4 * j)) end
  emit(heredoc([==[
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, tid, R, PT;
    --:-:-:-:6      MOV32I lk, 0x1;
    --:-:-:-:6      MOV krow, tid;
    --:-:-:-:6      NOP;
    --:-:-:-:6 @!P0 BRA PARTED#{lab};
    PART#{lab}:
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, lk, nq, PT;
    --:-:-:-:6      IADD krow, krow, R;
    --:-:-:-:6      ISCADD t, krow, 0x4000, 0x6;
    --:-:-:-:6      NOP;
    --:-:-:-:6 @!P1 BRA PARTED#{lab};
  ]==], { lab = lab }))
  for j = 0, 3 do emit(string.format("--:-:2:-:1      LDS.U.128 x%02d, [t+0x%x];", 4 * j, 16 * j)) end
  emit("02:-:-:-:6      NOP;")
  for c = 0, 15 do emit(string.format("--:-:-:-:1      FADD %s%02d, %s%02d, x%02d;", a, c, a, c, c)) end
  emit(heredoc([==[
    --:-:-:-:6      IADD lk, lk, 0x1;
    --:-:-:-:6      BRA PART#{lab};
    PARTED#{lab}:
    --:-:-:-:6      IADD t, cg, off;
    --:-:-:-:6      MOV32I t2, 0xc0;
    #{st_mul}
    --:-:-:-:6      SHL t, tm, 0x2;
    --:-:-:-:6      IADD #{op}0.CC, t, param_#{lab}0;
    --:-:-:-:6      IADD.X #{op}1, RZ, param_#{lab}1;
  ]==], { lab = lab, op = lab == "G" and "gp" or "spu", st_mul = mul("tm", "t", "t2", "r", "f") }))
  -- (gate and up stores through different address registers: a store's late lines re-read their address)
  for c = 0, 15 do
    emit(string.format(dedent([==[
      --:-:-:-:6      ISETP.GT.U32.AND P2, PT, ncol, 0x%x, P0;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:1:-:-:1  @P2 STG.E [%s+0x%x], %s%02d;
    ]==]), c, lab == "G" and "gp" or "spu", 768 * c, a, c))
  end
  emit("01:-:-:-:5      BAR.SYNC 0x0;")
end
emit(heredoc([==[
  --:-:-:-:6      IADD cg, cg, cgw;
  --:-:-:-:6      MOV t, cg;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, t, n, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 BRA COLS;
  --:-:-:-:6      NOP;
  --:-:-:-:6      BRA ITEM;
]==]))
end

if OP == "dn" then
emit(heredoc([==[
  <REGISTER_MAPPING>
      0-15   : acc<00-15>
      16-31  : x<00-15>
      32-35  : q<0-3>
      36     : dsc
      37     : tid
      38     : item
      39     : nitems
      40     : count
      41     : first
      42     : off
      43     : n
      44     : r
      45     : m
      46     : u
      47     : cg
      48     : ncol
      49     : t
      50     : t2
      51     : tm
      52     : wq
      53     : f
      54     : w
      55     : sm
      56     : lc
      57     : lk
      58     : wv
      59     : pos
      60     : mag
      61     : moff
      62,63  : ip<0-1>
      64,65  : hp<0-1>
      66,67  : cp<0-1>
      68,69  : dp<0-1>
      70,71  : op<0-1>
      72-75  : h<0-3>
      76     : slot
      77     : ldx4
      78-141 : wv<00-63>
      142-157 : s<00-15>
      158-161 : sb<0-3>
      164-179 : xb<00-15>
      180    : cgw
      181    : rem
      182,183 : fx<0-1>
      184    : hf
      185    : fr
      186    : dscn
  </REGISTER_MAPPING>
]==]))
emit(heredoc([==[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:-:-:6      MOV ip0, param_NI0;
    --:-:-:-:6      MOV ip1, param_NI1;
    --:-:2:-:2      LDG.E nitems, [ip];
    --:-:-:-:6      MOV32I mag, 0x4b400000;
    --:-:-:-:6      MOV32I moff, 0xcb400001;
    --:-:-:-:6      MOV t, param_ldx;
    --:-:-:-:6      SHL ldx4, t, 0x2;
    ITEM:
    03:-:-:-:6      ISETP.EQ.U32.AND P1, PT, tid, RZ, PT;
    --:-:-:-:6      MOV ip0, param_NI0;
    --:-:-:-:6      MOV ip1, param_NI1;
    --:-:-:-:6      MOV32I t, 0x1;
    --:-:-:-:6      NOP;
    --:-:2:-:2  @P1 ATOM.E.ADD item, [ip+0x8], t;
    02:1:-:-:1  @P1 STS [RZ+0xa000], item;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:2:-:2      LDS item, [RZ+0xa000];
    02:-:-:-:5      BAR.SYNC 0x0;
    03:-:-:-:6      ISETP.GE.U32.AND P0, PT, item, nitems, PT;
    --:-:-:-:6      SHL t, item, 0x4;
    --:-:-:-:6      IADD ip0.CC, t, param_IT0;
    --:-:-:-:6      IADD.X ip1, RZ, param_IT1;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 EXIT;
    --:-:2:-:1      LDG.E t, [ip];
    --:-:2:-:1      LDG.E off, [ip+0x4];
    --:-:2:-:2      LDG.E n, [ip+0x8];
    02:-:-:-:6      LOP32I.AND count, t, 0xff;
    --:-:-:-:6      SHR.U32 first, t, 0x8;
    --:-:-:-:6      MOV cg, RZ;
    COLS:
]==]))
emit(COLW)
-- stage: thread (lc = tid >> 4 column, lk = tid & 15) loads H[off + 16 cg + lc][64 u + 4 lk .. + 3] for units u < count, times the
-- pair's weight, into Hs[64 u + 4 lk + j][lc]; threads lk == 0 also put the column's position at 0x3000 + 4 lc
emit(heredoc([==[
    --:-:-:-:6      SHR.U32 lc, tid, 0x4;
    --:-:-:-:6      LOP32I.AND lk, tid, 0xf;
    --:-:-:-:6      ISETP.LT.U32.AND P4, PT, lc, ncol, PT;
    --:-:-:-:6      IADD t, cg, lc;
    --:-:-:-:6      IADD t, t, off;
    --:-:-:-:6      SHL t2, t, 0x2;
    --:-:-:-:6      IADD ip0.CC, t2, param_LW0;
    --:-:-:-:6      IADD.X ip1, RZ, param_LW1;
    --:-:-:-:6      IADD op0.CC, t2, param_LP0;
    --:-:-:-:6      IADD.X op1, RZ, param_LP1;
    --:-:-:-:6      MOV wv, RZ;
    --:-:-:-:6      MOV pos, RZ;
    --:-:2:-:1  @P4 LDG.E wv, [ip];
    --:-:2:-:1  @P4 LDG.E pos, [op];
    --:-:-:-:6      MOV32I t2, 0xc0;
    #{dstage_mul}
    --:-:-:-:6      ISCADD tm, lk, tm, 0x2;
    --:-:-:-:6      SHL tm, tm, 0x2;
    --:-:-:-:6      IADD hp0.CC, tm, param_H0;
    --:-:-:-:6      IADD.X hp1, RZ, param_H1;
    --:-:-:-:6      ISETP.EQ.U32.AND P5, PT, lk, RZ, PT;
    --:-:-:-:6      SHL t, lc, 0x2;
    --:-:-:-:6      NOP;
    02:1:-:-:1  @P5 STS [t+0x3000], pos;
]==], { dstage_mul = mul("tm", "t", "t2", "RZ", "f") }))
for uu = 0, 2 do
  emit(string.format(dedent([==[
      01:-:-:-:6      ISETP.GT.U32.AND P6, PT, count, 0x%x, P4;
      --:-:-:-:6      MOV h0, RZ;
      --:-:-:-:6      MOV h1, RZ;
      --:-:-:-:6      MOV h2, RZ;
      --:-:-:-:6      MOV h3, RZ;
      --:-:3:-:2  @P6 LDG.E.128 h0, [hp+0x%x];
      04:-:-:-:6      FMUL h0, h0, wv;
      --:-:-:-:6      FMUL h1, h1, wv;
      --:-:-:-:6      FMUL h2, h2, wv;
      --:-:-:-:6      FMUL h3, h3, wv;
      --:-:-:-:6      ISCADD t, lk, 0x%x, 0x2;
  ]==]), uu, 256 * uu, 256 * uu))
  -- Hs[64 uu + 4 lk + j][lc] at 4 ((64 uu + 4 lk + j) 16 + lc)
  emit("--:-:-:-:6      ISCADD t, t, lc, 0x4;\n--:-:-:-:6      SHL sm, t, 0x2;")
  for j = 0, 3 do
    emit(string.format("--:1:-:-:1      STS [sm+0x%x], h%d;", 64 * j, j))
  end
  emit("01:-:-:-:6      NOP;")
end
edit_out(function(text)
  return (text:gsub("ISCADD t, lk, 0x(%x+), 0x2;\n", function(h)
    return "ISCADD t, lk, RZ, 0x2;\n--:-:-:-:6      IADD t, t, 0x" .. string.format("%x", tonumber(h, 16) // 4) .. ";\n"
  end))
end)
-- the down codes of (row block m, unit u): slot = first + u at WD + 46080 slot, row r = 256 m + tid: 16 B at + 16 r, the fp16 scale
-- at + 0xa000 + 2 r. Loaded one (m, u) ahead: the first before ROW, the next inside the k loop once its BFEs have read q (kloop hook)
local function wptrs(slot, r)
  return heredoc([==[
    --:-:-:-:6      MOV32I t2, 0xb400;
    #{unit_mul}
    --:-:-:-:6      ISCADD t, #{r}, tm, 0x4;
    --:-:-:-:6      IADD cp0.CC, t, param_WD0;
    --:-:-:-:6      IADD.X cp1, RZ, param_WD1;
    --:-:-:-:6      ISCADD t, #{r}, tm, 0x1;
    --:-:-:-:6      IADD t, t, 0xa000;
    --:-:-:-:6      IADD dp0.CC, t, param_WD0;
    --:-:-:-:6      IADD.X dp1, RZ, param_WD1;
  ]==], { unit_mul = mul("tm", slot, "t2", "RZ", "f"), r = r })
end
emit("--:-:-:-:5      BAR.SYNC 0x0;\n--:-:-:-:6      MOV m, RZ;\n--:-:-:-:6      MOV slot, first;")
emit(wptrs("slot", "tid"))
emit("--:-:2:-:1      LDG.E.128 q0, [cp];\n--:-:2:-:2      LDG.E.U16 dscn, [dp];")
emit(heredoc([==[
  ROW:
  --:-:-:-:6      ISCADD r, m, tid, 0x8;
]==]))
for i = 0, 15 do
  emit(string.format("--:-:-:-:1      MOV acc%02d, RZ;", i))
end
emit(heredoc([==[
  --:-:-:-:6      MOV u, RZ;
  UNIT:
  --:-:-:-:6      IADD lk, u, 0x1;
  --:-:-:-:6      ISETP.LT.U32.AND P4, PT, lk, count, PT;
  --:-:-:-:6      IADD lc, m, 0x1;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      SEL lk, lk, RZ, P4;
  --:-:-:-:6      SEL lc, m, lc, P4;
  --:-:-:-:6      ISETP.LT.U32.AND P4, PT, lc, 0xa, PT;
  --:-:-:-:6      IADD slot, first, lk;
  --:-:-:-:6      ISCADD lc, lc, tid, 0x8;
  02:-:-:-:6      NOP;
]==]))
if os.getenv("EB_DN_PF") == "0" then   -- (debugging: this unit's codes loaded here and waited for, no prefetch)
  emit("--:-:-:-:6      IADD slot, first, u;")
  emit(wptrs("slot", "r"))
  emit("--:-:2:-:1      LDG.E.128 q0, [cp];\n--:-:2:-:2      LDG.E.U16 dscn, [dp];\n02:-:4:-:6      F2F.F32.F16 dsc, dscn;\n--:-:-:-:6      SHL sm, u, 0xc;")
  emit(kloops("d"))
else
emit(wptrs("slot", "lc"))   -- (after the wait: this unit's codes were loaded through cp / dp)
emit("02:-:4:-:6      F2F.F32.F16 dsc, dscn;\n--:-:-:-:6      SHL sm, u, 0xc;")
emit(kloops("d", "08:-:-:-:6      NOP;\n--:-:2:-:1  @P4 LDG.E.128 q0, [cp];\n--:-:2:-:1  @P4 LDG.E.U16 dscn, [dp];\n"))
end
emit(heredoc([==[
  --:-:-:-:6      IADD u, u, 0x1;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, u, count, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 BRA UNIT;
]==]))
local epi = os.getenv("EB_DN_EPI") or "lds"   -- lds: each column's position from shared memory (batch: all 16 conversions, then 16 REDs, wrong in lanes 4-7 of each 8; old: a dependent LDG)
if epi == "old" or epi == "lds" then
emit(heredoc([==[
  #{outp_mul}
  --:-:-:-:6      IADD op0.CC, tm, param_PT0;
  --:-:-:-:6      IADD.X op1, RZ, param_PT1;
  --:-:-:-:6      IADD t, cg, off;
  --:-:-:-:6      SHL t, t, 0x2;
  --:-:-:-:6      IADD dp0.CC, t, param_LP0;
  --:-:-:-:6      IADD.X dp1, RZ, param_LP1;
]==], { outp_mul = mul("tm", "r", "ldx4", "RZ", "f") }))
for c = 0, 15 do
  local ld = epi == "old" and string.format("@P2 LDG.E pos, [dp+0x%x];", 4 * c) or string.format("@P2 LDS pos, [RZ+0x%x];", 0x3000 + 4 * c)
  emit(string.format(dedent([==[
    --:-:-:-:6      ISETP.GT.U32.AND P2, PT, ncol, 0x%x, PT;
    --:-:-:-:6      MOV pos, RZ;
    --:-:-:-:6      NOP;
    --:-:2:-:2  %s
    --:-:-:-:6      FMUL32I fr, acc%02d, 0x49800000;
    --:-:3:-:2      F2I.S32.F32 fx0, fr;
    02:-:-:-:6      SHL t, pos, 0x2;
    --:-:-:-:6      IADD hp0.CC, op0, t;
    --:-:-:-:6      IADD.X hp1, op1, RZ;
    --:-:-:-:6      NOP;
    04:1:-:-:1  @P2 RED.E.ADD [hp], fx0;
    01:-:-:-:6      NOP;
  ]==]), c, ld, c))
end
else
-- PART[r][pos_c] += acc_c for c < ncol; pos_c = LP[off + 16 cg + c], staged at 0x3000 + 4 c (four broadcast LDS.128, not a dependent
-- global load per column); all 16 conversions, then all 16 atomics (addresses in the free wv pairs, values in s); PART is int32 fixed
-- point, value 2^20
emit(heredoc([==[
  #{outp_mul}
  --:-:-:-:6      IADD op0.CC, tm, param_PT0;
  --:-:-:-:6      IADD.X op1, RZ, param_PT1;
  --:-:2:-:1      LDS.U.128 xb00, [RZ+0x3000];
  --:-:2:-:1      LDS.U.128 xb04, [RZ+0x3010];
  --:-:2:-:1      LDS.U.128 xb08, [RZ+0x3020];
  --:-:2:-:1      LDS.U.128 xb12, [RZ+0x3030];
]==], { outp_mul = mul("tm", "r", "ldx4", "RZ", "f") }))
for c = 0, 15 do emit(string.format("--:-:-:-:1      FMUL32I s%02d, acc%02d, 0x49800000;", c, c)) end
emit("--:-:-:-:5      NOP;")
-- (at most four conversions on one barrier: 16 F2Is on one left the upper four lanes of each eight with stale results)
for c = 0, 15 do emit(string.format("--:-:%d:-:1      F2I.S32.F32 s%02d, s%02d;", 3 + c // 4, c, c)) end
for c = 0, 15 do emit(string.format("%s:-:-:-:1      SHL xb%02d, xb%02d, 0x2;", c == 0 and "02" or "--", c, c)) end
emit("--:-:-:-:5      NOP;")
for c = 0, 15 do
  emit(string.format("--:-:-:-:6      IADD wv%02d.CC, op0, xb%02d;\n--:-:-:-:1      IADD.X wv%02d, op1, RZ;", 2 * c, c, 2 * c + 1))   -- (the carry: 6 cycles)
end
for c = 0, 15 do
  local p = (c % 2 == 0) and "P2" or "P3"
  emit(string.format("--:-:-:-:6      ISETP.GT.U32.AND %s, PT, ncol, 0x%x, PT;", p, c))
  if c > 0 then
    local pp = (c % 2 == 1) and "P2" or "P3"
    emit(string.format("%s:1:-:-:1  @%s RED.E.ADD [wv%02d], s%02d;", c == 1 and "3c" or "--", pp, 2 * (c - 1), c - 1))
  end
end
emit("--:-:-:-:6      NOP;\n--:1:-:-:1  @P3 RED.E.ADD [wv30], s15;\n01:-:-:-:6      NOP;")
end
emit(heredoc([==[
  --:-:-:-:6      IADD m, m, 0x1;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, m, 0xa, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 BRA ROW;
  --:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:6      IADD cg, cg, cgw;
  --:-:-:-:6      MOV t, cg;
  --:-:-:-:6      ISETP.LT.U32.AND P0, PT, t, n, PT;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6      NOP;
  --:-:-:-:6  @P0 BRA COLS;
  --:-:-:-:6      NOP;
  --:-:-:-:6      BRA ITEM;
]==]))
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

-- every exit fences the op's stores (sassfence.lua)
io.write(dofile((arg[0]:match("^(.*/)") or "") .. "sassfence.lua").fence_exits(barfix(table.concat(out))))
