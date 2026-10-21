#!/usr/bin/env lua5.4
-- SLAB_OP=silu|mix|write|rms|conv lua5.4 slab.lua > slab_<op>.sass: the elementwise and per-position reduction ops of the prefill
-- chunk program, on k-major slabs X[row][ldx] (a position per column, ldx a power of two >= 128, the GEMM contract of
-- gen_gemm.lua: rows of C are the next GEMM's k rows). Bodies for the persistent die kernel (die.pl), 'cta' units of 256
-- threads; every exit is warp-uniform (a split warp would not reconverge at the unit-loop end). Not bit-exact with the
-- per-position ops: exp and reciprocals use MUFU.EX2 / MUFU.RCP / MUFU.RSQ.
--
--   silu   Y = SiLU(0.25 X) for the float4s [0, us4) of X, a copy for [us4, n4) (the hc down slab: 320 rows of down,
--          then the 4 inject rows that ride along). 0x140 X  0x148 Y  0x150 n4  0x154 us4.  units = ceil(n4 / 256)
--   mix    Z[c0 + c][p] = 0.25 sum_l sigmoid(U[l cols + c][p]) XH[2560 l + c0 + c][p], c < cols (this die's z columns of the
--          hc read, U = its 4 cols rows of up). 0x140 U  0x148 XH  0x150 Z  0x158 lgx  0x15c cols  0x160 c0.
--          units = cols ldx / 1024
--   write  S[2560 l + c][p] += 2 sigmoid(0.25 D[320 + l][p]) Y[c][p] (the hc write; D = the down slab, its inject rows).
--          0x140 S  0x148 Y  0x150 D  0x158 lgx.  units = 10 ldx / 1024 ... 10240 ldx / 1024
--   rms    Y[r][p] = X[r][p] W[w(r)] rsqrt(mean_v X^2 + eps) (sigmoid(G[r][p]) when G != 0) over vectors of vlen rows,
--          r = v vlen + k, w(r) = v wvs + k: the hc norms (vlen 2560, wvs 2560) and the DeltaNet output norm with its gate
--          (vlen 128, wvs 0). CTA (v, g): positions 32 g + lane, warp kq sums rows kq klen .. + klen (klen = vlen / 8).
--          0x140 X  0x148 W  0x150 Y  0x158 G  0x160 ldx  0x164 klen  0x168 rvlen (1 / vlen)  0x16c eps  0x170 lgg
--          (log2 of ldx / 32)  0x174 wvs.  units = nvec ldx / 32
--   conv   U[c][p] = SiLU(w0 x(t-3) + w1 x(t-2) + w2 x(t-1) + w3 x(t)), t = t0 + p, x(t) = X[c][p] for p >= 0 and the
--          state ring STATE[t & 3][c] before the chunk (kern/conv1d.sass's ring); the first thread of a row then writes the
--          chunk's last four inputs into the ring. Outputs at p >= npos are zero. 0x140 X  0x148 STATE  0x150 W ([n][4])
--          0x158 U  0x160 t0  0x164 n  0x168 lgx  0x16c npos.  units = n ldx / 1024
--   bias   X[r][p] += B[r] for r < rows (in place; the ssm alpha / beta rows' dt bias). 0x140 X  0x148 B  0x150 n4 (rows ldx / 4)
--          0x154 lgx.  units = ceil(n4 / 256)
--   router the router of every position: R [513][ldx] (512 expert logits, then the shared-expert gate logit), the top ten
--          experts of column p (the larger logit first, the lower index on a tie, as ref/router.c), their softmax weights over
--          the ten (exp(m_j - m_0) / sum) and s = sigmoid(R[512][p]) into IDX [p][10] (u32), W [p][10], SW [p]. CTA g: positions
--          32 g + lane; warp q keeps the top ten of experts 64 q .. 64 q + 63, warp 0 merges the eight lists.
--          0x140 R  0x148 IDX  0x150 W  0x158 SW  0x160 ldx.  units = ldx / 32
--   tr     Y[p][r] = X[r][p] (k-major -> position-major, rows a multiple of 32): 32 x 32 tiles through shared memory.
--          0x140 X  0x148 Y  0x150 rows  0x154 lgx.  units = rows ldx / 1024
--   group  the routed pairs of a chunk sorted by expert, for this die's units (kern/hold2.c): from IDX / W [p][10] (slab_router),
--          LP [q] (position) and LW [q] (router weight) with the pairs of expert e at OFF(e) .. OFF(e) + n_e (any order inside),
--          and the work items of the die: experts with units here (EMAP[e] & 0xff > 0) and n_e > 0, ITEMS [i][4] = (EMAP[e],
--          OFF(e), n_e, e), NITEMS. One CTA (shared counters, a serial scan by thread 0). 0x140 IDX  0x148 W  0x150 EMAP
--          0x158 LP  0x160 LW  0x168 ITEMS  0x170 NITEMS  0x178 npairs (10 npos).  units = 1
--   swiglu Y = SiLU(G) U elementwise over n4 float4s. 0x140 G  0x148 U  0x150 Y  0x158 n4.  units = ceil(n4 / 256)
--   saxpy  P[r][p] += SW[p] X[r][p] (the shared expert's down rows, weighted per position; P the MoE partial, int32 value 2^20). 0x140 P  0x148 X  0x150 SW  0x158 n4
--          0x15c lgx.  units = ceil(n4 / 256)
--   zero   Y = 0 over n4 float4s. 0x140 Y  0x148 n4.  units = ceil(n4 / 256)
--   rs     the reduce-scatter of the MoE partial: P [2560][ldx] (int32, value 2^20; sent as f32) row r goes to the inbox of its owner o (the y split: the
--          5 bounds B1..B5) at slot + ((me maxr + r - B_o) ldx) 4 (TAB: the chunk xtab, u64 inbox bases), write-through, each
--          thread then MEMBAR.SYS (a send: the framework publishes the flags). 0x140 P  0x148 TAB  0x150 slot  0x154 me
--          0x158 maxr  0x15c lgx  0x160 B1 .. 0x170 B5.  units = 10 ldx / 4
--   sum    Y[r][p] = sum_d X[d maxr + r][p], d < nd, r < rows (the owner's rows of the gathered parts). 0x140 X  0x148 Y
--          0x150 n4 (rows ldx / 4)  0x154 nd  0x158 maxr  0x15c lgx.  units = ceil(n4 / 256)
--   pgate  block 1's PLE gate (kern/plegate.sass) of every position: s_l = sum_c KN[2560 l + c][p] QN[2560 l + c][p], s' = s_l / sqrt(2560),
--          gate = sigmoid(sign(s') sqrt(max(|s'|, 1e-6))), G[2560 l + c][p] = gate V[c][p]. CTA (l, g): positions 32 g + lane, warp kq
--          rows 320 kq .. + 319. 0x140 KN  0x148 QN  0x150 V  0x158 G  0x160 lgx  0x164 lgg (log2 of ldx / 32).  units = 4 ldx / 32
--   pconv  block 1's PLE conv (kern/pleconv.sass): y = w0 x(t - 9) + w1 x(t - 6) + w2 x(t - 3) + w3 x(t), x(t) = X[c][t - t0] in the chunk,
--          RI[t mod 10][c] before it (0 before position 0); S[c][p] += G[c][p] + SiLU(y); RO[t mod 10][c] = x(t) for the chunk's last 10
--          positions (RO keeps RI's entries of the positions before t0 among those). Thread (c, p). 0x140 X  0x148 RI  0x150 RO
--          0x158 W (fp16 [10240][4])  0x160 G  0x168 S  0x170 t0  0x174 t0m (t0 mod 10)  0x178 npos  0x17c lgx.  units = 40 ldx

-- Ruby: OP = ENV.fetch('SLAB_OP', 'silu')
local OP = os.getenv("SLAB_OP") or "silu"

-- Ruby's abort: the message and a newline on stderr, then exit status 1
local function abort(msg)
  io.stderr:write(msg, "\n")
  os.exit(1)
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
    assert(v ~= nil, "slab: no value for #{" .. name .. "}")
    return v
  end))
end

-- the dedent runs on the literal text, so a line that starts with #{...} has no indentation (as in Ruby)
local function heredoc(text, vals)
  return interp(dedent(text), vals or {})
end

-- Ruby's format('0x%08x', n)
local function hex8(n)
  return string.format("0x%08x", n)
end

-- 32 x 32 -> low 32 multiply (MaxAs's XMAD.LO expansion), d = a * b + c; x is a temporary, d must differ from a
-- (the same expansion as gen_gemm.lua's xmad_lo; Ruby's mul(d, a, b, c, x))
local function mul(d, a, b, c, x)
  return string.format("--:-:-:-:6      XMAD.MRG %s, %s, %s.H1, RZ;\n", x, a, b)
      .. string.format("--:-:-:-:6      XMAD %s, %s, %s, %s;\n", d, a, b, c)
      .. string.format("--:-:-:-:6      XMAD.PSL.CBCC %s, %s.H1, %s.H1, %s;\n", d, a, x, d)
end

-- Ruby's (a..b).map { |i| format(fmt, i) }
local function seq(fmt, from, to)
  local t = {}
  for i = from, to do
    t[#t + 1] = string.format(fmt, i)
  end
  return t
end

-- Ruby's ptrs lambda (rms): pointers dst[2i..2i+1] = base + o_i for i < 8, one pair each (a load reads its address late)
local function ptrs(dst, base)
  local s = {}
  for i = 0, 7 do
    s[#s + 1] = string.format("--:-:-:-:1      IADD %s%02d.CC, %s0, o%d;\n--:-:-:-:1      IADD.X %s%02d, %s1, RZ;\n", dst, 2 * i, base, i, dst, 2 * i + 1, base)
  end
  return table.concat(s)
end

-- Ruby's ptr8 lambda (pgate): pointers dst[2i..2i+1] = base + i stride for i < 8 (tm = stride i), one pair each
local function ptr8(dst, base)
  local s = {}
  for i = 0, 7 do
    s[#s + 1] = string.format("--:-:-:-:6      XMAD tm, stride, 0x%x, RZ;\n--:-:-:-:6      IADD %s%02d.CC, tm, %s0;\n--:-:-:-:6      IADD.X %s%02d, RZ, %s1;\n", i, dst, 2 * i, base, dst, 2 * i + 1, base)
  end
  return table.concat(s)
end

-- Ruby's String#to_i: leading white space, an optional sign, then decimal digits (a single underscore between digits is skipped)
local function ruby_to_i(s)
  local sign, body = s:match("^%s*([+-]?)(%d[%d_]*)")
  if not body then return 0 end
  local n = 0
  local prev_us = false
  for ch in body:gmatch(".") do
    if ch == "_" then
      if prev_us then break end
      prev_us = true
    else
      n = n * 10 + tonumber(ch)
      prev_us = false
    end
  end
  if sign == "-" then n = -n end
  return n
end

-- the lines of a string, each with its newline (Ruby's String#lines)
local function ruby_lines(text)
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
  return lines
end

local function chomp(s)
  return (s:gsub("\n$", ""))
end

-- Ruby's line =~ /^(\h\h|--):(\d|-):(\d|-):-:1(?=\s)/: returns the wait mask, the read and the write barrier, or nil
local function ctl_match(line)
  local w, rd, wr, after = line:match("^(..):(.):(.):%-:1()")
  if not w then return nil end
  if not (w == "--" or w:match("^%x%x$")) then return nil end
  if not (rd == "-" or rd:match("^%d$")) then return nil end
  if not (wr == "-" or wr:match("^%d$")) then return nil end
  if not line:sub(after, after):match("^%s$") then return nil end
  return w, rd, wr
end

local out = {}
local function emit(s)
  out[#out + 1] = s .. "\n"
end

local NL2E = 0xbfb8aa3b   -- -log2(e)
local NL2E_HEX = hex8(NL2E)

-- sigmoid of the registers xs into ys (all distinct, 4 at a time): y = 1 / (1 + 2^(-log2e x)); two write barriers
local function sigm(ys, xs, nl2e, bar1, bar2)
  bar1 = bar1 or 3
  bar2 = bar2 or 4
  local s = {}
  local n = #ys
  for i, y in ipairs(ys) do
    s[#s + 1] = string.format("--:-:-:-:%d      FMUL %s, %s, %s;\n", i == n and 6 or 1, y, xs[i], nl2e)
  end
  for i, y in ipairs(ys) do
    s[#s + 1] = string.format("--:-:-:-:%d      RRO.EX2 %s, %s;\n", i == n and 6 or 1, y, y)
  end
  for _, y in ipairs(ys) do
    s[#s + 1] = string.format("--:-:%d:-:1      MUFU.EX2 %s, %s;\n", bar1, y, y)
  end
  for i, y in ipairs(ys) do
    s[#s + 1] = string.format("%s:-:-:-:%d      FADD %s, %s, 1;\n", i == 1 and string.format("%02x", 1 << (bar1 - 1)) or "--", i == n and 6 or 1, y, y)
  end
  for _, y in ipairs(ys) do
    s[#s + 1] = string.format("--:-:%d:-:1      MUFU.RCP %s, %s;\n", bar2, y, y)
  end
  s[#s + 1] = string.format("%02x:-:-:-:6      NOP;\n", 1 << (bar2 - 1))
  return table.concat(s)
end

-- Ruby: %w[...].include?(OP)
local VALID = {}
for _, name in ipairs({ "silu", "mix", "write", "rms", "conv", "bias", "router", "tr", "group", "swiglu", "saxpy", "zero", "rs", "sum", "pgate", "pconv", "xs", "xh", "uh" }) do
  VALID[name] = true
end
if not VALID[OP] then
  abort("SLAB_OP must be silu, mix, write, rms, conv, bias, router, tr, group, swiglu, saxpy, zero, rs, sum, pgate, pconv, xs, xh or uh")
end

local PARAMS = {
  silu  = { "X0", "X1", "Y0", "Y1", "n4", "us4" },
  mix   = { "U0", "U1", "XH0", "XH1", "Z0", "Z1", "lgx", "cols", "c0" },
  write = { "S0", "S1", "Y0", "Y1", "D0", "D1", "lgx" },
  rms   = { "X0", "X1", "W0", "W1", "Y0", "Y1", "G0", "G1", "ldx", "klen", "rvlen", "eps", "lgg", "wvs" },
  conv  = { "X0", "X1", "ST0", "ST1", "W0", "W1", "U0", "U1", "t0", "n", "lgx", "npos" },
  bias  = { "X0", "X1", "B0", "B1", "n4", "lgx" },
  router = { "R0", "R1", "IDX0", "IDX1", "W0", "W1", "SW0", "SW1", "ldx" },
  tr    = { "X0", "X1", "Y0", "Y1", "rows", "lgx" },
  group = { "IDX0", "IDX1", "W0", "W1", "EM0", "EM1", "LP0", "LP1", "LW0", "LW1", "IT0", "IT1", "NI0", "NI1", "npairs" },
  swiglu = { "G0", "G1", "U0", "U1", "Y0", "Y1", "n4" },
  saxpy = { "P0", "P1", "X0", "X1", "SW0", "SW1", "n4", "lgx" },
  zero  = { "Y0", "Y1", "n4" },
  rs    = { "P0", "P1", "TAB0", "TAB1", "slot", "me", "maxr", "lgx", "B1", "B2", "B3", "B4", "B5" },
  sum   = { "X0", "X1", "Y0", "Y1", "n4", "nd", "maxr", "lgx" },
  pgate = { "KN0", "KN1", "QN0", "QN1", "V0", "V1", "G0", "G1", "lgx", "lgg" },
  pconv = { "X0", "X1", "RI0", "RI1", "RO0", "RO1", "W0", "W1", "G0", "G1", "S0", "S1", "t0", "t0m", "npos", "lgx" },
  xs    = { "TAB0", "TAB1", "SRC0", "SRC1", "n16", "myoff", "slot", "nd" },
  xh    = { "TAB0", "TAB1", "SRC0", "SRC1", "n16", "myoff", "slot", "dmask" },
  uh    = { "LOC0", "LOC1", "PAR0", "PAR1", "DST0", "DST1", "n16", "rlo0", "rhi0", "rlo1", "rhi1" },
}

emit("# Kernel: slab_" .. OP .. "  (generated by SLAB_OP=" .. OP .. " slab.lua; do not edit)")
emit("<CONSTANT_MAPPING>")
for i, p in ipairs(PARAMS[OP]) do
  emit(string.format("    param_%-6s : c[0x0][0x%x]", p, 0x140 + 4 * (i - 1)))
end
emit("</CONSTANT_MAPPING>")

if OP == "silu" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4-7   : u<0-3>
        8-11  : y<0-3>
        12    : tid
        13    : bid
        14    : q
        15    : t
        16    : nl2e
        18,19 : xp<0-1>
        20,21 : yp<0-1>
    </REGISTER_MAPPING>
  ]]))
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, q, param_n4, PT;
    --:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n4, PT;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, q, param_us4, PT;
    --:-:-:-:6      SHL t, q, 0x4;
    --:-:-:-:6      IADD xp0.CC, t, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
    --:-:-:-:6      IADD yp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6  @P2 EXIT;
    --:-:2:-:2 @!P0 LDG.E.128 x0, [xp];
    02:-:-:-:1      FMUL u0, x0, 0.25;
    --:-:-:-:1      FMUL u1, x1, 0.25;
    --:-:-:-:1      FMUL u2, x2, 0.25;
    --:-:-:-:6      FMUL u3, x3, 0.25;
  ]], { NL2E = NL2E_HEX }))
  emit(sigm({ "y0", "y1", "y2", "y3" }, { "u0", "u1", "u2", "u3" }, "nl2e"))
  for i = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL y%d, y%d, u%d;", i == 3 and 6 or 1, i, i, i))
  end
  for i = 0, 3 do
    emit(string.format("--:-:-:-:%d @!P1 MOV y%d, x%d;", i == 3 and 6 or 1, i, i))
  end
  emit("--:1:-:-:1 @!P0 STG.E.128 [yp], y0;")
  emit("01:-:-:-:f      EXIT;")

elseif OP == "mix" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-15  : u<00-15>
        16-31 : x<00-15>
        32-35 : z<0-3>
        36    : tid
        37    : bid
        38    : q
        39    : c
        40    : p4
        41    : t
        42    : t2
        43    : nl2e
        44    : lm
        45    : rs
        46    : ldx
        47    : tm
        48-55 : up<0-7>
        56-63 : xp<0-7>
        64,65 : zp<0-1>
        66-81 : g<00-15>
    </REGISTER_MAPPING>
  ]]))
  -- thread q: z column c = q >> (lgx - 2), positions 4 p4 .. 4 p4 + 3; the 4 up rows l cols + c, the 4 xh rows 2560 l + c0 + c
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD rs, t, -0x2;
    --:-:-:-:6      MOV32I t2, 0x1;
    --:-:-:-:6      SHL ldx, t2, t;
    --:-:-:-:6      SHL lm, t2, rs;
    --:-:-:-:6      IADD lm, lm, -0x1;
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      SHR.U32 c, q, rs;
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, c, param_cols, PT;
    --:-:-:-:6      LOP32I.AND t, tid, 0xffffffe0;
    --:-:-:-:6      ISCADD t, bid, t, 0x8;
    --:-:-:-:6      SHR.U32 t, t, rs;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_cols, PT;
  ]], { NL2E = NL2E_HEX }))
  -- up row l cols + c: byte ((l cols + c) ldx + 4 p4) 4; xh row 2560 l + c0 + c
  for l = 0, 3 do
    emit(string.format(heredoc([[
      --:-:-:-:6      MOV t, param_cols;
      --:-:-:-:6      XMAD t, t, 0x%x, c;
#{M_TM}--:-:-:-:6      ISCADD t, p4, tm, 0x2;
      --:-:-:-:6      SHL t, t, 0x2;
      --:-:-:-:6      MOV t2, RZ;
      --:-:-:-:6      IADD up%d.CC, t, param_U0;
      --:-:-:-:6      IADD.X up%d, RZ, param_U1;
    ]], { M_TM = mul("tm", "t", "ldx", "RZ", "t2") }), l, 2 * l, 2 * l + 1))
    emit(string.format(heredoc([[
      --:-:-:-:6      MOV t, param_c0;
      --:-:-:-:6      IADD t, t, c;
      --:-:-:-:6      IADD t, t, 0x%x;
#{M_TM}--:-:-:-:6      ISCADD t, p4, tm, 0x2;
      --:-:-:-:6      SHL t, t, 0x2;
      --:-:-:-:6      IADD xp%d.CC, t, param_XH0;
      --:-:-:-:6      IADD.X xp%d, RZ, param_XH1;
    ]], { M_TM = mul("tm", "t", "ldx", "RZ", "t2") }), 2560 * l, 2 * l, 2 * l + 1))
  end
  emit(heredoc([[
    --:-:-:-:6      MOV t, param_c0;
    --:-:-:-:6      IADD t, t, c;
#{M_TM}--:-:-:-:6      ISCADD t, p4, tm, 0x2;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD zp0.CC, t, param_Z0;
    --:-:-:-:6      IADD.X zp1, RZ, param_Z1;
    --:-:-:-:6  @P2 EXIT;
  ]], { M_TM = mul("tm", "t", "ldx", "RZ", "t2") }))
  for l = 0, 3 do
    emit(string.format("--:-:2:-:1 @!P0 LDG.E.128 u%02d, [up%d];", 4 * l, 2 * l))
  end
  for l = 0, 3 do
    emit(string.format("--:-:2:-:1 @!P0 LDG.E.128 x%02d, [xp%d];", 4 * l, 2 * l))
  end
  emit("02:-:-:-:6      NOP;")
  emit(sigm(seq("g%02d", 0, 15), seq("u%02d", 0, 15), "nl2e"))
  -- z_j = 0.25 (g0j x0j + g1j x1j + g2j x2j + g3j x3j), position j of the four
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL z%d, g%02d, x%02d;", j == 3 and 6 or 1, j, j, j))
  end
  for l = 1, 3 do
    for j = 0, 3 do
      emit(string.format("--:-:-:-:%d      FFMA z%d, g%02d, x%02d, z%d;", j == 3 and 6 or 1, j, 4 * l + j, 4 * l + j, j))
    end
  end
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL z%d, z%d, 0.25;", j == 3 and 6 or 1, j, j))
  end
  emit("--:1:-:-:1 @!P0 STG.E.128 [zp], z0;")
  emit("01:-:-:-:f      EXIT;")

elseif OP == "write" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : s<0-3>
        4-7   : y<0-3>
        8-11  : d<0-3>
        12-15 : g<0-3>
        16    : tid
        17    : bid
        18    : q
        19    : row
        20    : p4
        21    : l
        22    : c
        23    : t
        24    : t2
        25    : nl2e
        26    : lm
        27    : rs
        28    : ldx
        29    : tm
        30,31 : sp<0-1>
        32,33 : yp<0-1>
        34,35 : dp<0-1>
    </REGISTER_MAPPING>
  ]]))
  -- thread q over S's float4s: row = q >> (lgx - 2) (< 10240), lane l = row / 2560, column c = row - 2560 l
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD rs, t, -0x2;
    --:-:-:-:6      MOV32I t2, 0x1;
    --:-:-:-:6      SHL ldx, t2, t;
    --:-:-:-:6      SHL lm, t2, rs;
    --:-:-:-:6      IADD lm, lm, -0x1;
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      SHR.U32 row, q, rs;
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, row, 0xa00, PT;
    --:-:-:-:6      ISETP.GE.U32.AND P1, PT, row, 0x1400, PT;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, row, 0x1e00, PT;
    --:-:-:-:6      MOV l, RZ;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 IADD l, l, 0x1;
    --:-:-:-:6  @P1 IADD l, l, 0x1;
    --:-:-:-:6  @P2 IADD l, l, 0x1;
    --:-:-:-:6      XMAD c, l, 0xa00, RZ;
    --:-:-:-:6      IADD c, row, -c;
    --:-:-:-:6      SHL t, q, 0x4;
    --:-:-:-:6      IADD sp0.CC, t, param_S0;
    --:-:-:-:6      IADD.X sp1, RZ, param_S1;
#{M_T}    --:-:-:-:6      ISCADD t, p4, t, 0x2;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD yp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6      IADD t, l, 0x140;
#{M_TM}--:-:-:-:6      ISCADD t, p4, tm, 0x2;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD dp0.CC, t, param_D0;
    --:-:-:-:6      IADD.X dp1, RZ, param_D1;
    --:-:2:-:1      LDG.E.128 s0, [sp];
    --:-:2:-:1      LDG.E.128 y0, [yp];
    --:-:2:-:2      LDG.E.128 d0, [dp];
    02:-:-:-:1      FMUL d0, d0, 0.25;
    --:-:-:-:1      FMUL d1, d1, 0.25;
    --:-:-:-:1      FMUL d2, d2, 0.25;
    --:-:-:-:6      FMUL d3, d3, 0.25;
  ]], { NL2E = NL2E_HEX, M_T = mul("t", "c", "ldx", "RZ", "t2"), M_TM = mul("tm", "t", "ldx", "RZ", "t2") }))
  emit(sigm({ "g0", "g1", "g2", "g3" }, { "d0", "d1", "d2", "d3" }, "nl2e"))
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FADD g%d, g%d, g%d;", j == 3 and 6 or 1, j, j, j))
  end
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FFMA s%d, g%d, y%d, s%d;", j == 3 and 6 or 1, j, j, j, j))
  end
  emit("--:1:-:-:1      STG.E.128 [sp], s0;")
  emit("01:-:-:-:f      EXIT;")

elseif OP == "rms" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-7    : x<0-7>
        8,9    : acc<0-1>
        94,95  : acc<2-3>
        10     : tid
        11     : bid
        12     : lane
        13     : kq
        14     : v
        15     : g
        16     : p
        17     : t
        18     : t2
        19     : step
        20     : k
        21     : kend
        22     : ss
        23     : r
        24     : wv
        25     : step8
        26     : nl2e
        27     : sm
        28,29  : xp<0-1>
        30,31  : yp<0-1>
        32,33  : wp<0-1>
        34,35  : gp<0-1>
        36-43  : w<0-7>
        44-51  : gg<0-7>
        52-67  : a<00-15>
        68-83  : b<00-15>
        84-91  : o<0-7>
        92,93  : x0p<0-1>
    </REGISTER_MAPPING>
  ]]))
  -- CTA (v, g): lane -> position p = 32 g + lane, warp kq -> rows v vlen + kq klen + i, i < klen (klen a multiple of 8);
  -- 8 rows per round, each through its own pointer (a load reads its address late: no pointer moves until the round's
  -- loads have returned)
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 kq, tid, 0x5;
    --:-:-:-:6      SHR.U32 v, bid, param_lgg;
    --:-:-:-:6      MOV32I t, 0x1;
    --:-:-:-:6      SHL t, t, param_lgg;
    --:-:-:-:6      IADD t, t, -0x1;
    --:-:-:-:6      LOP.AND g, bid, t;
    --:-:-:-:6      ISCADD p, g, lane, 0x5;
    --:-:-:-:6      MOV t, param_klen;
    --:-:-:-:6      SHL t2, t, 0x3;
    #{M_K}
    #{M_KQ}
    --:-:-:-:6      IADD k, k, t2;
    --:-:-:-:6      MOV t, param_ldx;
    #{M_T2}
    --:-:-:-:6      SHL t2, t2, 0x2;
    --:-:-:-:6      IADD xp0.CC, t2, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
    --:-:-:-:6      MOV x0p0, xp0;
    --:-:-:-:6      MOV x0p1, xp1;
    --:-:-:-:6      IADD yp0.CC, t2, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6      IADD gp0.CC, t2, param_G0;
    --:-:-:-:6      IADD.X gp1, RZ, param_G1;
    --:-:-:-:6      SHL step, t, 0x2;
    --:-:-:-:6      SHL step8, t, 0x5;
    --:-:-:-:6      MOV o0, RZ;
  ]], {
    NL2E = NL2E_HEX,
    M_K = mul("k", "v", "t2", "RZ", "sm"),
    M_KQ = mul("t2", "kq", "t", "RZ", "sm"),
    M_T2 = mul("t2", "k", "t", "p", "sm"),
  }))
  for i = 1, 7 do
    emit(string.format("--:-:-:-:6      IADD o%d, o%d, step;", i, i - 1))
  end
  emit(heredoc([[
    --:-:-:-:6      MOV t, param_klen;
    #{M_WV}
    --:-:-:-:6      MOV t, param_wvs;
    #{M_T2}
    --:-:-:-:6      SHL t2, t2, 0x2;
    --:-:-:-:6      IADD wp0.CC, t2, param_W0;
    --:-:-:-:6      IADD.X wp1, RZ, param_W1;
    --:-:-:-:6      MOV acc0, RZ;
    --:-:-:-:6      MOV acc1, RZ;
    --:-:-:-:6      MOV acc2, RZ;
    --:-:-:-:6      MOV acc3, RZ;
    --:-:-:-:6      MOV k, RZ;
    --:-:-:-:6      MOV kend, param_klen;
    SUMSQ:
  ]], {
    M_WV = mul("wv", "kq", "t", "RZ", "sm"),
    M_T2 = mul("t2", "v", "t", "wv", "sm"),
  }))
  emit(ptrs("a", "xp"))
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E x%d, [a%02d];", i, 2 * i))
  end
  emit(heredoc([[
    02:-:-:-:1      FFMA acc0, x0, x0, acc0;
    --:-:-:-:1      FFMA acc1, x1, x1, acc1;
    --:-:-:-:1      FFMA acc2, x2, x2, acc2;
    --:-:-:-:1      FFMA acc3, x3, x3, acc3;
    --:-:-:-:1      IADD k, k, 0x8;
    --:-:-:-:1      IADD xp0.CC, xp0, step8;
    --:-:-:-:1      IADD.X xp1, xp1, RZ;
    --:-:-:-:1      FFMA acc0, x4, x4, acc0;
    --:-:-:-:1      FFMA acc1, x5, x5, acc1;
    --:-:-:-:1      FFMA acc2, x6, x6, acc2;
    --:-:-:-:6      FFMA acc3, x7, x7, acc3;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, k, kend, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA SUMSQ;
    --:-:-:-:6      FADD acc0, acc0, acc1;
    --:-:-:-:6      FADD acc2, acc2, acc3;
    --:-:-:-:6      FADD acc0, acc0, acc2;
    --:-:-:-:6      SHL sm, tid, 0x2;
    --:1:-:-:1      STS [sm], acc0;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      SHL sm, lane, 0x2;
    --:-:2:-:1      LDS x0, [sm];
    --:-:2:-:1      LDS x1, [sm+0x80];
    --:-:2:-:1      LDS x2, [sm+0x100];
    --:-:2:-:1      LDS x3, [sm+0x180];
    --:-:2:-:1      LDS x4, [sm+0x200];
    --:-:2:-:1      LDS x5, [sm+0x280];
    --:-:2:-:1      LDS x6, [sm+0x300];
    --:-:2:-:2      LDS x7, [sm+0x380];
    02:-:-:-:6      FADD ss, x0, x1;
    --:-:-:-:6      FADD ss, ss, x2;
    --:-:-:-:6      FADD ss, ss, x3;
    --:-:-:-:6      FADD ss, ss, x4;
    --:-:-:-:6      FADD ss, ss, x5;
    --:-:-:-:6      FADD ss, ss, x6;
    --:-:-:-:6      FADD ss, ss, x7;
    --:-:-:-:6      FMUL ss, ss, param_rvlen;
    --:-:-:-:6      FADD ss, ss, param_eps;
    --:-:3:-:2      MUFU.RSQ r, ss;
    --:-:-:-:6      MOV xp0, x0p0;
    --:-:-:-:6      MOV xp1, x0p1;
    --:-:-:-:6      MOV t, param_G0;
    --:-:-:-:6      LOP.OR t, t, param_G1;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, t, RZ, PT;
    --:-:-:-:6      MOV k, RZ;
    04:-:-:-:6      NOP;
    SCALE:
  ]]))
  -- 8 rows per round: x, the weight (one address across the warp) and the gate; y = x w r (sigmoid(g))
  emit(ptrs("a", "xp"))
  emit(ptrs("b", "gp"))
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E x%d, [a%02d];", i, 2 * i))
  end
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E w%d, [wp+0x%x];", i, 4 * i))
  end
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1  @P1 LDG.E gg%d, [b%02d];", i, 2 * i))
  end
  for i = 0, 7 do
    emit(string.format("%s:-:-:-:%d      FMUL x%d, x%d, r;", i == 0 and "02" or "--", i == 7 and 6 or 1, i, i))
  end
  for i = 0, 7 do
    emit(string.format("--:-:-:-:%d      FMUL x%d, x%d, w%d;", i == 7 and 6 or 1, i, i, i))
  end
  emit("--:-:-:-:6 @!P1 BRA NOGATE;")
  emit(sigm(seq("w%d", 0, 7), seq("gg%d", 0, 7), "nl2e"))
  for i = 0, 7 do
    emit(string.format("--:-:-:-:%d      FMUL x%d, x%d, w%d;", i == 7 and 6 or 1, i, i, i))
  end
  emit("NOGATE:")
  emit(ptrs("b", "yp"))
  for i = 0, 7 do
    emit(string.format("--:1:-:-:1      STG.E [b%02d], x%d;", 2 * i, i))
  end
  emit(heredoc([[
    --:-:-:-:6      IADD k, k, 0x8;
    --:-:-:-:1      IADD xp0.CC, xp0, step8;
    --:-:-:-:1      IADD.X xp1, xp1, RZ;
    --:-:-:-:1      IADD gp0.CC, gp0, step8;
    --:-:-:-:1      IADD.X gp1, gp1, RZ;
    --:-:-:-:1      IADD yp0.CC, yp0, step8;
    --:-:-:-:1      IADD.X yp1, yp1, RZ;
    --:-:-:-:1      IADD wp0.CC, wp0, 0x20;
    --:-:-:-:6      IADD.X wp1, wp1, RZ;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, k, kend, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    01:-:-:-:6  @P0 BRA SCALE;
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:f      EXIT;
  ]]))

elseif OP == "conv" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4-7   : pv<0-3>
        8-11  : w<0-3>
        12-15 : y<0-3>
        16-19 : sg<0-3>
        20    : tid
        21    : bid
        22    : q
        23    : c
        24    : p4
        25    : t
        26    : t2
        27    : nl2e
        28    : lm
        29    : rs
        30    : ldx
        31    : tt
        32,33 : xp<0-1>
        34,35 : wp<0-1>
        36,37 : up<0-1>
        38,39 : sp<0-1>
        40    : j
        41    : pos
        42,43 : pp<0-1>
        44    : tm
    </REGISTER_MAPPING>
  ]]))
  -- thread q: channel c = q >> (lgx - 2), positions 4 p4 .. 4 p4 + 3; pv = the three inputs before them (x(4 p4 - 3 .. 4 p4 - 1))
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD rs, t, -0x2;
    --:-:-:-:6      MOV32I t2, 0x1;
    --:-:-:-:6      SHL ldx, t2, t;
    --:-:-:-:6      SHL lm, t2, rs;
    --:-:-:-:6      IADD lm, lm, -0x1;
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      SHR.U32 c, q, rs;
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, c, param_n, PT;
    --:-:-:-:6      LOP32I.AND t, tid, 0xffffffe0;
    --:-:-:-:6      ISCADD t, bid, t, 0x8;
    --:-:-:-:6      SHR.U32 t, t, rs;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P1, PT, p4, RZ, PT;
#{M_T}    --:-:-:-:6      ISCADD t, p4, t, 0x2;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD xp0.CC, t, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
    --:-:-:-:6      IADD up0.CC, t, param_U0;
    --:-:-:-:6      IADD.X up1, RZ, param_U1;
    --:-:-:-:6      SHL t, c, 0x4;
    --:-:-:-:6      IADD wp0.CC, t, param_W0;
    --:-:-:-:6      IADD.X wp1, RZ, param_W1;
    --:-:-:-:6  @P2 EXIT;
    --:-:2:-:1 @!P0 LDG.E.128 x0, [xp];
    --:-:2:-:1 @!P0 LDG.E.128 w0, [wp];
    --:-:-:-:6      PSETP.AND.AND P3, PT, P1, !P0, PT;
    --:-:-:-:6      PSETP.AND.AND P4, PT, !P1, !P0, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      IADD pp0.CC, xp0, -0x10;
    --:-:-:-:6      IADD.X pp1, xp1, -0x1;
    --:-:2:-:1  @P4 LDG.E.128 pv0, [pp];
  ]], { NL2E = NL2E_HEX, M_T = mul("t", "c", "ldx", "RZ", "t2") }))
  -- the first float4 of a row takes x(t0 - 3 .. t0 - 1) from the ring: slot (t0 - k) & 3 of channel c, k = 3, 2, 1 into pv1..pv3
  for _, pair in ipairs({ { 1, 3 }, { 2, 2 }, { 3, 1 } }) do
    local reg, k = pair[1], pair[2]
    emit(string.format(heredoc([[
      --:-:-:-:6      MOV t, param_t0;
      --:-:-:-:6      IADD t, t, -0x%x;
      --:-:-:-:6      LOP32I.AND t, t, 0x3;
      --:-:-:-:6      MOV t2, param_n;
#{M_TM}--:-:-:-:6      SHL t, tm, 0x2;
      --:-:-:-:6      IADD sp0.CC, t, param_ST0;
      --:-:-:-:6      IADD.X sp1, RZ, param_ST1;
      --:-:2:-:2  @P3 LDG.E pv%d, [sp];
    ]], { M_TM = mul("tm", "t", "t2", "c", "tt") }), k, reg))
  end
  -- y_j = w0 x(p - 3) + w1 x(p - 2) + w2 x(p - 1) + w3 x(p): the 7 inputs pv1 pv2 pv3 x0 x1 x2 x3
  emit("02:-:-:-:6      NOP;")
  local win = { "pv1", "pv2", "pv3", "x0", "x1", "x2", "x3" }
  for j = 0, 3 do
    emit(string.format("--:-:-:-:6      FMUL y%d, w0, %s;", j, win[j + 1]))
    emit(string.format("--:-:-:-:6      FFMA y%d, w1, %s, y%d;", j, win[j + 2], j))
    emit(string.format("--:-:-:-:6      FFMA y%d, w2, %s, y%d;", j, win[j + 3], j))
    emit(string.format("--:-:-:-:6      FFMA y%d, w3, %s, y%d;", j, win[j + 4], j))
  end
  emit(sigm({ "sg0", "sg1", "sg2", "sg3" }, { "y0", "y1", "y2", "y3" }, "nl2e"))
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL y%d, y%d, sg%d;", j == 3 and 6 or 1, j, j, j))
  end
  -- positions past npos are zero
  for j = 0, 3 do
    emit(string.format(heredoc([[
      --:-:-:-:6      ISCADD pos, p4, 0x%x, 0x2;
      --:-:-:-:6      ISETP.GE.U32.AND P5, PT, pos, param_npos, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P5 MOV y%d, RZ;
    ]]), j, j))
  end
  emit("--:1:-:-:1 @!P0 STG.E.128 [up], y0;")
  -- the first thread of a row: the ring gets the chunk's last four inputs x(t0 + npos - 4 + j) (those at or after t0)
  -- (predicated, not branched: P5 = P3 and the input exists; a branch here would split the warp)
  emit("01:-:-:-:6      NOP;")
  for j = 0, 3 do
    emit(string.format(heredoc([[
      --:-:-:-:6      MOV t, param_npos;
      --:-:-:-:6      IADD pos, t, -0x4;
      --:-:-:-:6      IADD pos, pos, 0x%x;
      --:-:-:-:6      ISETP.GE.AND P5, PT, pos, RZ, P3;
#{M_T}      --:-:-:-:6      SHL t, t, 0x2;
      --:-:-:-:6      IADD xp0.CC, t, param_X0;
      --:-:-:-:6      IADD.X xp1, RZ, param_X1;
      --:-:2:-:2  @P5 LDG.E sg%d, [xp];
    ]], { M_T = mul("t", "c", "ldx", "pos", "t2") }), j, j))
    emit(string.format(heredoc([[
      --:-:-:-:6      MOV t, param_t0;
      --:-:-:-:6      IADD t, t, pos;
      --:-:-:-:6      LOP32I.AND t, t, 0x3;
      --:-:-:-:6      MOV t2, param_n;
#{M_TM}--:-:-:-:6      SHL t, tm, 0x2;
      --:-:-:-:6      IADD sp0.CC, t, param_ST0;
      --:-:-:-:6      IADD.X sp1, RZ, param_ST1;
      02:1:-:-:1  @P5 STG.E [sp], sg%d;
      01:-:-:-:6      NOP;
    ]], { M_TM = mul("tm", "t", "t2", "c", "tt") }), j))
  end
  emit("DONE:")
  emit("01:-:-:-:f      EXIT;")
end

if OP == "bias" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4     : tid
        5     : bid
        6     : q
        7     : t
        8     : row
        9     : bv
        10    : rs
        12,13 : xp<0-1>
        14,15 : bp<0-1>
    </REGISTER_MAPPING>
  ]]))
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD rs, t, -0x2;
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, q, param_n4, PT;
    --:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n4, PT;
    --:-:-:-:6      SHR.U32 row, q, rs;
    --:-:-:-:6      SHL t, q, 0x4;
    --:-:-:-:6      IADD xp0.CC, t, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
    --:-:-:-:6      SHL t, row, 0x2;
    --:-:-:-:6      IADD bp0.CC, t, param_B0;
    --:-:-:-:6      IADD.X bp1, RZ, param_B1;
    --:-:-:-:6  @P2 EXIT;
    --:-:2:-:1 @!P0 LDG.E.128 x0, [xp];
    --:-:2:-:2 @!P0 LDG.E bv, [bp];
    02:-:-:-:1      FADD x0, x0, bv;
    --:-:-:-:1      FADD x1, x1, bv;
    --:-:-:-:1      FADD x2, x2, bv;
    --:-:-:-:6      FADD x3, x3, bv;
    --:1:-:-:1 @!P0 STG.E.128 [xp], x0;
    01:-:-:-:f      EXIT;
  ]]))
end

if OP == "router" then
  -- Ruby: stop = ->(n) { ENV['SLAB_STOP'] && ENV['SLAB_STOP'].to_i == n ? "--:-:-:-:f      EXIT;\n" : '' }
  local function stop(n)
    local v = os.getenv("SLAB_STOP")
    if v and ruby_to_i(v) == n then return "--:-:-:-:f      EXIT;\n" end
    return ""
  end
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-9   : m<0-9>
        10-19 : ix<0-9>
        20    : v
        21    : iv
        22    : tmax
        23    : timx
        24    : tid
        25    : bid
        26    : lane
        27    : wq
        28    : p
        29    : e
        30    : eend
        31    : t
        32    : step
        33    : sm
        34    : vn
        36,37 : rp<0-1>
        38-47 : ex<0-9>
        48    : den
        49    : nl2e
        50    : w0
        52,53 : op<0-1>
        54    : k
        55    : sh
    </REGISTER_MAPPING>
  ]]))
  -- one insertion of (v, iv) into the sorted list: the larger value moves down the list, the smaller one on (ties keep the list's)
  local function ins()
    local s = {}
    for j = 0, 9 do
      s[#s + 1] = string.format("--:-:-:-:6      FSETP.GT.AND P0, PT, v, m%d, PT;\n", j)
      s[#s + 1] = string.format("--:-:-:-:6      FMNMX tmax, v, m%d, !PT;\n", j)
      s[#s + 1] = string.format("--:-:-:-:6      FMNMX v, v, m%d, PT;\n", j)
      s[#s + 1] = "--:-:-:-:6      NOP;\n"
      s[#s + 1] = string.format("--:-:-:-:6      MOV m%d, tmax;\n", j)
      s[#s + 1] = string.format("--:-:-:-:6      SEL timx, iv, ix%d, P0;\n", j)
      s[#s + 1] = string.format("--:-:-:-:6      SEL iv, ix%d, iv, P0;\n", j)
      s[#s + 1] = string.format("--:-:-:-:6      MOV ix%d, timx;\n", j)
    end
    return table.concat(s)
  end
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 wq, tid, 0x5;
    --:-:-:-:6      ISCADD p, bid, lane, 0x5;
    --:-:-:-:6      SHL e, wq, 0x6;
    --:-:-:-:6      IADD eend, e, 0x40;
    --:-:-:-:6      MOV t, param_ldx;
    --:-:-:-:6      SHL step, t, 0x2;
    #{M_E}
    --:-:-:-:6      ISCADD t, p, t, 0x2;
    --:-:-:-:6      IADD rp0.CC, t, param_R0;
    --:-:-:-:6      IADD.X rp1, RZ, param_R1;
  ]], { M_E = mul("t", "e", "step", "RZ", "sm") }))
  for j = 0, 9 do
    emit(string.format("--:-:-:-:1      MOV32I m%d, 0xff800000;\n--:-:-:-:1      MOV ix%d, RZ;", j, j))
  end
  emit(stop(1))
  -- experts e .. eend - 1 of the column: each logit loaded one ahead
  emit(heredoc([[
    --:-:2:-:2      LDG.E vn, [rp];
    TOP:
    02:-:-:-:6      MOV v, vn;
    --:-:-:-:6      MOV iv, e;
    --:-:-:-:6      IADD e, e, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, e, eend, PT;
    --:-:-:-:6      IADD rp0.CC, rp0, step;
    --:-:-:-:6      IADD.X rp1, rp1, RZ;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:2:-:1  @P1 LDG.E vn, [rp];
  ]]))
  emit(ins())
  emit(heredoc([[
    --:-:-:-:6  @P1 BRA TOP;
  ]]))
  emit(stop(2))
  -- the lists to shared memory: warp q's entry j of lane l at 4 (320 q + 32 j + l) (values) and + 0x2800 (indices)
  for j = 0, 9 do
    emit(string.format(heredoc([[
      01:-:-:-:6      XMAD sm, wq, 0x140, lane;
      --:-:-:-:6      IADD sm, sm, 0x%x;
      --:-:-:-:6      SHL sm, sm, 0x2;
      --:1:-:-:1      STS [sm], m%d;
      --:1:-:-:1      STS [sm+0x2800], ix%d;
    ]]), 32 * j, j, j))
  end
  emit(stop(3))
  emit(heredoc([[
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      ISETP.NE.U32.AND P2, PT, wq, RZ, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P2 BRA DONE;
    --:-:-:-:6      MOV32I k, 0xa;
    MERGE:
    --:-:-:-:6      ISCADD sm, k, lane, 0x5;
    --:-:-:-:6      SHL sm, sm, 0x2;
    --:-:2:-:1      LDS v, [sm];
    --:-:2:-:2      LDS iv, [sm+0x2800];
    --:-:-:-:6      IADD k, k, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, k, 0x50, PT;
    02:-:-:-:6      NOP;
  ]]))
  emit(ins())
  emit("--:-:-:-:6  @P1 BRA MERGE;\n" .. stop(4))
  emit(heredoc([[
    --:-:-:-:6      MOV32I nl2e, 0x3fb8aa3b;
  ]]))
  -- ex_j = 2^((m_j - m_0) log2 e), den = the sum in rank order, w_j = ex_j / den
  for j = 0, 9 do
    emit(string.format("--:-:-:-:6      FADD ex%d, m%d, -m0;\n--:-:-:-:6      FMUL ex%d, ex%d, nl2e;\n--:-:-:-:6      RRO.EX2 ex%d, ex%d;\n--:-:3:-:2      MUFU.EX2 ex%d, ex%d;", j, j, j, j, j, j, j, j))
  end
  emit("04:-:-:-:6      MOV den, ex0;")
  for j = 1, 9 do
    emit(string.format("--:-:-:-:6      FADD den, den, ex%d;", j))
  end
  emit(heredoc([[
    --:-:3:-:2      MUFU.RCP den, den;
    --:-:-:-:6      XMAD t, p, 0x28, RZ;
    --:-:-:-:6      IADD op0.CC, t, param_IDX0;
    --:-:-:-:6      IADD.X op1, RZ, param_IDX1;
  ]]))
  for j = 0, 9 do
    emit(string.format("--:1:-:-:1      STG.E [op+0x%x], ix%d;", 4 * j, j))
  end
  emit(heredoc([[
    --:-:-:-:6      XMAD t, p, 0x28, RZ;
    01:-:-:-:6      IADD op0.CC, t, param_W0;
    --:-:-:-:6      IADD.X op1, RZ, param_W1;
  ]]))
  for j = 0, 9 do
    emit(string.format("%s:-:-:-:6      FMUL ex%d, ex%d, den;", j == 0 and "04" or "--", j, j))
  end
  for j = 0, 9 do
    emit(string.format("--:1:-:-:1      STG.E [op+0x%x], ex%d;", 4 * j, j))
  end
  emit(stop(5))
  -- s = sigmoid(R[512][p])
  emit(heredoc([[
    --:-:-:-:6      MOV t, param_ldx;
    --:-:-:-:6      SHL t, t, 0xb;
    --:-:-:-:6      ISCADD t, p, t, 0x2;
    01:-:-:-:6      IADD op0.CC, t, param_R0;
    --:-:-:-:6      IADD.X op1, RZ, param_R1;
    --:-:2:-:2      LDG.E sh, [op];
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    02:-:-:-:6      FMUL sh, sh, nl2e;
    --:-:-:-:6      RRO.EX2 sh, sh;
    --:-:3:-:2      MUFU.EX2 sh, sh;
    04:-:-:-:6      FADD sh, sh, 1;
    --:-:3:-:2      MUFU.RCP sh, sh;
    --:-:-:-:6      SHL t, p, 0x2;
    --:-:-:-:6      IADD op0.CC, t, param_SW0;
    --:-:-:-:6      IADD.X op1, RZ, param_SW1;
    04:1:-:-:1      STG.E [op], sh;
    DONE:
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:f      EXIT;
  ]], { NL2E = NL2E_HEX }))
end

if OP == "tr" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4     : tid
        5     : bid
        6     : tx
        7     : ty
        8     : nc
        9     : br
        10    : bc
        11    : t
        12    : t2
        13    : ldx
        14    : sm
        15    : lgc
        16,17 : xp<0-1>
        18,19 : yp<0-1>
        20    : rstep
        21    : tm
        22,23 : zp<0-1>
        24,25 : wp<0-1>
    </REGISTER_MAPPING>
  ]]))
  -- tile bid: rows 32 br .., columns 32 bc .. with nc = ldx / 32 tiles per row band; thread (tx = tid & 31, ty = tid >> 5) moves
  -- rows ty, ty + 8, ty + 16, ty + 24 of the tile (column tx) into shared [32][33], then out transposed
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD lgc, t, -0x5;
    --:-:-:-:6      MOV32I t2, 0x1;
    --:-:-:-:6      SHL ldx, t2, t;
    01:-:-:-:6      LOP32I.AND tx, tid, 0x1f;
    --:-:-:-:6      SHR.U32 ty, tid, 0x5;
    --:-:-:-:6      SHR.U32 br, bid, lgc;
    --:-:-:-:6      SHL t, t2, lgc;
    --:-:-:-:6      IADD t, t, -0x1;
    --:-:-:-:6      LOP.AND bc, bid, t;
    --:-:-:-:6      ISCADD t, br, ty, 0x5;
    #{M_TM}
    --:-:-:-:6      ISCADD t, bc, tx, 0x5;
    --:-:-:-:6      IADD t, tm, t;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD xp0.CC, t, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
  ]], { M_TM = mul("tm", "t", "ldx", "RZ", "t2") }))
  -- four loads through four offsets (8 rows apart): pointers xp + 8 i ldx 4 computed up front
  emit(heredoc([[
    --:-:-:-:6      SHL rstep, ldx, 0x5;
    --:-:-:-:6      IADD yp0.CC, xp0, rstep;
    --:-:-:-:6      IADD.X yp1, xp1, RZ;
    --:-:-:-:6      IADD zp0.CC, yp0, rstep;
    --:-:-:-:6      IADD.X zp1, yp1, RZ;
    --:-:-:-:6      IADD wp0.CC, zp0, rstep;
    --:-:-:-:6      IADD.X wp1, zp1, RZ;
    --:-:2:-:1      LDG.E x0, [xp];
    --:-:2:-:1      LDG.E x1, [yp];
    --:-:2:-:1      LDG.E x2, [zp];
    --:-:2:-:1      LDG.E x3, [wp];
    --:-:-:-:6      XMAD sm, ty, 0x21, tx;
    --:-:-:-:6      SHL sm, sm, 0x2;
    02:1:-:-:1      STS [sm], x0;
    --:1:-:-:1      STS [sm+0x420], x1;
    --:1:-:-:1      STS [sm+0x840], x2;
    --:1:-:-:1      STS [sm+0xc60], x3;
    01:-:-:-:5      BAR.SYNC 0x0;
  ]]))
  -- out: Y[32 bc + ty + 8 i][32 br + tx] = tile[tx][ty + 8 i]; Y's row is `rows` floats
  emit(heredoc([[
    --:-:-:-:6      XMAD sm, tx, 0x21, ty;
    --:-:-:-:6      SHL sm, sm, 0x2;
    --:-:2:-:1      LDS x0, [sm];
    --:-:2:-:1      LDS x1, [sm+0x20];
    --:-:2:-:1      LDS x2, [sm+0x40];
    --:-:2:-:1      LDS x3, [sm+0x60];
    --:-:-:-:6      MOV t2, param_rows;
    --:-:-:-:6      ISCADD t, bc, ty, 0x5;
    #{M_TID}
    --:-:-:-:6      ISCADD t, br, tx, 0x5;
    --:-:-:-:6      IADD t, tm, t;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD xp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X xp1, RZ, param_Y1;
    --:-:-:-:6      SHL rstep, t2, 0x5;
    --:-:-:-:6      IADD yp0.CC, xp0, rstep;
    --:-:-:-:6      IADD.X yp1, xp1, RZ;
    02:1:-:-:1      STG.E [xp], x0;
    --:1:-:-:1      STG.E [yp], x1;
    01:-:-:-:6      IADD xp0.CC, yp0, rstep;
    --:-:-:-:6      IADD.X xp1, yp1, RZ;
    --:-:-:-:6      IADD yp0.CC, xp0, rstep;
    --:-:-:-:6      IADD.X yp1, xp1, RZ;
    --:1:-:-:1      STG.E [xp], x2;
    --:1:-:-:1      STG.E [yp], x3;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:f      EXIT;
  ]], { M_TID = mul("tm", "t", "t2", "RZ", "tid") }))
end

if OP == "group" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0     : tid
        1     : q
        2     : e
        3     : t
        4     : old
        5     : one
        6     : c
        7     : m
        8     : acc
        9     : ni
        10    : sm
        11    : pos
        12    : wv
        13    : t2
        14,15 : ip<0-1>
        16,17 : wp<0-1>
        18,19 : op<0-1>
        20,21 : vp<0-1>
        22    : mc
        23    : lane
    </REGISTER_MAPPING>
  ]]))
  -- a pair loop: q = tid + 256 k; the loop ends for the whole warp when its first q reaches npairs (P0: this thread's q is a pair)
  local function loop_head(lbl, exit_lbl)
    return lbl .. ":\n"
      .. "--:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;\n"
      .. "--:-:-:-:6      ISETP.GE.U32.AND P4, PT, t, param_npairs, PT;\n"
      .. "--:-:-:-:6      ISETP.LT.U32.AND P0, PT, q, param_npairs, PT;\n"
      .. "--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n"
      .. "--:-:-:-:6  @P4 BRA " .. exit_lbl .. ";\n"
  end
  -- shared: cnt [512] at 0, fill [512] at 0x800, emap [512] at 0x1000
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    01:-:-:-:6      SHL sm, tid, 0x2;
    --:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:1:-:-:1      STS [sm], RZ;
    --:1:-:-:1      STS [sm+0x400], RZ;
    --:-:-:-:6      IADD ip0.CC, sm, param_EM0;
    --:-:-:-:6      IADD.X ip1, RZ, param_EM1;
    --:-:2:-:1      LDG.E m, [ip];
    --:-:2:-:2      LDG.E mc, [ip+0x400];
    02:1:-:-:1      STS [sm+0x1000], m;
    --:1:-:-:1      STS [sm+0x1400], mc;
    --:-:-:-:6      MOV32I one, 0x1;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      MOV q, tid;
  ]]))
  emit(loop_head("COUNT", "COUNTED"))
  emit(heredoc([[
    --:-:-:-:6      SHL t, q, 0x2;
    --:-:-:-:6      IADD ip0.CC, t, param_IDX0;
    --:-:-:-:6      IADD.X ip1, RZ, param_IDX1;
    --:-:2:-:2  @P0 LDG.E e, [ip];
    02:-:-:-:6      SHL t, e, 0x2;
    --:-:3:-:2  @P0 ATOMS.ADD old, [t], one;
    04:-:-:-:6      IADD q, q, 0x100;
    --:-:-:-:6      BRA COUNT;
    COUNTED:
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      ISETP.GE.U32.AND P1, PT, tid, 0x20, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P5, PT, lane, RZ, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P1 BRA SCANNED;
  ]]))
  -- warp 0 (lane 0 stores): off(e) = the counts before e (into fill), the items of experts with units here and pairs
  emit(heredoc([[
    --:-:-:-:6      MOV acc, RZ;
    --:-:-:-:6      MOV ni, RZ;
    --:-:-:-:6      MOV e, RZ;
    --:-:-:-:6      MOV op0, param_IT0;
    --:-:-:-:6      MOV op1, param_IT1;
    SCAN:
    01:-:-:-:6      SHL t, e, 0x2;
    --:-:2:-:1      LDS c, [t];
    --:-:2:-:2      LDS m, [t+0x1000];
    02:1:-:-:1  @P5 STS [t+0x800], acc;
    --:-:-:-:6      LOP32I.AND t2, m, 0xff;
    --:-:-:-:6      IMNMX.U32 t2, t2, c, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P2, PT, t2, RZ, PT;
    --:-:-:-:6      SHL t2, ni, 0x4;
    --:-:-:-:6      IADD vp0.CC, t2, op0;
    --:-:-:-:6      IADD.X vp1, RZ, op1;
    --:-:-:-:6      PSETP.AND.AND P6, PT, P2, P5, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:1:-:-:1  @P6 STG.E [vp], m;
    --:1:-:-:1  @P6 STG.E [vp+0x4], acc;
    --:1:-:-:1  @P6 STG.E [vp+0x8], c;
    --:1:-:-:1  @P6 STG.E [vp+0xc], e;
    01:-:-:-:6  @P2 IADD ni, ni, 0x1;
    --:-:-:-:6      IADD acc, acc, c;
    --:-:-:-:6      IADD e, e, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P3, PT, e, 0x200, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P3 BRA SCAN;
    --:-:-:-:6      MOV vp0, param_NI0;
    --:-:-:-:6      MOV vp1, param_NI1;
    --:1:-:-:1  @P5 STG.E [vp], ni;
    --:1:-:-:1  @P5 STG.E [vp+0x4], RZ;
    --:1:-:-:1  @P5 STG.E [vp+0x8], RZ;
    SCANNED:
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      MOV q, tid;
  ]]))
  emit(loop_head("FILL", "FILLED"))
  -- every pair: slot = fill(e)++, LP[slot] = q / 10, LW[slot] = W[q]
  emit(heredoc([[
    --:-:-:-:6      SHL t, q, 0x2;
    --:-:-:-:6      IADD ip0.CC, t, param_IDX0;
    --:-:-:-:6      IADD.X ip1, RZ, param_IDX1;
    --:-:-:-:6      IADD wp0.CC, t, param_W0;
    --:-:-:-:6      IADD.X wp1, RZ, param_W1;
    --:-:2:-:1  @P0 LDG.E e, [ip];
    --:-:2:-:2  @P0 LDG.E wv, [wp];
    02:-:-:-:6      SHL t, e, 0x2;
    --:-:3:-:2  @P0 ATOMS.ADD old, [t+0x800], one;
    --:-:-:-:6      MOV32I t2, 0xcccd;
    --:-:-:-:6      XMAD pos, q, t2, RZ;
    --:-:-:-:6      SHR.U32 pos, pos, 0x13;
    04:-:-:-:6      SHL t, old, 0x2;
    --:-:-:-:6      IADD op0.CC, t, param_LP0;
    --:-:-:-:6      IADD.X op1, RZ, param_LP1;
    --:-:-:-:6      IADD vp0.CC, t, param_LW0;
    --:-:-:-:6      IADD.X vp1, RZ, param_LW1;
    --:1:-:-:1  @P0 STG.E [op], pos;
    --:1:-:-:1  @P0 STG.E [vp], wv;
    01:-:-:-:6      IADD q, q, 0x100;
    --:-:-:-:6      BRA FILL;
    FILLED:
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:f      EXIT;
  ]]))
end

-- the elementwise head: q = 256 bid + tid over n4 float4s, P0 = q is one, the warp leaves when its first q is past n4
local EHEAD = "--:-:1:-:1      S2R tid, SR_TID.X;\n--:-:1:-:1      S2R bid, SR_CTAID.X;\n"
  .. "01:-:-:-:6      ISCADD q, bid, tid, 0x8;\n"
  .. "--:-:-:-:6      ISETP.GE.U32.AND P0, PT, q, param_n4, PT;\n"
  .. "--:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;\n"
  .. "--:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n4, PT;\n"
  .. "--:-:-:-:6      SHL t, q, 0x4;\n"

if OP == "swiglu" or OP == "saxpy" or OP == "zero" or OP == "sum" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : g<0-3>
        4-7   : u<0-3>
        8-11  : y<0-3>
        12    : tid
        13    : bid
        14    : q
        15    : t
        16    : nl2e
        17    : d
        18,19 : gp<0-1>
        20,21 : up<0-1>
        22,23 : yp<0-1>
        24    : p4
        25    : rs
        26    : lm
        27    : t2
        28    : tm
        29    : rstride
        32-39 : pi<0-7>
        40-43 : fl, fh, hf, fr
    </REGISTER_MAPPING>
  ]]))
  emit(EHEAD)
end
if OP == "swiglu" then
  emit(heredoc([[
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    --:-:-:-:6      IADD gp0.CC, t, param_G0;
    --:-:-:-:6      IADD.X gp1, RZ, param_G1;
    --:-:-:-:6      IADD up0.CC, t, param_U0;
    --:-:-:-:6      IADD.X up1, RZ, param_U1;
    --:-:-:-:6      IADD yp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6  @P2 EXIT;
    --:-:2:-:1 @!P0 LDG.E.128 g0, [gp];
    --:-:2:-:1 @!P0 LDG.E.128 u0, [up];
    02:-:-:-:6      NOP;
  ]], { NL2E = NL2E_HEX }))
  emit(sigm({ "y0", "y1", "y2", "y3" }, { "g0", "g1", "g2", "g3" }, "nl2e"))
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL y%d, y%d, g%d;", j == 3 and 6 or 1, j, j, j))
  end
  for j = 0, 3 do
    emit(string.format("--:-:-:-:%d      FMUL y%d, y%d, u%d;", j == 3 and 6 or 1, j, j, j))
  end
  emit("--:1:-:-:1 @!P0 STG.E.128 [yp], y0;\n01:-:-:-:f      EXIT;")
elseif OP == "zero" then
  emit(heredoc([[
    --:-:-:-:6      IADD yp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6      MOV y0, RZ;
    --:-:-:-:6      MOV y1, RZ;
    --:-:-:-:6      MOV y2, RZ;
    --:-:-:-:6      MOV y3, RZ;
    --:-:-:-:6  @P2 EXIT;
    --:1:-:-:1 @!P0 STG.E.128 [yp], y0;
    01:-:-:-:f      EXIT;
  ]]))
elseif OP == "saxpy" then
  -- p4 = q mod (ldx / 4): the float4 of positions 4 p4 .. 4 p4 + 3 of SW. P is the MoE partial in int32 fixed point (value
  -- 2^20, ebatch dn's atomics): P += rint(2^20 SW X)
  emit(heredoc([[
    --:-:-:-:6      IADD gp0.CC, t, param_P0;
    --:-:-:-:6      IADD.X gp1, RZ, param_P1;
    --:-:-:-:6      IADD up0.CC, t, param_X0;
    --:-:-:-:6      IADD.X up1, RZ, param_X1;
    --:-:-:-:6      MOV t2, param_lgx;
    --:-:-:-:6      IADD rs, t2, -0x2;
    --:-:-:-:6      MOV32I lm, 0x1;
    --:-:-:-:6      SHL lm, lm, rs;
    --:-:-:-:6      IADD lm, lm, -0x1;
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      SHL t2, p4, 0x4;
    --:-:-:-:6      IADD yp0.CC, t2, param_SW0;
    --:-:-:-:6      IADD.X yp1, RZ, param_SW1;
    --:-:-:-:6  @P2 EXIT;
    --:-:2:-:1 @!P0 LDG.E.128 pi0, [gp];
    --:-:2:-:1 @!P0 LDG.E.128 u0, [up];
    --:-:2:-:2 @!P0 LDG.E.128 y0, [yp];
    02:-:-:-:1      FMUL g0, y0, u0;
    --:-:-:-:1      FMUL g1, y1, u1;
    --:-:-:-:1      FMUL g2, y2, u2;
    --:-:-:-:6      FMUL g3, y3, u3;
  ]]))
  for j = 0, 3 do
    emit(string.format(dedent([[
      --:-:-:-:6      FMUL32I fr, g%d, 0x49800000;
      --:-:3:-:2      F2I.S32.F32 fl, fr;
      04:-:-:-:6      IADD pi%d, pi%d, fl;
    ]]), j, j, j))
  end
  emit(heredoc([[
    --:1:-:-:1 @!P0 STG.E.128 [gp], pi0;
    01:-:-:-:f      EXIT;
  ]]))
elseif OP == "sum" then
  -- row r = q >> (lgx - 2) of the owner's rows; part d at X + (d maxr + r) ldx 4 + 16 p4 = X + 16 q + d maxr ldx 4
  emit(heredoc([[
    --:-:-:-:6      IADD gp0.CC, t, param_X0;
    --:-:-:-:6      IADD.X gp1, RZ, param_X1;
    --:-:-:-:6      IADD yp0.CC, t, param_Y0;
    --:-:-:-:6      IADD.X yp1, RZ, param_Y1;
    --:-:-:-:6      MOV t2, param_lgx;
    --:-:-:-:6      MOV32I tm, 0x1;
    --:-:-:-:6      SHL tm, tm, t2;
    --:-:-:-:6      MOV t2, param_maxr;
    #{M_RSTRIDE}
    --:-:-:-:6      SHL rstride, rstride, 0x2;
    --:-:-:-:6  @P2 EXIT;
    --:-:-:-:6      MOV y0, RZ;
    --:-:-:-:6      MOV y1, RZ;
    --:-:-:-:6      MOV y2, RZ;
    --:-:-:-:6      MOV y3, RZ;
    --:-:-:-:6      MOV d, RZ;
    SUM:
    --:-:2:-:2 @!P0 LDG.E.128 g0, [gp];
    02:-:-:-:1      FADD y0, y0, g0;
    --:-:-:-:1      FADD y1, y1, g1;
    --:-:-:-:1      FADD y2, y2, g2;
    --:-:-:-:6      FADD y3, y3, g3;
    --:-:-:-:6      IADD gp0.CC, gp0, rstride;
    --:-:-:-:6      IADD.X gp1, gp1, RZ;
    --:-:-:-:6      IADD d, d, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, d, param_nd, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P1 BRA SUM;
    --:1:-:-:1 @!P0 STG.E.128 [yp], y0;
    01:-:-:-:f      EXIT;
  ]], { M_RSTRIDE = mul("rstride", "t2", "tm", "RZ", "rs") }))
end

if OP == "rs" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4     : tid
        5     : bid
        6     : q
        7     : t
        8     : t2
        9     : row
        10    : o
        11    : bo
        12    : rs
        13    : lm
        14    : p4
        15    : tm
        16    : ldx
        18,19 : xp<0-1>
        20,21 : tp<0-1>
        22,23 : ib<0-1>
        24,25 : dp<0-1>
        28-35 : pi<0-7>
    </REGISTER_MAPPING>
  ]]))
  -- thread q over P's float4s (2560 ldx / 4 of them: no tail), row = q >> (lgx - 2), owner o = #{B_i <= row}, bo = B_o
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV t, param_lgx;
    --:-:-:-:6      IADD rs, t, -0x2;
    --:-:-:-:6      MOV32I t2, 0x1;
    --:-:-:-:6      SHL ldx, t2, t;
    --:-:-:-:6      SHL lm, t2, rs;
    --:-:-:-:6      IADD lm, lm, -0x1;
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      SHR.U32 row, q, rs;
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      MOV o, RZ;
    --:-:-:-:6      MOV bo, RZ;
  ]]))
  for _, bnd in ipairs({ "B1", "B2", "B3", "B4", "B5" }) do
    emit(heredoc([[
      --:-:-:-:6      ISETP.GE.U32.AND P1, PT, row, param_#{BND}, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6  @P1 IADD o, o, 0x1;
      --:-:-:-:6  @P1 MOV bo, param_#{BND};
    ]], { BND = bnd }))
  end
  emit(heredoc([[
    --:-:-:-:6      SHL t, q, 0x4;
    --:-:-:-:6      IADD xp0.CC, t, param_P0;
    --:-:-:-:6      IADD.X xp1, RZ, param_P1;
    --:-:2:-:1      LDG.E.128 pi0, [xp];
    --:-:-:-:6      SHL t, o, 0x3;
    --:-:-:-:6      IADD tp0.CC, t, param_TAB0;
    --:-:-:-:6      IADD.X tp1, RZ, param_TAB1;
    --:-:3:-:2      LDG.E.64 ib0, [tp];
    --:-:-:-:6      MOV t, param_me;
    --:-:-:-:6      MOV t2, param_maxr;
    #{M_TM}
    --:-:-:-:6      IADD tm, tm, -bo;
    #{M_T}
    --:-:-:-:6      LOP.AND p4, q, lm;
    --:-:-:-:6      ISCADD t, p4, t, 0x2;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD t, t, param_slot;
    04:-:-:-:6      IADD dp0.CC, t, ib0;
    --:-:-:-:6      IADD.X dp1, RZ, ib1;
    02:-:2:-:1      I2F.F32.S32 x0, pi0;
    --:-:2:-:1      I2F.F32.S32 x1, pi1;
    --:-:2:-:1      I2F.F32.S32 x2, pi2;
    --:-:2:-:2      I2F.F32.S32 x3, pi3;
    02:-:-:-:1      FMUL32I x0, x0, 0x35800000;
    --:-:-:-:1      FMUL32I x1, x1, 0x35800000;
    --:-:-:-:1      FMUL32I x2, x2, 0x35800000;
    --:-:-:-:6      FMUL32I x3, x3, 0x35800000;
    --:1:-:-:1      STG.E.WT.128 [dp], x0;
    01:-:-:-:6      MEMBAR.SYS;
    --:-:-:-:f      EXIT;
  ]], {
    M_TM = mul("tm", "t", "t2", "row", "p4"),
    M_T = mul("t", "tm", "ldx", "RZ", "t2"),
  }))
end

if OP == "pgate" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-7   : x<0-7>
        8-15  : y<0-7>
        16-19 : acc<0-3>
        20    : tid
        21    : bid
        22    : lane
        23    : kq
        24    : l
        25    : g
        26    : p
        27    : t
        28    : t2
        29    : k
        30    : ss
        31    : sm
        32    : stride
        33    : gate
        34    : mag
        35    : rt
        36    : tm
        37    : nl2e
        38,39 : kp<0-1>
        40,41 : qp<0-1>
        42,43 : vp<0-1>
        44,45 : gp<0-1>
        46-61 : a<00-15>
        62-77 : b<00-15>
        78    : st8
    </REGISTER_MAPPING>
  ]]))
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    01:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
    --:-:-:-:6      SHR.U32 kq, tid, 0x5;
    --:-:-:-:6      SHR.U32 l, bid, param_lgg;
    --:-:-:-:6      MOV32I t, 0x1;
    --:-:-:-:6      SHL t, t, param_lgg;
    --:-:-:-:6      IADD t, t, -0x1;
    --:-:-:-:6      LOP.AND g, bid, t;
    --:-:-:-:6      ISCADD p, g, lane, 0x5;
    --:-:-:-:6      MOV32I t, 0x1;
    --:-:-:-:6      SHL stride, t, param_lgx;
    --:-:-:-:6      SHL stride, stride, 0x2;
    --:-:-:-:6      SHL st8, stride, 0x3;
    --:-:-:-:6      XMAD t, l, 0xa00, RZ;
    --:-:-:-:6      XMAD t2, kq, 0x140, RZ;
    --:-:-:-:6      IADD t, t, t2;
    --:-:-:-:6      SHL t, t, param_lgx;
    --:-:-:-:6      IADD t, t, p;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD kp0.CC, t, param_KN0;
    --:-:-:-:6      IADD.X kp1, RZ, param_KN1;
    --:-:-:-:6      IADD qp0.CC, t, param_QN0;
    --:-:-:-:6      IADD.X qp1, RZ, param_QN1;
    --:-:-:-:6      IADD gp0.CC, t, param_G0;
    --:-:-:-:6      IADD.X gp1, RZ, param_G1;
    --:-:-:-:6      SHL t, t2, param_lgx;
    --:-:-:-:6      IADD t, t, p;
    --:-:-:-:6      SHL t, t, 0x2;
    --:-:-:-:6      IADD vp0.CC, t, param_V0;
    --:-:-:-:6      IADD.X vp1, RZ, param_V1;
    --:-:-:-:6      MOV acc0, RZ;
    --:-:-:-:6      MOV acc1, RZ;
    --:-:-:-:6      MOV acc2, RZ;
    --:-:-:-:6      MOV acc3, RZ;
    --:-:-:-:6      MOV k, RZ;
    DOT:
  ]], { NL2E = NL2E_HEX }))
  -- 8 rows a round of KN and QN; the pointers move by 8 rows after the round's loads returned
  emit(ptr8("a", "kp"))
  emit(ptr8("b", "qp"))
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E x%d, [a%02d];", i, 2 * i))
  end
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E y%d, [b%02d];", i, 2 * i))
  end
  emit(heredoc([[
    02:-:-:-:1      FFMA acc0, x0, y0, acc0;
    --:-:-:-:1      FFMA acc1, x1, y1, acc1;
    --:-:-:-:1      FFMA acc2, x2, y2, acc2;
    --:-:-:-:1      FFMA acc3, x3, y3, acc3;
    --:-:-:-:1      IADD kp0.CC, kp0, st8;
    --:-:-:-:1      IADD.X kp1, kp1, RZ;
    --:-:-:-:1      FFMA acc0, x4, y4, acc0;
    --:-:-:-:1      FFMA acc1, x5, y5, acc1;
    --:-:-:-:1      FFMA acc2, x6, y6, acc2;
    --:-:-:-:6      FFMA acc3, x7, y7, acc3;
    --:-:-:-:6      IADD qp0.CC, qp0, st8;
    --:-:-:-:6      IADD.X qp1, qp1, RZ;
    --:-:-:-:6      IADD k, k, 0x8;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, k, 0x140, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA DOT;
    --:-:-:-:6      FADD acc0, acc0, acc1;
    --:-:-:-:6      FADD acc2, acc2, acc3;
    --:-:-:-:6      FADD acc0, acc0, acc2;
    --:-:-:-:6      SHL sm, tid, 0x2;
    --:1:-:-:1      STS [sm], acc0;
    01:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      SHL sm, lane, 0x2;
  ]]))
  for i = 0, 7 do
    emit(string.format("--:-:2:-:%d      LDS x%d, [sm+0x%x];", i == 7 and 2 or 1, i, 128 * i))
  end
  -- s' = s / sqrt(2560); gate = sigmoid(sign(s') sqrt(max(|s'|, 1e-6)))
  emit(heredoc([[
    02:-:-:-:6      FADD ss, x0, x1;
    --:-:-:-:6      FADD ss, ss, x2;
    --:-:-:-:6      FADD ss, ss, x3;
    --:-:-:-:6      FADD ss, ss, x4;
    --:-:-:-:6      FADD ss, ss, x5;
    --:-:-:-:6      FADD ss, ss, x6;
    --:-:-:-:6      FADD ss, ss, x7;
    --:-:-:-:6      MOV32I tm, 0x3ca1e89b;
    --:-:-:-:6      FMUL ss, ss, tm;
    --:-:-:-:6      MOV32I tm, 0x358637bd;
    --:-:-:-:6      FMNMX mag, |ss|, tm, !PT;
    --:-:3:-:1      MUFU.RSQ rt, mag;
    04:-:-:-:6      FMUL rt, rt, mag;
    --:-:-:-:6      FSETP.LT.AND P1, PT, ss, RZ, PT;
    --:-:-:-:6      FSETP.EQ.AND P2, PT, ss, RZ, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P1 FADD rt, RZ, -rt;
    --:-:-:-:6  @P2 MOV rt, RZ;
    --:-:-:-:6      FMUL gate, rt, nl2e;
    --:-:-:-:6      RRO.EX2 gate, gate;
    --:-:3:-:1      MUFU.EX2 gate, gate;
    04:-:-:-:6      FADD gate, gate, 1;
    --:-:3:-:1      MUFU.RCP gate, gate;
    04:-:-:-:6      MOV k, RZ;
    VAL:
  ]]))
  emit(ptr8("a", "vp"))
  for i = 0, 7 do
    emit(string.format("--:-:2:-:1      LDG.E x%d, [a%02d];", i, 2 * i))
  end
  emit(ptr8("b", "gp"))
  for i = 0, 7 do
    emit(string.format("%s:-:-:-:%d      FMUL x%d, x%d, gate;", i == 0 and "02" or "--", i == 7 and 6 or 1, i, i))
  end
  if os.getenv("PGATE_DBG") == "1" then
    for i = 0, 7 do
      emit(string.format("--:-:-:-:1      MOV x%d, ss;", i))
    end
  end
  if os.getenv("PGATE_DBG") == "2" then
    emit("--:-:-:-:1      MOV x0, ss;\n--:-:-:-:1      MOV x1, mag;\n--:-:-:-:1      MOV x2, rt;\n--:-:-:-:1      MOV x3, gate;\n--:-:-:-:6      MOV x4, nl2e;")
  end
  for i = 0, 7 do
    emit(string.format("--:1:-:-:1      STG.E [b%02d], x%d;", 2 * i, i))
  end
  emit(heredoc([[
    01:-:-:-:6      SHL tm, stride, 0x3;
    --:-:-:-:6      IADD vp0.CC, vp0, tm;
    --:-:-:-:6      IADD.X vp1, vp1, RZ;
    --:-:-:-:6      IADD gp0.CC, gp0, tm;
    --:-:-:-:6      IADD.X gp1, gp1, RZ;
    --:-:-:-:6      IADD k, k, 0x8;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, k, 0x140, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA VAL;
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:f      EXIT;
  ]]))
end

if OP == "pconv" then
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4-7   : w<0-3>
        8     : tid
        9     : bid
        10    : q
        11    : c
        12    : p
        13    : tt
        14    : lm
        15    : y
        16    : sg
        17    : gv
        18    : sv
        19    : nl2e
        20    : ldx
        21    : tm
        22    : i
        23    : pos
        24,25 : h<0-1>
        26,27 : xp<0-1>
        28,29 : wp<0-1>
        30,31 : gp<0-1>
        32,33 : sp<0-1>
        34-39 : rp<0-5>
        40-47 : op<0-7>
        48,49 : cp<0-1>
        50,51 : dp<0-1>
        52    : mt
    </REGISTER_MAPPING>
  ]]))
  -- v mod 10 for v < 10000: v - 10 ((v 6554) >> 16) (16 x 16 XMADs)
  local function mod10(d, v)
    return "--:-:-:-:6      XMAD mt, " .. v .. ", 0x199a, RZ;\n--:-:-:-:6      SHR.U32 mt, mt, 0x10;\n--:-:-:-:6      XMAD mt, mt, 0xa, RZ;\n--:-:-:-:6      IADD " .. d .. ", " .. v .. ", -mt;\n"
  end
  -- thread (c, p) = (q >> lgx, q & (ldx - 1)); P0: no channel / position (every store and load predicated: the warp stays whole)
  emit(heredoc([[
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV32I nl2e, #{NL2E};
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      SHR.U32 c, q, param_lgx;
    --:-:-:-:6      MOV32I tm, 0x1;
    --:-:-:-:6      SHL ldx, tm, param_lgx;
    --:-:-:-:6      IADD lm, ldx, -0x1;
    --:-:-:-:6      LOP.AND p, q, lm;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, p, param_npos, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      ISETP.GE.U32.OR P0, PT, c, 0x2800, P0;
    --:-:-:-:6      SHL tt, q, 0x2;
    --:-:-:-:6      IADD xp0.CC, tt, param_X0;
    --:-:-:-:6      IADD.X xp1, RZ, param_X1;
    --:-:-:-:6      IADD gp0.CC, tt, param_G0;
    --:-:-:-:6      IADD.X gp1, RZ, param_G1;
    --:-:-:-:6      IADD sp0.CC, tt, param_S0;
    --:-:-:-:6      IADD.X sp1, RZ, param_S1;
    --:-:-:-:6      SHL tt, c, 0x3;
    --:-:-:-:6      IADD wp0.CC, tt, param_W0;
    --:-:-:-:6      IADD.X wp1, RZ, param_W1;
    --:-:-:-:6      MOV h0, RZ;
    --:-:-:-:6      MOV h1, RZ;
    --:-:-:-:6      MOV gv, RZ;
    --:-:-:-:6      MOV sv, RZ;
    --:-:2:-:1 @!P0 LDG.E.64 h0, [wp];
    --:-:2:-:1 @!P0 LDG.E gv, [gp];
    --:-:2:-:1 @!P0 LDG.E sv, [sp];
    --:-:2:-:1 @!P0 LDG.E x3, [xp];
  ]], { NL2E = NL2E_HEX }))
  -- x(t - back), back = 9, 6, 3: the chunk's column p - back (p >= back), else the ring slot (t0 mod 10 + p + 10 - back) mod 10 (t >= back)
  local BACKS = { 9, 6, 3 }
  for j = 0, 2 do
    local back = BACKS[j + 1]
    emit(heredoc([[
      --:-:-:-:6      MOV x#{J}, RZ;
      --:-:-:-:6      ISETP.GE.U32.AND P1, PT, p, 0x#{BACK}, PT;
      --:-:-:-:6      MOV tm, param_t0;
      --:-:-:-:6      IADD tm, tm, p;
      --:-:-:-:6      ISETP.GE.U32.AND P2, PT, tm, 0x#{BACK}, PT;
      --:-:-:-:6      IADD rp#{J2}.CC, xp0, -0x#{BACK4};
      --:-:-:-:6      IADD.X rp#{J21}, xp1, -0x1;
      --:-:-:-:6      MOV tm, param_t0m;
      --:-:-:-:6      IADD tm, tm, p;
      --:-:-:-:6      IADD tm, tm, 0x#{BACK10};
    ]], {
      J = tostring(j),
      J2 = tostring(2 * j),
      J21 = tostring(2 * j + 1),
      BACK = string.format("%x", back),
      BACK4 = string.format("%x", 4 * back),
      BACK10 = string.format("%x", 10 - back),
    }))
    emit(mod10("i", "tm"))
    emit(heredoc([[
      --:-:-:-:6      XMAD tm, i, 0x2800, c;
      --:-:-:-:6      SHL tm, tm, 0x2;
      --:-:-:-:6      IADD op#{J2}.CC, tm, param_RI0;
      --:-:-:-:6      IADD.X op#{J21}, RZ, param_RI1;
      --:-:-:-:6      PSETP.AND.AND P3, PT, !P1, P2, PT;
      --:-:-:-:6      PSETP.AND.AND P1, PT, P1, !P0, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      PSETP.AND.AND P3, PT, P3, !P0, PT;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:-:-:6      NOP;
      --:-:3:-:1  @P1 LDG.E x#{J}, [rp#{J2}];
      --:-:3:-:1  @P3 LDG.E x#{J}, [op#{J2}];
    ]], {
      J = tostring(j),
      J2 = tostring(2 * j),
      J21 = tostring(2 * j + 1),
    }))
  end
  emit(heredoc([[
    02:-:4:-:1      F2F.F32.F16 w0, h0.H0;
    --:-:4:-:1      F2F.F32.F16 w1, h0.H1;
    --:-:4:-:1      F2F.F32.F16 w2, h1.H0;
    --:-:4:-:1      F2F.F32.F16 w3, h1.H1;
    0c:-:-:-:6      FMUL y, w0, x0;
    --:-:-:-:6      FFMA y, w1, x1, y;
    --:-:-:-:6      FFMA y, w2, x2, y;
    --:-:-:-:6      FFMA y, w3, x3, y;
    --:-:-:-:6      FMUL sg, y, nl2e;
    --:-:-:-:6      RRO.EX2 sg, sg;
    --:-:3:-:1      MUFU.EX2 sg, sg;
    04:-:-:-:6      FADD sg, sg, 1;
    --:-:3:-:1      MUFU.RCP sg, sg;
    04:-:-:-:6      FMUL y, y, sg;
    --:-:-:-:6      FADD y, gv, y;
    --:-:-:-:6      FADD sv, sv, y;
    --:1:-:-:1 @!P0 STG.E [sp], sv;
  ]]))
  -- the ring: x(t) of the chunk's last ten positions to RO[(t0 + p) mod 10]
  emit(heredoc([[
    --:-:-:-:6      MOV tm, param_npos;
    --:-:-:-:6      IADD tm, tm, -0xa;
    --:-:-:-:6      ISETP.LT.AND P5, PT, p, tm, PT;
    --:-:-:-:6      MOV tm, param_t0m;
    --:-:-:-:6      IADD tm, tm, p;
  ]]))
  emit(mod10("i", "tm"))
  emit(heredoc([[
    --:-:-:-:6      XMAD tm, i, 0x2800, c;
    --:-:-:-:6      SHL tm, tm, 0x2;
    --:-:-:-:6      IADD op6.CC, tm, param_RO0;
    --:-:-:-:6      IADD.X op7, RZ, param_RO1;
    --:-:-:-:6      PSETP.OR.AND P5, PT, P5, P0, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:1:-:-:1 @!P5 STG.E [op6], x3;
  ]]))
  -- the entries of the positions before t0 among the last ten (npos < 10): thread p = 0 of each channel copies RI's, k < 10 - npos
  emit(heredoc([[
    01:-:-:-:6      ISETP.NE.U32.OR P4, PT, p, RZ, P0;
    --:-:-:-:6      MOV i, RZ;
    COPY:
    --:-:-:-:6      MOV tm, param_npos;
    --:-:-:-:6      IADD pos, tm, i;
    --:-:-:-:6      ISETP.LT.U32.AND P1, PT, pos, 0xa, PT;
    --:-:-:-:6      MOV tm, param_t0;
    --:-:-:-:6      IADD tm, tm, pos;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, tm, 0xa, PT;
    --:-:-:-:6      MOV tm, param_t0m;
    --:-:-:-:6      IADD tt, tm, pos;
  ]]))
  emit(mod10("pos", "tt"))
  emit(heredoc([[
    --:-:-:-:6      XMAD tm, pos, 0x2800, c;
    --:-:-:-:6      SHL tm, tm, 0x2;
    --:-:-:-:6      IADD cp0.CC, tm, param_RI0;
    --:-:-:-:6      IADD.X cp1, RZ, param_RI1;
    --:-:-:-:6      IADD dp0.CC, tm, param_RO0;
    --:-:-:-:6      IADD.X dp1, RZ, param_RO1;
    --:-:-:-:6      PSETP.AND.AND P1, PT, P1, P2, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      PSETP.AND.AND P1, PT, P1, !P4, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:3:-:2  @P1 LDG.E x0, [cp];
    04:1:-:-:1  @P1 STG.E [dp], x0;
    01:-:-:-:6      IADD i, i, 0x1;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, i, 0xa, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA COPY;
    --:-:-:-:f      EXIT;
  ]]))
end

if OP == "xs" then
  -- the chunk exchange's send in 16 B words: thread q = 256 unit + tid copies word q of SRC (n16 words) to slot + myoff + 16 q of
  -- every die's inbox (TAB: u64 inbox bases of dies 0..5), write-through; a warp past the end exits whole, a lane past the end
  -- stores nothing. The framework publishes the flags (descriptor word 30 bit 0).
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-3   : x<0-3>
        4-15  : ib<00-11>
        16,17 : sp<0-1>
        18,19 : tp<0-1>
        20    : tid
        21    : bid
        22    : q
        23    : t
        24    : off
        25    : nd
        26-37 : dp<00-11>
    </REGISTER_MAPPING>
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV tp0, param_TAB0;
    --:-:-:-:6      MOV tp1, param_TAB1;
    --:-:2:-:1      LDG.E.64 ib00, [tp+0x0];
    --:-:2:-:1      LDG.E.64 ib02, [tp+0x8];
    --:-:2:-:1      LDG.E.64 ib04, [tp+0x10];
    --:-:2:-:1      LDG.E.64 ib06, [tp+0x18];
    --:-:2:-:1      LDG.E.64 ib08, [tp+0x20];
    --:-:2:-:1      LDG.E.64 ib10, [tp+0x28];
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n16, PT;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, q, param_n16, PT;
    --:-:-:-:6      MOV nd, param_nd;
    --:-:-:-:6      SHL t, q, 0x4;
    --:-:-:-:6      IADD sp0.CC, t, param_SRC0;
    --:-:-:-:6      IADD.X sp1, RZ, param_SRC1;
    --:-:-:-:6      IADD off, t, param_myoff;
    --:-:-:-:6      IADD off, off, param_slot;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, nd, 0x1, !P0;
    --:-:-:-:6      ISETP.GT.U32.AND P3, PT, nd, 0x2, !P0;
    --:-:-:-:6      ISETP.GT.U32.AND P4, PT, nd, 0x3, !P0;
    --:-:-:-:6      ISETP.GT.U32.AND P5, PT, nd, 0x4, !P0;
    --:-:-:-:6      ISETP.GT.U32.AND P6, PT, nd, 0x5, !P0;
    --:-:-:-:6  @P2 EXIT;
    --:-:3:-:2 @!P0 LDG.E.128 x0, [sp];
  ]]))
  for e = 0, 5 do
    emit(string.format("%s:-:-:-:6      IADD dp%02d.CC, off, ib%02d;\n--:-:-:-:6      IADD.X dp%02d, RZ, ib%02d;", e == 0 and "02" or "--", 2 * e, 2 * e, 2 * e + 1, 2 * e + 1))
  end
  local pr = { "!P0", "P1", "P3", "P4", "P5", "P6" }
  for e = 0, 5 do
    emit(string.format("%s:1:-:-:1  @%s STG.E.WT.128 [dp%02d], x0;", e == 0 and "04" or "--", pr[e + 1], 2 * e))
  end
  emit("01:-:-:-:6      MEMBAR.SYS;\n--:-:-:-:f      EXIT;")
end

if OP == "xh" then
  -- the chunk exchange's send in fp16: thread q = 256 unit + tid rounds floats 8 q .. 8 q + 7 of SRC to halves and stores the
  -- 16 B to slot + myoff + 16 q of the inbox of every die in dmask (TAB: u64 inbox bases of dies 0..5), write-through (n16 words);
  -- a warp past the end exits whole, a lane past the end stores nothing. The framework publishes the flags (word 30 bit 0).
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-7   : x<0-7>
        8-19  : ib<00-11>
        20,21 : sp<0-1>
        22,23 : tp<0-1>
        24    : tid
        25    : bid
        26    : q
        27    : t
        28    : off
        29    : msk
        30-41 : dp<00-11>
    </REGISTER_MAPPING>
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    --:-:-:-:6      MOV tp0, param_TAB0;
    --:-:-:-:6      MOV tp1, param_TAB1;
    --:-:2:-:1      LDG.E.64 ib00, [tp+0x0];
    --:-:2:-:1      LDG.E.64 ib02, [tp+0x8];
    --:-:2:-:1      LDG.E.64 ib04, [tp+0x10];
    --:-:2:-:1      LDG.E.64 ib06, [tp+0x18];
    --:-:2:-:1      LDG.E.64 ib08, [tp+0x20];
    --:-:2:-:1      LDG.E.64 ib10, [tp+0x28];
    01:-:-:-:6      ISCADD q, bid, tid, 0x8;
    --:-:-:-:6      LOP32I.AND t, q, 0xffffffe0;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, t, param_n16, PT;
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, q, param_n16, PT;
    --:-:-:-:6      SHL t, q, 0x5;
    --:-:-:-:6      IADD sp0.CC, t, param_SRC0;
    --:-:-:-:6      IADD.X sp1, RZ, param_SRC1;
    --:-:-:-:6      SHL off, q, 0x4;
    --:-:-:-:6      IADD off, off, param_myoff;
    --:-:-:-:6      IADD off, off, param_slot;
    --:-:-:-:6  @P2 EXIT;
    --:-:3:-:1 @!P0 LDG.E.128 x0, [sp];
    --:-:3:-:1 @!P0 LDG.E.128 x4, [sp+0x10];
    --:-:-:-:6      MOV msk, param_dmask;
  ]]))
  local pr = { "P1", "P2", "P3", "P4", "P5", "P6" }
  for e = 0, 5 do
    emit(string.format("--:-:-:-:6      LOP32I.AND t, msk, 0x%x;\n--:-:-:-:6      ISETP.NE.U32.AND %s, PT, t, RZ, !P0;", 1 << e, pr[e + 1]))
  end
  for e = 0, 5 do
    emit(string.format("%s:-:-:-:6      IADD dp%02d.CC, off, ib%02d;\n--:-:-:-:6      IADD.X dp%02d, RZ, ib%02d;", e == 0 and "02" or "--", 2 * e, 2 * e, 2 * e + 1, 2 * e + 1))
  end
  for i = 0, 7 do
    emit(string.format("%s:-:%d:-:1      F2F.F16.F32 x%d, x%d;", i == 0 and "04" or "--", 4 + i // 4, i, i))   -- (at most four conversions per barrier)
  end
  emit("18:-:-:-:6      PRMT x0, x0, 0x5410, x1;\n--:-:-:-:6      PRMT x1, x2, 0x5410, x3;\n--:-:-:-:6      PRMT x2, x4, 0x5410, x5;\n--:-:-:-:6      PRMT x3, x6, 0x5410, x7;")
  local xv = os.getenv("SLAB_XH_VAR") or ""
  for e = 0, 5 do
    emit(string.format("--:1:-:-:1  @%s STG.E%s.128 [dp%02d], x0;", pr[e + 1], xv == "nowt" and "" or ".WT", 2 * e))
  end
  emit("01:-:-:-:6      MEMBAR.SYS;" .. (xv == "membar2" and "\n--:-:-:-:6      MEMBAR.SYS;" or "") .. "\n--:-:-:-:f      EXIT;")
end

if OP == "uh" then
  -- the fp16 exchange's unpack: thread (unit u, tid) writes the 16 B words q = 1024 u + 256 j + tid (j < 4, q < n16) of fp32 to
  -- DST + 16 q from the 4 halves at 8 q of this die's inbox slot (LOC), or for q in [rlo0, rhi0) or [rlo1, rhi1) of its card
  -- partner's (PAR, over P2P: sources on other cards send to one die of each card, the partner the ones of the other parity)
  emit(heredoc([[
    <REGISTER_MAPPING>
        0-15  : fv<00-15>
        16-23 : h<0-7>
        24-31 : ap<0-7>
        32-35 : q<0-3>
        36    : tid
        37    : bid
        38,39 : b<0-1>
        40-47 : dp<0-7>
        48    : t
    </REGISTER_MAPPING>
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:1:-:1      S2R bid, SR_CTAID.X;
    01:-:-:-:6      SHL q0, bid, 0xa;
    --:-:-:-:6      IADD q0, q0, tid;
    --:-:-:-:6      ISETP.GE.U32.AND P6, PT, q0, param_n16, PT;
    --:-:-:-:6      IADD q1, q0, 0x100;
    --:-:-:-:6      IADD q2, q0, 0x200;
    --:-:-:-:6      IADD q3, q0, 0x300;
    --:-:-:-:6      LOP32I.AND t, q0, 0xffffffe0;
    --:-:-:-:6      ISETP.GE.U32.AND P5, PT, t, param_n16, PT;
    --:-:-:-:6  @P5 EXIT;
  ]]))
  for j = 0, 3 do
    emit(heredoc(string.format([[
      --:-:-:-:6      ISETP.LT.U32.AND P%d, PT, q%d, param_n16, PT;
      --:-:-:-:6      ISETP.GE.U32.AND P4, PT, q%d, param_rlo0, PT;
      --:-:-:-:6      ISETP.LT.U32.AND P4, PT, q%d, param_rhi0, P4;
      --:-:-:-:6      ISETP.GE.U32.AND P5, PT, q%d, param_rlo1, PT;
      --:-:-:-:6      ISETP.LT.U32.AND P5, PT, q%d, param_rhi1, P5;
      --:-:-:-:6      MOV b0, param_LOC0;
      --:-:-:-:6      MOV b1, param_LOC1;
      --:-:-:-:6  @P4 MOV b0, param_PAR0;
      --:-:-:-:6  @P4 MOV b1, param_PAR1;
      --:-:-:-:6  @P5 MOV b0, param_PAR0;
      --:-:-:-:6  @P5 MOV b1, param_PAR1;
      --:-:-:-:6      SHL t, q%d, 0x3;
      --:-:-:-:6      IADD ap%d.CC, t, b0;
      --:-:-:-:6      IADD.X ap%d, RZ, b1;
      --:-:2:-:1  @P%d LDG.E.CV.64 h%d, [ap%d];
    ]], j, j, j, j, j, j, j, 2 * j, 2 * j + 1, j, 2 * j, 2 * j)))
  end
  for j = 0, 3 do
    local w = j == 0 and "02" or "--"
    emit(string.format("%s:-:-:-:6      LOP32I.AND fv%02d, h%d, 0xffff;\n--:-:-:-:6      SHR.U32 fv%02d, h%d, 0x10;\n--:-:-:-:6      LOP32I.AND fv%02d, h%d, 0xffff;\n--:-:-:-:6      SHR.U32 fv%02d, h%d, 0x10;",
      w, 4 * j, 2 * j, 4 * j + 1, 2 * j, 4 * j + 2, 2 * j + 1, 4 * j + 3, 2 * j + 1))
  end
  for i = 0, 15 do
    emit(string.format("--:-:%d:-:1      F2F.F32.F16 fv%02d, fv%02d;", 3 + i // 4, i, i))   -- (at most four conversions per barrier)
  end
  if os.getenv("SLAB_UH_VAR") == "nop" then emit("3c:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;") end   -- (debugging: cycles between the conversions' barrier and their use)
  -- (an address register per store: a 128-bit store's read barrier can clear before its last lines re-read the address, so a
  -- shared one, moved for the next store, sent the upper lanes' 16 B there)
  for j = 0, 3 do
    emit(string.format("--:-:-:-:6      SHL t, q%d, 0x4;\n--:-:-:-:6      IADD dp%d.CC, t, param_DST0;\n--:-:-:-:6      IADD.X dp%d, RZ, param_DST1;", j, 2 * j, 2 * j + 1))
  end
  for j = 0, 3 do
    emit(string.format("%s:1:-:-:1  @P%d STG.E.128 [dp%d], fv%02d;", j == 0 and "3c" or "--", j, 2 * j, 4 * j))
  end
  emit("01:-:-:-:f      EXIT;")
  if os.getenv("SLAB_UH_VAR") == "cg" then   -- (debugging: the inbox read through L2 only)
    for i, s in ipairs(out) do out[i] = s:gsub("LDG%.E%.CV%.64 h", "LDG.E.CG.64 h") end
  end
end

-- A barrier an instruction sets is active a cycle after it issues: an instruction that waits on it right after (stall 1) can pass
-- before it is set. Such stalls become 2.
local function barfix(text)
  local lines = ruby_lines(text)
  local n = #lines
  for i = 1, n - 1 do
    local _, rd, wr = ctl_match(lines[i])
    if rd then
      local j = i + 1
      while j <= n and (lines[j]:match("^%s*$") or chomp(lines[j]):match("^%s*[%w_%.]+:%s*$") or lines[j]:match("^%s*#")) do
        j = j + 1
      end
      local mh = j <= n and lines[j]:match("^(%x%x):")
      if mh then
        local mask = tonumber(mh, 16)
        local hit = false
        for _, bar in ipairs({ rd, wr }) do
          if bar ~= "-" and (mask & (1 << (tonumber(bar) - 1))) ~= 0 then
            hit = true
          end
        end
        if hit then
          lines[i] = lines[i]:sub(1, 9) .. "2" .. lines[i]:sub(11)
        end
      end
    end
  end
  return table.concat(lines)
end

-- every exit fences the op's stores (sassfence.lua)
io.write(dofile((arg[0]:match("^(.*/)") or "") .. "sassfence.lua").fence_exits(barfix(table.concat(out))))
