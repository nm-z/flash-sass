#!/usr/bin/env lua5.4
-- sbcheck.lua < op.sass: the scoreboard hazards of a MaxAs op body, in program order (branches are not followed; a loop body is
-- checked a second time after its backward branch with the barriers still outstanding at the branch):
--   RAW  an instruction reads a register a variable-latency instruction writes (write barrier b) with b not waited on since,
--   WAR  an instruction writes a register an earlier instruction reads late (read barrier b: stores, loads' addresses) with b
--        not waited on since,
--   WAW  an instruction writes a register an earlier variable-latency instruction still writes,
--   LAT  (SB_LAT=1) an instruction reads a register a fixed-latency instruction wrote fewer than 6 issue cycles before (the
--        stall counts in between; branches reset the count; memory instructions' addresses are read late and not counted),
--   STW  (SB_STW=1) an instruction writes a register a global store / reduction (STG, RED, ATOM) read fewer than 40 issue cycles
--        before, even with its read barrier waited on: a store's late lines (one 128 B line per replay) re-read their address and
--        data, so a moved address register sent upper lanes elsewhere (slab_uh),
--   UNINIT  (SB_UNINIT=1) an instruction reads a register nothing before it wrote (on the first pass in program order).
-- Registers come from the REGISTER_MAPPING block (names, ranges, x<0-3> lists); .64 / .128 widen a data operand, [x] names the
-- pair x0, x1 or the register x and x + 1 of a 64-bit address (.E).

local text = io.read("a")
local reg = {}   -- name -> number

local function expand(field)
  local out = {}
  for tok in field:gmatch("[^%s,]+") do
    local pre, a, b, post = tok:match("^([%w_]*)<(%d+)%-(%d+)>([%w_]*)$")
    if pre then
      local w = #a
      for i = tonumber(a), tonumber(b) do out[#out + 1] = pre .. string.format("%0" .. w .. "d", i) .. post end
    else
      out[#out + 1] = tok
    end
  end
  return out
end

for block in text:gmatch("<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>") do
  for line in block:gmatch("[^\n]+") do
    local nums, names = line:match("^%s*([%d%s,%-]+)%s*:%s*(.-)%s*$")
    if nums and names and names ~= "" then
      local n = {}
      for tok in nums:gmatch("[^%s,]+") do
        local a, b = tok:match("^(%d+)%-(%d+)$")
        if a then for i = tonumber(a), tonumber(b) do n[#n + 1] = i end else n[#n + 1] = tonumber(tok) end
      end
      local nm = expand(names)
      if #n == #nm then for i = 1, #n do reg[nm[i]] = n[i] end end
    end
  end
end

local function regs_of(op, width, out)
  -- a data operand: the register (and the next width - 1)
  op = op:gsub("^%-", ""):gsub("^|", ""):gsub("|$", ""):gsub("%.reuse$", ""):gsub("%.CC$", ""):gsub("%.H[01]$", ""):gsub("%.B[0-3]$", "")
  local r = reg[op]
  if r then for i = 0, width - 1 do out[#out + 1] = r + i end end
end

local function addr_regs(inner, out, wide)
  -- [x], [x+0x10], [x0+...]: a 64-bit pair (x0, x1 for a name x without a number of its own), or x and x + 1
  local base = inner:match("^%s*([%w_]+)")
  if not base or base == "RZ" then return end
  if reg[base] then out[#out + 1] = reg[base]; if wide then out[#out + 1] = reg[base] + 1 end
  elseif reg[base .. "0"] then out[#out + 1] = reg[base .. "0"]; out[#out + 1] = reg[base .. "0"] + 1 end
end

local NODST = { STG = 1, STS = 1, STL = 1, RED = 1, BRA = 1, BAR = 1, NOP = 1, MEMBAR = 1, EXIT = 1, CCTL = 1, SYNC = 1, BRX = 1 }
local function decode(body)
  local pred, rest = body:match("^(@!?P%d)%s+(.*)$")
  rest = rest or body
  local opc, ops = rest:match("^([%w%.]+)%s*(.-);")
  if not opc then return nil end
  local base = opc:match("^(%w+)")
  local width = opc:match("%.128") and 4 or (opc:match("%.64") and 2 or 1)
  local operands = {}
  for o in (ops .. ","):gmatch("%s*(.-)%s*,") do operands[#operands + 1] = o end
  local rd, wr, ad = {}, {}, {}
  local isstore = base == "STG" or base == "STS" or base == "STL" or base == "RED"
  local dsti
  if not NODST[base] and #operands > 0 then
    dsti = 1
    if base == "SHFL" or base:match("SETP$") then dsti = (base == "SHFL") and 2 or nil end
  end
  for i, o in ipairs(operands) do
    local inner = o:match("^%[(.*)%]$")
    if inner then addr_regs(inner, rd, opc:match("%.E[%.%s]") or opc:match("%.E$")); addr_regs(inner, ad, opc:match("%.E[%.%s]") or opc:match("%.E$"))
    elseif i == dsti then
      local wdt = (base == "LDG" or base == "LDS" or base == "LDL" or base == "LD" or base == "ATOM") and width or 1
      regs_of(o, wdt, wr)
    else
      local wdt = (isstore and i == #operands) and width or 1
      regs_of(o, wdt, rd)
    end
  end
  if pred then end
  return opc, rd, wr, ad
end

local lines = {}
for l in text:gmatch("[^\n]*") do lines[#lines + 1] = l end

local pendw, pendr = {}, {}   -- register -> barrier (1..6)
local written, first = {}, true
local ready, cyc = {}, 0   -- register -> the cycle its fixed-latency result is ready; the issue cycle
local stread = {}          -- register -> the issue cycle of the last global store that read it
local seen = {}
local function report(kind, n, r, b, line)
  local key = kind .. n .. ":" .. r
  if seen[key] then return end
  seen[key] = true
  local name = "?"
  for k, v in pairs(reg) do if v == r and (name == "?" or #k < #name) then name = k end end
  io.write(string.format("%s line %d: R%d (%s) barrier %d: %s\n", kind, n, r, name, b, line:gsub("^%s+", "")))
end

local labels = {}
for i, l in ipairs(lines) do local lb = l:match("^%s*([%w_]+):%s*$"); if lb then labels[lb] = i end end

local function run(from, to)
  local i = from
  while i <= to do
    local l = lines[i]
    local wait, rb, wb, stall, body = l:match("^%s*(%S%S):(%S):(%S):%S:(%S)%s+(.*)$")
    if l:match("^%s*[%w_]+:%s*$") then ready = {}; stread = {} end   -- a label: control may arrive from anywhere
    if wait then
      if wait ~= "--" then
        local m = tonumber(wait, 16)
        for r, b in pairs(pendw) do if m & (1 << (b - 1)) ~= 0 then pendw[r] = nil end end
        for r, b in pairs(pendr) do if m & (1 << (b - 1)) ~= 0 then pendr[r] = nil end end
      end
      local opc, rd, wr, ad = decode(body)
      if opc then
        for _, r in ipairs(rd) do if pendw[r] then report("RAW", i, r, pendw[r], l) end end
        if first and os.getenv("SB_UNINIT") then for _, r in ipairs(rd) do if not written[r] then report("UNINIT", i, r, 0, l) end end end
        for _, r in ipairs(wr) do written[r] = true end
        local isad = {}; for _, r in ipairs(ad or {}) do isad[r] = true end
        if os.getenv("SB_STW") then for _, r in ipairs(wr) do if stread[r] and cyc - stread[r] < 40 then report("STW", i, r, cyc - stread[r], l) end end end
        if opc:match("^ST[G]?%.E") or opc:match("^STG") or opc:match("^RED") or opc:match("^ATOM") then for _, r in ipairs(rd) do stread[r] = cyc end end
        if os.getenv("SB_LAT") then for _, r in ipairs(rd) do if not isad[r] and ready[r] and ready[r] > cyc then report("LAT", i, r, ready[r] - cyc, l) end end end
        local fixed = wb == "-" and not opc:match("^ST") and not opc:match("^RED")
        for _, r in ipairs(wr) do ready[r] = fixed and cyc + 6 or nil end
        cyc = cyc + (tonumber(stall, 16) or 1)
        if opc:match("^BRA") or opc:match("^BRX") then ready = {} end
        if (opc:match("^BRA") or opc:match("^EXIT")) and not body:match("^@") then pendw, pendr = {}, {} end   -- (what follows is reached by a jump)
        for _, r in ipairs(wr) do
          if pendr[r] then report("WAR", i, r, pendr[r], l) end
          if pendw[r] then report("WAW", i, r, pendw[r], l) end
        end
        for _, r in ipairs(wr) do pendw[r] = nil end
        if wb ~= "-" then for _, r in ipairs(wr) do pendw[r] = tonumber(wb) end end
        if rb ~= "-" then for _, r in ipairs(rd) do pendr[r] = tonumber(rb) end end
        local tgt = body:match("BRA%s+([%w_]+);")
        if tgt and labels[tgt] and labels[tgt] < i and not seen["loop" .. i] then
          seen["loop" .. i] = true
          local f0 = first; first = false
          run(labels[tgt], i)   -- the loop body again with what is still outstanding
          first = f0
        end
      end
    end
    i = i + 1
  end
end
local nr = 0; for _ in pairs(reg) do nr = nr + 1 end
if os.getenv("SB_VERBOSE") then io.stderr:write(nr, " registers named\n") end
run(1, #lines)
