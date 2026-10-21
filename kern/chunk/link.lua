-- Append native chunk operators to the scalar die's descriptor ABI.
-- Usage: lua5.4 chunk/link.lua BASE.sass BASE_HEADER OUTPUT.sass OUTPUT.h OP_DIR
-- The scalar program and its opcode numbers remain intact. New operators reuse
-- its 16 framework registers, 16 resident CTAs, descriptor and phase barriers.
local function read(path)
  local f = assert(io.open(path, "rb")); local s = f:read("a"); f:close(); return s
end
local function write(path, s)
  local f = assert(io.open(path, "wb")); assert(f:write(s)); assert(f:close())
end
local tags = {}
for f in ("bf16 q2 q3 q4k q5k q6k iq4 q40 q50 iq4nl q80"):gmatch("%S+") do tags[#tags+1] = "gemm_"..f end
for f in ("silu mix write rms conv bias router tr group swiglu saxpy zero rs sum pgate pconv xs xh uh"):gmatch("%S+") do tags[#tags+1] = "slab_"..f end
for f in ("gu gu2 dn"):gmatch("%S+") do tags[#tags+1] = "ebatch_"..f end
for f in ("kv idx sel app attn"):gmatch("%S+") do tags[#tags+1] = "cattn_"..f end
tags[#tags+1] = "dnstepc"
tags[#tags+1] = "qact_s"
tags[#tags+1] = "qact_p"

local function replace_once(s, from, to)
  local a,b = assert(s:find(from, 1, true), "missing anchor: "..from), nil
  b = a + #from - 1
  assert(not s:find(from, b+1, true), "nonunique anchor: "..from)
  return s:sub(1,a-1)..to..s:sub(b+1)
end
local function expand(field)
  local names = {}
  for tok in field:gmatch("[^,%s]+") do
    local n,a,b = tok:match("^(%w+)<(%d+)%-(%d+)>$")
    if n then
      for r=tonumber(a),tonumber(b) do names[#names+1] = n..string.format("%0"..#a.."d",r) end
    else names[#names+1] = tok end
  end
  return names
end
local function op(tag, path)
  local s = read(path):gsub("#[^\n]*", "")
  assert(not s:find("<CODE>",1,true), "operator must be expanded SASS: "..tag)
  if tag == "dnstepc" then
    local map = assert(s:match("<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>"))
    return map, (s:gsub("<REGISTER_MAPPING>.-</REGISTER_MAPPING>", ""))
  end
  local params, names, maps, maxreg = {}, {}, {}, -1
  s = s:gsub("<CONSTANT_MAPPING>(.-)</CONSTANT_MAPPING>", function(m)
    for name,bank,addr in m:gmatch("([%w_]+)%s*:%s*c%[0x([02])%]%[0x(%x+)%]") do
      addr = tonumber(addr,16)
      assert(bank == "0", "chunk operator cannot access paired descriptor")
      assert(addr == 0x14 or (addr >= 0x140 and addr < 0x1b0), "parameter overlaps descriptor flags")
      params[#params+1] = {name, addr == 0x14 and 1 or 2+(addr-0x140)//4}
    end
    return ""
  end)
  s = s:gsub("<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>", function(m)
    for line in m:gmatch("[^\n]+") do
      local nums, field = line:match("^%s*([%d,%s%-]+)%s*:%s*(.-)%s*$")
      if nums then
        nums=nums:gsub("%s", "")
        local regs, a, b = {}, nums:match("^(%d+)%-(%d+)$")
        if a then for r=tonumber(a),tonumber(b) do regs[#regs+1]=r end
        else for r in nums:gmatch("%d+") do regs[#regs+1]=tonumber(r) end end
        local expanded=expand(field)
        assert(#regs==#expanded, tag..": mapping length: "..line)
        for _,n in ipairs(expanded) do names[n]=true end
        for _,r in ipairs(regs) do maxreg=math.max(maxreg,r) end
        local shifted={}; for _,r in ipairs(regs) do shifted[#shifted+1]=r+16 end
        local fields={}
        for tok in field:gmatch("[^,%s]+") do
          local base=tok:match("^([%w_]+)<")
          if base then names[base]=true end
          fields[#fields+1]=tag.."_"..tok
        end
        maps[#maps+1]="    "..table.concat(shifted,",").." : "..table.concat(fields,", ").."\n"
      end
    end
    return ""
  end)
  local pro={}
  for i,p in ipairs(params) do
    names[p[1]]=true
    maps[#maps+1]=string.format("    %d : %s_%s\n",16+maxreg+i,tag,p[1])
    pro[#pro+1]=string.format("--:-:1:-:%d      LDS %s_%s, [RZ+0x%x];\n",i==#params and 2 or 1,tag,p[1],0xbf80+4*p[2])
  end
  assert(16+maxreg+#params <= 255, tag..": register overflow")
  if #params>0 then pro[#pro+1]="01:-:-:-:6      NOP;\n" end
  for label in s:gmatch("\n%s*([%w_]+):%s*\n") do names[label]=true end
  s=s:gsub("%f[%w_][%a_][%w_]*", function(n) return names[n] and tag.."_"..n or n end)
  s=s:gsub("[^\n]*S2R%s+([%w_]+),%s*SR_TID%.X;[^\n]*", "--:-:-:-:6      MOV %1, vtid;")
  s=s:gsub("[^\n]*S2R%s+([%w_]+),%s*SR_CTAID%.X;[^\n]*", "--:-:-:-:6      MOV %1, vid;")
  s=s:gsub("EXIT;", "BRA "..tag.."_END;")
  local setup=tag.."_SETUP:\n--:-:-:-:6      MOV vid, cta;\n--:-:-:-:6      MOV vtid, tid;\n"
    .."--:-:-:-:6      ISETP.GE.U32.AND P6, PT, vid, units, PT;\n"
    ..string.rep("--:-:-:-:6      NOP;\n",3).."--:-:-:-:6  @P6 BRA POSITION_END;\n"..tag.."_BEGIN:\n"
  local tail=tag.."_END:\n3f:-:-:-:6      BAR.SYNC 0x0;\n--:-:-:-:6      IADD vid, vid, 0x10;\n"
    .."--:-:-:-:6      ISETP.LT.U32.AND P6, PT, vid, units, PT;\n"
    ..string.rep("--:-:-:-:6      NOP;\n",3).."--:-:-:-:6  @P6 BRA "..tag.."_BEGIN;\n--:-:-:-:6      BRA POSITION_END;\n"
  io.stderr:write(string.format("%-14s last register %d\n",tag,16+maxreg+#params))
  return table.concat(maps),setup..table.concat(pro)..s.."\n"..tail
end

local base, header, out, hout, dir=table.unpack(arg)
assert(dir, "usage: link.lua BASE.sass BASE_HEADER OUTPUT.sass OUTPUT.h OP_DIR")
local s,h=read(base),read(header)
local count=assert(tonumber(h:match("#define OP_COUNT (%d+)")))
local maps,body,dispatch,defs,opnames={},{},{},{},{}
for i,tag in ipairs(tags) do
  local m,c=op(tag,dir.."/"..tag..".sass")
  maps[#maps+1]=m; body[#body+1]=c
  dispatch[#dispatch+1]="--:-:-:-:6      BRA "..tag.."_SETUP;\n"..string.rep("--:-:-:-:6      NOP;\n",2)
  defs[#defs+1]=string.format("#define OP_%s %d\n",tag:upper(),count+i-1)
  opnames[#opnames+1]='"'..tag..'"'
end
local n=count+#tags
s=replace_once(s,string.format("ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;",count),string.format("ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;",n))
s=replace_once(s,"</REGISTER_MAPPING>",table.concat(maps).."</REGISTER_MAPPING>")
s=replace_once(s,"--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:",
  table.concat(dispatch).."--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:")
-- Insert after all existing operators, immediately before POSITION_END. Do not
-- fall through into the appended bodies from normal scalar completion.
s=replace_once(s,"POSITION_END:",table.concat(body,"\n").."\nPOSITION_END:")
if os.getenv("DIE_FENCE_PHASE") == "1" then
  -- The legacy grid barrier fences warp zero. This diagnostic variant first
  -- orders every writer's stores before any thread enters the phase boundary.
  s=replace_once(s,"POSITION_END:\n", "POSITION_END:\n3f:-:-:-:6      MEMBAR.GL;\n")
end
if os.getenv("DIE_COHERENT_LOADS") == "1" then
  -- A persistent kernel rewrites activations while it is still running.
  -- Maxwell's .CI read-only path is non-coherent. Start with a uniform
  -- diagnostic conversion; retain .CI only for proven immutable data later.
  local replaced
  s,replaced=s:gsub("LDG%.E%.CI", "LDG.E")
  assert(replaced>0,"no read-only global loads found")
  io.stderr:write(string.format("coherent global loads: replaced %d instructions\n",replaced))
end
local table_code=assert(s:match("DISPATCH_TABLE:\n(.-)\nembrow_SETUP:"))
local instructions={}
for line in table_code:gmatch("[^\n]+") do
  if line:match("^%x%x:") or line:match("^%-%-:") then instructions[#instructions+1]=line end
end
assert(#instructions==3*n+1, "dispatch table has an unexpected stride or fallback")
for i,tag in ipairs(tags) do
  assert(instructions[3*(count+i-1)+1]:find("BRA "..tag.."_SETUP;",1,true),"misaligned dispatch: "..tag)
end
write(out,s)
write(hout,"/* generated by chunk/link.lua */\n#ifndef FLASH_CHUNK_OPS_H\n#define FLASH_CHUNK_OPS_H\n#include \"dieops.h\"\n"
  ..table.concat(defs)..string.format("#define CHUNK_OP_COUNT %d\n",n)
  .."static const char *const chunk_opname[] __attribute__((unused)) = {"..table.concat(opnames,",").."};\n"
  .."static inline const char *chunk_op_name(unsigned op) { return op < OP_COUNT ? die_opname[op] : op < CHUNK_OP_COUNT ? chunk_opname[op-OP_COUNT] : \"?\"; }\n#endif\n")
