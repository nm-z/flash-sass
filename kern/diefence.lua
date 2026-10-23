-- diefence.lua: perl die.pl | lua5.4 diefence.lua > die_<name>.sass. Every thread of a CTA fences its global stores
-- (MEMBAR.GL) at the phase end, before the CTA's BAR.SYNC and the grid barrier, whose fence and arrival are warp 0's
-- alone: without it a phase on another SM can read what this phase's other warps stored before it is visible (the
-- chunk attention read stale prompt-cache rows that way). With DIE_SEND_FENCE_ALL the send phases' MEMBAR.SYS stays.
local text = io.read("a")
local done = 0
text = text:gsub("(\nPHASE_END:\n)", function(label)
  done = done + 1
  return label .. "--:-:-:-:6      MEMBAR.GL;\n"
end)
if done ~= 1 then io.stderr:write(string.format("diefence: %d PHASE_END labels (one expected)\n", done)); os.exit(1) end
io.write(text)
