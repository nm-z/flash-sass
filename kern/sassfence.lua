-- sassfence.lua: fence_exits(text) puts a MEMBAR.GL before every EXIT of a die-op body (same predicate; the EXIT's wait
-- mask moves to the fence). The die kernel's grid barrier fences warp 0 only: a phase that reads what another SM's CTA
-- stored in the phase before can see stale data unless every thread fenced its own stores (cattn app -> attn did).
local M = {}

local function fence_line(line)
  local ind, wait, rd, wr, yield, stall, sp, pred, sp2 = line:match("^(%s*)(%S%S):(%S):(%S):(%S):(%S)(%s+)(@!?P%d)(%s+)EXIT;%s*$")
  if wait then
    return string.format("%s%s:-:-:-:6%s%s%sMEMBAR.GL;\n%s--:%s:%s:%s:%s%s%s%sEXIT;", ind, wait, sp, pred, sp2, ind, rd, wr, yield, stall, sp, pred, sp2)
  end
  ind, wait, rd, wr, yield, stall, sp = line:match("^(%s*)(%S%S):(%S):(%S):(%S):(%S)(%s+)EXIT;%s*$")
  if wait then
    return string.format("%s%s:-:-:-:6%sMEMBAR.GL;\n%s--:%s:%s:%s:%s%sEXIT;", ind, wait, sp, ind, rd, wr, yield, stall, sp)
  end
  return line
end

function M.fence_exits(text)
  return (text:gsub("[^\n]+", fence_line))
end

return M
