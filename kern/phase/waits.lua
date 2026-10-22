-- Remove only the local grid poll already subsumed by the following exchange
-- acquire. Stores/fences, peer completion flags and generation updates remain.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local at=assert(s:find('\nPHASE_END:',1,true));local prefix=s:sub(1,at-1);local tail=s:sub(at)
local function once(a,b)
 local i=assert(tail:find(a,1,true),'missing '..a)
 assert(not tail:find(a,i+#a,true),'ambiguous '..a)
 tail=tail:sub(1,i-1)..b..tail:sub(i+#a)
end
once('BSPIN:',[=[# Drain the publication stores' register reads before reusing xc0/bar_old.
02:-:1:-:2      LDS bar_old, [RZ+0xbf58];
01:-:-:-:6      LOP32I.AND bar_old, bar_old, 0x20000;
--:-:-:-:6      ISETP.NE.U32.AND P4, PT, bar_old, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P4 BRA EXCHANGE_ARRIVAL_DONE;
BSPIN:]=])
once('02:-:-:-:6      IADD bar_t, cta, tid;',
 'EXCHANGE_ARRIVAL_DONE:\n02:-:-:-:6      IADD bar_t, cta, tid;')
f=assert(io.open(assert(arg[2]),'w'));f:write(prefix..tail);f:close()
