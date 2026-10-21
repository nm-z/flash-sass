-- Same-CTA dependent slices use a local memory fence and CTA rendezvous only.
-- Grid-dependent chains keep their original per-descriptor grid-barrier bits.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local anchor='CHAIN_NEXT:';local i=assert(s:find(anchor,1,true))
local code=[=[CHAIN_NEXT:
--:-:1:-:2 LDS bar_g, [RZ+0xbf58];
01:-:-:-:6 LOP32I.AND bar_g, bar_g, 0x80000000;
--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, bar_g, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P5 BRA LOCAL_CHAIN_READY;
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 BAR.SYNC 0x1;
LOCAL_CHAIN_READY:]=]
s=s:sub(1,i-1)..code..s:sub(i+#anchor)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
