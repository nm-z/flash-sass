-- The final argmax butterfly leaves the same winner in every lane of its
-- surviving warp. Publish to the six peer inboxes with six lanes, instead of
-- sending every peer serially from lane0. Payloads, offsets, reduction order,
-- local result/counter writes and the phase-footer system fence are unchanged.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local routine=assert(s:find('\nargmax_SETUP:',1,true))
local first=assert(s:find('--:-:1:-:2 LDS.64 edge_table0, [RZ+0xbfb8];',routine,true))
local last=assert(s:find('\nargmax_PUB_DONE:',first,true))
local old=s:sub(first,last-1)
assert(old:find('argmax_PUB_LOOP:',1,true) and old:find('@P5 BRA argmax_PUB_LOOP;',1,true))
assert(old:find('STG.E.WT [edge_dest], argmax_im;',1,true))
local replacement=[=[--:-:1:-:2 LDS.64 edge_table0, [RZ+0xbfb8];
--:-:2:-:2 LDS edge_offset, [RZ+0xbfc0];
--:-:-:Y:d SHL edge_peer, lane, 0x3;
--:-:-:Y:d ISETP.LT.U32.AND P4, PT, lane, 0x6, PT;
01:-:-:Y:d IADD edge_table0.CC, edge_table0, edge_peer;
--:-:-:Y:d IADD.X edge_table1, edge_table1, RZ;
--:-:1:-:2 @P4 LDG.E.CG.64 edge_dest0, [edge_table];
03:-:-:Y:d IADD edge_dest0.CC, edge_dest0, edge_offset;
--:-:-:Y:d IADD.X edge_dest1, edge_dest1, RZ;
--:2:-:-:2 @P4 STG.E.WT [edge_dest], argmax_im;
--:2:-:-:2 @P4 STG.E.WT [edge_dest+0x4], argmax_m;
02:-:-:Y:d NOP;]=]
s=s:sub(1,first-1)..replacement..s:sub(last)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
