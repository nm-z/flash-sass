-- Every CTA publishes its winner; the existing final AMAX phase reduces all
-- 6*16 pairs. The exchange acquire already joins these producers, so a second
-- last-CTA atomic handoff and local reduction are unnecessary.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local routine=assert(s:find('\nargmax_SETUP:',1,true))
local first=assert(s:find('--:-:-:-:6      SHL argmax_t, argmax_cta, 0x3;',routine,true))
local last=assert(s:find('\nargmax_END:',first,true))
assert(s:sub(first,last):find('ATOM.E.ADD',1,true))
local replacement=[=[--:-:-:Y:d ISETP.NE.U32.AND P6, PT, argmax_warp, RZ, PT;
--:-:-:Y:d @P6 BRA argmax_END;
--:-:-:Y:d SHL argmax_t, argmax_cta, 0x3;
--:-:-:Y:d IADD argmax_pp0.CC, argmax_t, argmax_param_PART0;
--:-:-:Y:d IADD.X argmax_pp1, argmax_param_PART1, RZ;
--:-:-:Y:d ISETP.EQ.U32.AND P4, PT, lane, RZ, PT;
--:1:-:-:2 @P4 STG.E [argmax_pp], argmax_im;
--:1:-:-:2 @P4 STG.E [argmax_pp+0x4], argmax_m;
01:-:1:-:2 LDS edge_flags, [RZ+0xbff8];
01:-:-:Y:d LOP32I.AND edge_flags, edge_flags, 0x80;
--:-:-:Y:d ISETP.EQ.U32.AND P6, PT, edge_flags, RZ, PT;
--:-:-:Y:d @P6 BRA argmax_END;
--:-:1:-:2 LDS.64 edge_table0, [RZ+0xbfb8];
--:-:2:-:2 LDS edge_offset, [RZ+0xbfc0];
--:-:-:Y:d SHL edge_peer, lane, 0x3;
--:-:-:Y:d ISETP.LT.U32.AND P4, PT, lane, 0x6, PT;
01:-:-:Y:d IADD edge_table0.CC, edge_table0, edge_peer;
--:-:-:Y:d IADD.X edge_table1, edge_table1, RZ;
02:-:-:Y:d IADD edge_offset, edge_offset, argmax_t;
--:-:1:-:2 @P4 LDG.E.CG.64 edge_dest0, [edge_table];
01:-:-:Y:d IADD edge_dest0.CC, edge_dest0, edge_offset;
--:-:-:Y:d IADD.X edge_dest1, edge_dest1, RZ;
--:2:-:-:2 @P4 STG.E.WT [edge_dest], argmax_im;
--:2:-:-:2 @P4 STG.E.WT [edge_dest+0x4], argmax_m;
02:-:-:Y:d NOP;]=]
s=s:sub(1,first-1)..replacement..s:sub(last)

-- Lane l reduces pairs l, l+32, l+64 before the unchanged warp butterfly.
-- Pair p belongs to die floor(p/16), whose original vocabulary base is added
-- before comparison. No new floating-point accumulation is introduced.
routine=assert(s:find('\namax_SETUP:',1,true))
first=assert(s:find('--:-:-:-:6      ISETP.LT.U32.AND P0, PT, amax_lane, amax_param_ndies, PT;',routine,true))
last=assert(s:find('--:-:2:-:1      SHFL.BFLY PT, amax_mo, amax_m,',first,true))
local out={}
local function e(x)out[#out+1]='--:-:-:Y:d '..x..';'end
for round=0,2 do
 e(string.format('IADD amax_t, amax_lane, 0x%x',32*round))
 e('SHL amax_t, amax_t, 0x3')
 e('IADD amax_pp0.CC, amax_t, amax_param_PAIRS0');e('IADD.X amax_pp1, amax_param_PAIRS1, RZ')
 e('SHR.U32 amax_t, amax_lane, 0x4');e(string.format('IADD amax_t, amax_t, 0x%x',2*round))
 e('SHL amax_t, amax_t, 0x2')
 e('IADD amax_bp0.CC, amax_t, amax_param_BASES0');e('IADD.X amax_bp1, amax_param_BASES1, RZ')
 out[#out+1]='--:-:1:-:2 LDG.E.CG amax_io, [amax_pp];'
 out[#out+1]='--:-:2:-:2 LDG.E.CG amax_mo, [amax_pp+0x4];'
 out[#out+1]='--:-:3:-:2 LDG.E.CG amax_base, [amax_bp];'
 out[#out+1]='07:-:-:Y:d IADD amax_io, amax_io, amax_base;'
 e('FSETP.GT.AND P1, PT, amax_mo, amax_m, PT')
 e('FSETP.EQ.AND P2, PT, amax_mo, amax_m, PT')
 e('ISETP.LT.U32.AND P2, PT, amax_io, amax_i, P2')
 e('PSETP.OR.AND P1, PT, P1, P2, PT')
 e('@P1 MOV amax_m, amax_mo');e('@P1 MOV amax_i, amax_io')
end
s=s:sub(1,first-1)..table.concat(out,'\n')..'\n'..s:sub(last)
assert(not s:sub(s:find('\nargmax_SETUP:',1,true),s:find('\nargmax_END:',1,true)):find('ATOM.E.ADD',1,true))
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
