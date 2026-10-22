-- The sixteen CTA winners stay in their producing die's buffer. Existing
-- system-fenced producer flags/acquires order the final warp's peer reads.
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
--:1:-:-:2 @P4 STG.E.WT [argmax_pp], argmax_im;
--:1:-:-:2 @P4 STG.E.WT [argmax_pp+0x4], argmax_m;
01:-:-:Y:d NOP;]=]
s=s:sub(1,first-1)..replacement..s:sub(last)

-- These scratch registers are dead in AMAX (its live registers end at55).
-- Keep three independent requests in flight instead of serializing three
-- PCIe round trips. CV reads prevent reuse of a previous token's peer data.
local map=assert(s:find('</REGISTER_MAPPING>',1,true))
s=s:sub(1,map-1)..[=[    180-185 : bh_pair<00-05>
    186-188 : bh_base<0-2>
]=]..s:sub(map)
routine=assert(s:find('\namax_SETUP:',1,true))
first=assert(s:find('--:-:-:-:6      ISETP.LT.U32.AND P0, PT, amax_lane, amax_param_ndies, PT;',routine,true))
last=assert(s:find('--:-:2:-:1      SHFL.BFLY PT, amax_mo, amax_m,',first,true))
local out={}
local function e(x)out[#out+1]='--:-:-:Y:d '..x..';'end
-- Logical2 is physical6 in the current compute mapping; its pair is issued
-- last so every die has a chipset-crossing request in the final issue group.
-- All six peer buffers and all96 values are consumed regardless of ordering.
local order={0,4,2}
for r=0,2 do
 out[#out+1]='08:-:-:Y:d NOP;'
 e('SHR.U32 amax_t, amax_lane, 0x4');e(string.format('IADD amax_t, amax_t, 0x%x',order[r+1]))
 e('SHL amax_base, amax_t, 0x2')
 e('IADD amax_bp0.CC, amax_base, amax_param_BASES0');e('IADD.X amax_bp1, amax_param_BASES1, RZ')
 out[#out+1]=string.format('--:4:5:-:2 LDG.E.CG bh_base%d, [amax_bp];',r)
 e('SHL amax_t, amax_t, 0x3')
 e('IADD amax_pp0.CC, amax_t, amax_param_PAIRS0');e('IADD.X amax_pp1, amax_param_PAIRS1, RZ')
 out[#out+1]='--:-:6:-:2 LDG.E.CG.64 amax_pp0, [amax_pp];'
 e('LOP32I.AND amax_t, amax_lane, 0xf');e('SHL amax_t, amax_t, 0x3')
 out[#out+1]='20:-:-:Y:d IADD amax_pp0.CC, amax_pp0, amax_t;'
 e('IADD.X amax_pp1, amax_pp1, RZ')
 out[#out+1]=string.format('--:4:%d:-:2 LDG.E.CV.64 bh_pair%02d, [amax_pp];',r+1,2*r)
end
out[#out+1]='3f:-:-:Y:d NOP;'
for r=0,2 do
 e(string.format('IADD amax_io, bh_pair%02d, bh_base%d',2*r,r))
 e(string.format('MOV amax_mo, bh_pair%02d',2*r+1))
 e('FSETP.GT.AND P1, PT, amax_mo, amax_m, PT')
 e('FSETP.EQ.AND P2, PT, amax_mo, amax_m, PT')
 e('ISETP.LT.U32.AND P2, PT, amax_io, amax_i, P2')
 e('PSETP.OR.AND P1, PT, P1, P2, PT')
 e('@P1 MOV amax_m, amax_mo');e('@P1 MOV amax_i, amax_io')
end
s=s:sub(1,first-1)..table.concat(out,'\n')..'\n'..s:sub(last)
assert(not s:sub(s:find('\nargmax_SETUP:',1,true),s:find('\nargmax_END:',1,true)):find('ATOM.E.ADD',1,true))
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
