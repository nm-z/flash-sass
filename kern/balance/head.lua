-- The output producer reduces its sixteen CTA winners locally. Existing
-- system-fenced producer flags/acquires order six final peer-winner reads.
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
-- Six lanes fetch the six die winners in one request wave. CV reads prevent
-- reuse of a previous token's peer data; inactive lanes retain -inf/UINT_MAX.
local map=assert(s:find('</REGISTER_MAPPING>',1,true))
s=s:sub(1,map-1)..[=[    180-185 : bh_pair<00-05>
    186-188 : bh_base<0-2>
]=]..s:sub(map)
routine=assert(s:find('\namax_SETUP:',1,true))
first=assert(s:find('--:-:-:-:6      ISETP.LT.U32.AND P0, PT, amax_lane, amax_param_ndies, PT;',routine,true))
last=assert(s:find('--:-:2:-:1      SHFL.BFLY PT, amax_mo, amax_m,',first,true))
local reads=[=[--:-:-:Y:d ISETP.LT.U32.AND P0, PT, amax_lane, amax_param_ndies, PT;
--:-:-:Y:d SHL amax_t, amax_lane, 0x2;
--:-:-:Y:d IADD amax_bp0.CC, amax_t, amax_param_BASES0;
--:-:-:Y:d IADD.X amax_bp1, amax_param_BASES1, RZ;
--:-:5:-:2 @P0 LDG.E.CG bh_base0, [amax_bp];
--:-:-:Y:d SHL amax_t, amax_lane, 0x3;
--:-:-:Y:d IADD amax_pp0.CC, amax_t, amax_param_PAIRS0;
--:-:-:Y:d IADD.X amax_pp1, amax_param_PAIRS1, RZ;
--:-:6:-:2 @P0 LDG.E.CG.64 amax_pp0, [amax_pp];
20:-:1:-:2 @P0 LDG.E.CV.64 bh_pair00, [amax_pp];
11:-:-:Y:d @P0 IADD amax_i, bh_pair00, bh_base0;
--:-:-:Y:d @P0 MOV amax_m, bh_pair01;
]=]
s=s:sub(1,first-1)..reads..s:sub(last)
assert(not s:sub(s:find('\nargmax_SETUP:',1,true),s:find('\nargmax_END:',1,true)):find('ATOM.E.ADD',1,true))
s=dofile('balance/head_producer.lua')(s)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
