-- Finish the shared expert's activation/quantization in the producing phase.
-- Only its consuming warp joins producer-ready words; no grid rendezvous is
-- introduced. The auxiliary FFNACT descriptor is skipped by the chain cursor.
-- Match router.lua's 98-slot token epoch, with layer+49 in slots50..97.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',[=[    16,17 : sharedjoin_ptr<0-1>
    18 : sharedjoin_epoch
    19 : sharedjoin_layer
    20 : sharedjoin_tmp
    21 : sharedjoin_value
</REGISTER_MAPPING>]=])
for _,tag in ipairs({'q2k2560','q3kk2560','q4kk2560','q5kk2560','q6kk2560','iq4k2560',
 'b32q40','b32q50','b32q80','b32iq4nl'})do
 rep(tag..'_END:',tag..[=[_END:
3f:-:1:-:2 LDS sharedjoin_tmp, [RZ+0xbff8];
01:-:-:-:6 LOP32I.AND sharedjoin_tmp, sharedjoin_tmp, 0x10000;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, sharedjoin_tmp, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA SHARED_ACT_JOIN;
]=])
end
local body=[=[SHARED_ACT_JOIN:
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:1:-:1 LDS.64 sharedjoin_ptr0, [RZ+0xbfd8];
--:-:1:-:1 LDS sharedjoin_layer, [RZ+0xbfe0];
--:-:1:-:2 LDS sharedjoin_epoch, [RZ+0xbf70];
01:-:-:-:6 SHL sharedjoin_tmp, sharedjoin_epoch, 0x6;
--:-:-:-:6 ISCADD sharedjoin_tmp, sharedjoin_epoch, sharedjoin_tmp, 0x5;
--:-:-:-:6 ISCADD sharedjoin_epoch, sharedjoin_epoch, sharedjoin_tmp, 0x1;
--:-:-:-:6 IADD sharedjoin_epoch, sharedjoin_epoch, sharedjoin_layer;
--:-:-:-:6 SHL sharedjoin_tmp, cta, 0x2;
--:-:-:-:6 IADD sharedjoin_ptr0.CC, sharedjoin_ptr0, sharedjoin_tmp;
--:-:-:-:6 IADD.X sharedjoin_ptr1, sharedjoin_ptr1, RZ;
--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, tid, RZ, PT;
--:-:-:-:6 MOV32I sharedjoin_tmp, 0x3;
--:2:-:-:6 STS [RZ+0xbf4c], sharedjoin_tmp;
--:-:-:-:6 NOP;
--:2:-:-:2 @P5 STG.E.WT [sharedjoin_ptr], sharedjoin_epoch;
--:-:-:-:6 IADD sharedjoin_tmp, cta, warp;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, sharedjoin_tmp, RZ, PT;
02:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA POSITION_END;
--:-:1:-:2 LDS.64 sharedjoin_ptr0, [RZ+0xbfd8];
--:-:-:-:6 LOP32I.AND sharedjoin_tmp, lane, 0xf;
--:-:-:-:6 SHL sharedjoin_tmp, sharedjoin_tmp, 0x2;
01:-:-:-:6 IADD sharedjoin_ptr0.CC, sharedjoin_ptr0, sharedjoin_tmp;
--:-:-:-:6 IADD.X sharedjoin_ptr1, sharedjoin_ptr1, RZ;
SHARED_ACT_WAIT:
--:-:1:-:2 LDG.E.CV sharedjoin_value, [sharedjoin_ptr];
01:-:-:-:6 ISETP.EQ.U32.AND P5, PT, sharedjoin_value, sharedjoin_epoch, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 VOTE.ALL P5, P5;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P5 BRA SHARED_ACT_WAIT;
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 IADD sharedjoin_tmp, pidx, 0x3;
--:-:-:-:6 SHL sharedjoin_tmp, sharedjoin_tmp, 0x7;
--:-:-:-:6 IADD sharedjoin_ptr0.CC, prog0, sharedjoin_tmp;
--:-:-:-:6 IADD.X sharedjoin_ptr1, prog1, RZ;
--:-:-:-:6 SHL sharedjoin_tmp, lane, 0x2;
--:-:-:-:6 IADD sharedjoin_ptr0.CC, sharedjoin_ptr0, sharedjoin_tmp;
--:-:-:-:6 IADD.X sharedjoin_ptr1, sharedjoin_ptr1, RZ;
--:-:1:-:2 LDG.E sharedjoin_value, [sharedjoin_ptr];
01:2:-:-:6 STS [sharedjoin_tmp+0xbf80], sharedjoin_value;
02:-:-:-:6 NOP;
--:-:-:-:6 MOV vid, RZ;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 BRA ffnact_BEGIN;
]=]
rep('POSITION_END:',body..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
