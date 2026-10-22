-- Sum each six-peer MoE vector once per CTA, in the original FP32 order.
-- The four residual branches and repeated RMS passes reuse that private cache.
-- This preserves the peer acquire and introduces no cross-CTA readiness wait.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]))
local body=assert(base:match('(bf16hcrwx_SETUP:.-)\nq4kk2560_SETUP:')):gsub('bf16hcrwx','grreadsum')
local function rep(x,a,b)
 local i=assert(x:find(a,1,true),'missing '..a)
 assert(not x:find(a,i+#a,true),'ambiguous '..a)
 return x:sub(1,i-1)..b..x:sub(i+#a)
end
local prelude=[=[
--:-:-:-:6 SHL grcache_offset, tid, 0x4;
--:-:-:-:6 XMAD grcache_block, cta, 0x2800, RZ;
GRCACHE_ROW:
--:-:-:-:6 ISETP.GE.U32.AND P6, PT, grcache_offset, 0x2800, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA GRCACHE_DONE;
--:-:1:-:1 LDS.64 grcache_src0, [RZ+0xbfe8];
--:-:1:-:2 LDS.64 grcache_dst0, [RZ+0xbfc8];
01:-:-:-:6 IADD grcache_src0.CC, grcache_src0, grcache_offset;
--:-:-:-:6 IADD.X grcache_src1, grcache_src1, RZ;
--:-:-:-:6 IADD grcache_dst0.CC, grcache_dst0, grcache_block;
--:-:-:-:6 IADD.X grcache_dst1, grcache_dst1, RZ;
--:-:-:-:6 IADD grcache_dst0.CC, grcache_dst0, grcache_offset;
--:-:-:-:6 IADD.X grcache_dst1, grcache_dst1, RZ;
--:-:-:-:6 MOV grcache_sum0, RZ;
--:-:-:-:6 MOV grcache_sum1, RZ;
--:-:-:-:6 MOV grcache_sum2, RZ;
--:-:-:-:6 MOV grcache_sum3, RZ;
--:-:-:-:6 MOV grcache_peer, RZ;
GRCACHE_PEER:
--:-:1:-:2 LDG.E.CI.128 grcache_tmp0, [grcache_src];
01:-:-:-:6 FADD grcache_sum0, grcache_sum0, grcache_tmp0;
--:-:-:-:6 FADD grcache_sum1, grcache_sum1, grcache_tmp1;
--:-:-:-:6 FADD grcache_sum2, grcache_sum2, grcache_tmp2;
--:-:-:-:6 FADD grcache_sum3, grcache_sum3, grcache_tmp3;
--:-:-:-:6 IADD grcache_src0.CC, grcache_src0, 0x2800;
--:-:-:-:6 IADD.X grcache_src1, grcache_src1, RZ;
--:-:-:-:6 IADD grcache_peer, grcache_peer, 0x1;
--:-:-:-:6 ISETP.LT.U32.AND P6, PT, grcache_peer, 0x6, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA GRCACHE_PEER;
--:2:-:-:2 STG.E.128 [grcache_dst], grcache_sum0;
02:-:-:-:6 IADD grcache_offset, grcache_offset, 0x1000;
--:-:-:-:6 BRA GRCACHE_ROW;
GRCACHE_DONE:
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 MEMBAR.GL.IVALLD;
]=]
body=rep(body,'grreadsum_BEGIN:','grreadsum_BEGIN:\n'..prelude)
body=rep(body,'LDS grreadsum_param_YG0, [RZ+0xbfe8];','LDS grreadsum_param_YG0, [RZ+0xbfc8];')
body=rep(body,'LDS grreadsum_param_YG1, [RZ+0xbfec];','LDS grreadsum_param_YG1, [RZ+0xbfcc];')
body=rep(body,'--:-:-:-:6      MOV grreadsum_tid, vtid;', [=[--:-:-:-:6 XMAD grreadsum_tmp0, cta, 0x2800, RZ;
--:-:-:-:6 IADD grreadsum_param_YG0.CC, grreadsum_param_YG0, grreadsum_tmp0;
--:-:-:-:6 IADD.X grreadsum_param_YG1, grreadsum_param_YG1, RZ;
--:-:-:-:6      MOV grreadsum_tid, vtid;]=])
local first=assert(s:find('grreadsum_SETUP:',1,true))
local last=assert(s:find('\nPOSITION_END:',first,true))
s=s:sub(1,first-1)..body..s:sub(last)
s=rep(s,'</REGISTER_MAPPING>',[=[    16,17 : grcache_src<0-1>
    18,19 : grcache_dst<0-1>
    20-23 : grcache_sum<0-3>
    24-27 : grcache_tmp<0-3>
    28 : grcache_offset
    29 : grcache_block
    31 : grcache_peer
</REGISTER_MAPPING>]=])
local f=assert(io.open(assert(arg[3]),'w'));f:write(s);f:close()
