-- Score CTAs publish completion; the sole selection CTA consumes their scores
-- and first histogram without a second whole-grid phase. Ready words identify
-- request-base+token*16+local-attention-slot, so preceding generations cannot
-- satisfy this acquire when HST scratch is reused. Descriptor/token loads use
-- separate completion barriers because they traverse different pipelines.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local maps={}
for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('idxscore_',1,true)then maps[#maps+1]=line:gsub('idxscore','scoreselect')end
end
maps[#maps+1]=[=[    16,17 : scorejoin_ptr<0-1>
    18 : scorejoin_value
    19 : scorejoin_tmp
    20 : scorejoin_token
    21 : scorejoin_epoch
]=]
local body=assert(base:match('(idxscore_SETUP:.-)\nidxsel_SETUP:')):gsub('idxscore','scoreselect')
local at=assert(body:find('scoreselect_END:',1,true))
body=body:sub(1,at-1)..[=[scoreselect_END:
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:1:-:1 LDS.64 scorejoin_ptr0, [RZ+0xbfa0];
--:-:2:-:1 LDS scorejoin_epoch, [RZ+0xbfa8];
--:-:3:-:2 LDS scorejoin_token, [RZ+0xbfb0];
06:-:-:-:6 SHL scorejoin_epoch, scorejoin_epoch, 0x4;
--:-:-:-:6 IADD scorejoin_epoch, scorejoin_epoch, scorejoin_token;
--:-:-:-:6 SHL scorejoin_tmp, cta, 0x2;
--:-:-:-:6 IADD scorejoin_tmp, scorejoin_tmp, 0x1200;
01:-:-:-:6 IADD scorejoin_ptr0.CC, scorejoin_ptr0, scorejoin_tmp;
--:-:-:-:6 IADD.X scorejoin_ptr1, scorejoin_ptr1, RZ;
--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, tid, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:2:-:-:2 @P5 STG.E.WT [scorejoin_ptr], scorejoin_epoch;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, cta, RZ, PT;
02:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA POSITION_END;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, warp, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA SCORE_JOIN_READY;
--:-:1:-:2 LDS.64 scorejoin_ptr0, [RZ+0xbfa0];
--:-:-:-:6 LOP32I.AND scorejoin_tmp, lane, 0xf;
--:-:-:-:6 SHL scorejoin_tmp, scorejoin_tmp, 0x2;
--:-:-:-:6 IADD scorejoin_tmp, scorejoin_tmp, 0x1200;
01:-:-:-:6 IADD scorejoin_ptr0.CC, scorejoin_ptr0, scorejoin_tmp;
--:-:-:-:6 IADD.X scorejoin_ptr1, scorejoin_ptr1, RZ;
SCORE_JOIN_WAIT:
--:-:1:-:2 LDG.E.CV scorejoin_value, [scorejoin_ptr];
01:-:-:-:6 ISETP.EQ.U32.AND P5, PT, scorejoin_value, scorejoin_epoch, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 VOTE.ALL P5, P5;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P5 BRA SCORE_JOIN_WAIT;
--:-:-:-:6 IADD scorejoin_tmp, pidx, 0x1;
--:-:-:-:6 SHL scorejoin_tmp, scorejoin_tmp, 0x7;
--:-:-:-:6 IADD scorejoin_ptr0.CC, prog0, scorejoin_tmp;
--:-:-:-:6 IADD.X scorejoin_ptr1, prog1, RZ;
--:-:-:-:6 SHL scorejoin_tmp, lane, 0x2;
--:-:-:-:6 IADD scorejoin_ptr0.CC, scorejoin_ptr0, scorejoin_tmp;
--:-:-:-:6 IADD.X scorejoin_ptr1, scorejoin_ptr1, RZ;
--:-:1:-:1 LDG.E scorejoin_value, [scorejoin_ptr];
--:-:2:-:2 LDS scorejoin_token, [RZ+0xbf70];
--:-:-:-:6 ISETP.EQ.U32.AND P4, PT, lane, 0x8, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
03:-:-:-:6 NOP;
--:-:-:-:6 @P4 IADD scorejoin_value, scorejoin_value, scorejoin_token;
--:2:-:-:6 STS [scorejoin_tmp+0xbf80], scorejoin_value;
02:-:-:-:6 NOP;
SCORE_JOIN_READY:
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 BRA selectone_SETUP;
]=]
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x67, PT;','ISETP.GE.U32.AND P6, PT, op, 0x68, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA scoreselect_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
