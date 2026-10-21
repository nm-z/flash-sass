-- Distributed router projection with a single consuming warp. Producers signal
-- row completion and proceed to the final phase barrier; only CTA0 warp0 waits
-- for logits. There is no intervening whole-grid rendezvous or second dispatch.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
local maps={}
for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('bf16k2560_',1,true)then maps[#maps+1]=line:gsub('bf16k2560','routerproj')end
end
maps[#maps+1]=[=[    16,17 : routejoin_ptr<0-1>
    18 : routejoin_epoch
    19 : routejoin_layer
    20 : routejoin_tmp
    21 : routejoin_value
    22 : routejoin_mask
]=]
local body=assert(base:match('(bf16k2560_SETUP:.-)\nbf16k320_SETUP:')):gsub('bf16k2560','routerproj')
local at=assert(body:find('routerproj_END:',1,true))
body=body:sub(1,at-1)..[=[routerproj_END:
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:1:-:1 LDS.64 routejoin_ptr0, [RZ+0xbf98];
--:-:1:-:1 LDS routejoin_layer, [RZ+0xbfa8];
--:-:1:-:2 LDS routejoin_epoch, [RZ+0xbf70];
01:-:-:-:6 XMAD routejoin_epoch, routejoin_epoch, 0x31, routejoin_layer;
--:-:-:-:6 ISCADD routejoin_tmp, cta, RZ, 0x2;
--:-:-:-:6 IADD routejoin_tmp, routejoin_tmp, 0x804;
--:-:-:-:6 IADD routejoin_ptr0.CC, routejoin_ptr0, routejoin_tmp;
--:-:-:-:6 IADD.X routejoin_ptr1, routejoin_ptr1, RZ;
--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, tid, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:2:-:-:2 @P5 STG.E.WT [routejoin_ptr], routejoin_epoch;
--:-:-:-:6 IADD routejoin_tmp, cta, warp;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, routejoin_tmp, RZ, PT;
02:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA POSITION_END;
--:-:1:-:2 LDS.64 routejoin_ptr0, [RZ+0xbf98];
--:-:-:-:6 LOP32I.AND routejoin_tmp, lane, 0xf;
--:-:-:-:6 SHL routejoin_tmp, routejoin_tmp, 0x2;
--:-:-:-:6 IADD routejoin_tmp, routejoin_tmp, 0x804;
01:-:-:-:6 IADD routejoin_ptr0.CC, routejoin_ptr0, routejoin_tmp;
--:-:-:-:6 IADD.X routejoin_ptr1, routejoin_ptr1, RZ;
ROUTE_JOIN:
--:-:1:-:2 LDG.E.CV routejoin_value, [routejoin_ptr];
01:-:-:-:6 ISETP.EQ.U32.AND P5, PT, routejoin_value, routejoin_epoch, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 VOTE.ALL P5, P5;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P5 BRA ROUTE_JOIN;
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 IADD routejoin_tmp, pidx, 0x1;
--:-:-:-:6 SHL routejoin_tmp, routejoin_tmp, 0x7;
--:-:-:-:6 IADD routejoin_ptr0.CC, prog0, routejoin_tmp;
--:-:-:-:6 IADD.X routejoin_ptr1, prog1, RZ;
--:-:-:-:6 SHL routejoin_tmp, lane, 0x2;
--:-:-:-:6 IADD routejoin_ptr0.CC, routejoin_ptr0, routejoin_tmp;
--:-:-:-:6 IADD.X routejoin_ptr1, routejoin_ptr1, RZ;
--:-:1:-:2 LDG.E routejoin_value, [routejoin_ptr];
01:2:-:-:6 STS [routejoin_tmp+0xbf80], routejoin_value;
02:-:-:-:6 NOP;
--:-:-:-:6 MOV vid, RZ;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 BRA routesel_BEGIN;
]=]
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x64, PT;','ISETP.GE.U32.AND P6, PT, op, 0x65, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA routerproj_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
