-- Reuse the existing production Q2 gate/up/SiLU/quantization body, without the
-- experimental framework/barrier changes in die_fused.sass.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local src=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
local maps={}
for line in assert(src:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('ga_',1,true)then maps[#maps+1]=line end
end
local body=assert(src:match('(GA_SETUP:.-)\nPOSITION_END:'))
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('--:-:-:-:6      MOV q2moegn_tid, vtid;', [=[
--:-:1:-:2 LDS ga_flag, [RZ+0xbff8];
01:-:-:-:6 LOP32I.AND ga_flag, ga_flag, 0x20;
--:-:-:-:6 ISETP.NE.U32.AND P4, PT, ga_flag, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 BRA GA_SETUP;
--:-:-:-:6      MOV q2moegn_tid, vtid;]=])
rep('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
