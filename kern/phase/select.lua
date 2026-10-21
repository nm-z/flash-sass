-- One CTA owns QSA selection. Reuse the original radix/tie-order arithmetic;
-- local histogram completion replaces four grid phase boundaries. The cross-CTA
-- count pass is unnecessary because there are no preceding CTA partitions.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local maps={}
for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('idxsel_',1,true)then maps[#maps+1]=line:gsub('idxsel','selectone')end
end
local body=assert(base:match('(idxsel_SETUP:.-)\nxcopy_SETUP:')):gsub('idxsel','selectone')
local function replace(x,a,b)local i=assert(x:find(a,1,true),'missing '..a);return x:sub(1,i-1)..b..x:sub(i+#a)end
body=replace(body,'selectone_BEGIN:',[=[
--:-:-:-:6 MOV32I selectone_t, 0x2;
--:2:-:-:6 STS [RZ+0x480], selectone_t;
02:-:-:-:6 BAR.SYNC 0x0;
selectone_BEGIN:]=])
body=replace(body,'LDS selectone_param_pass, [RZ+0xbfa8];','LDS selectone_param_pass, [RZ+0x480];')
body=replace(body,'--:-:-:-:6      IADD selectone_t, selectone_nv, 0xf;\n--:-:-:-:6      SHR.U32 selectone_t, selectone_t, 0x4;', '--:-:-:-:6 MOV selectone_t, selectone_nv;')
local at=assert(body:find('selectone_END:',1,true))
body=body:sub(1,at-1)..[=[selectone_END:
3f:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 ISETP.LE.U32.AND P5, PT, selectone_nv, selectone_param_nblk, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 ISETP.EQ.U32.OR P5, PT, selectone_param_pass, 0x6, P5;
--:-:-:-:6 IADD selectone_t, selectone_param_pass, 0x1;
--:-:-:-:6 ISETP.EQ.U32.AND P4, PT, selectone_t, 0x5, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 @P5 BRA POSITION_END;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 MOV32I selectone_t, 0x6;
--:2:-:-:6 STS [RZ+0x480], selectone_t;
02:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 BRA selectone_BEGIN;
]=]
local function rep(a,b)s=replace(s,a,b)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x66, PT;','ISETP.GE.U32.AND P6, PT, op, 0x67, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA selectone_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
