local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local maps={};local quant=dofile('phase/quant_emit.lua')(maps)
for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('attnmerge_',1,true)then maps[#maps+1]=line:gsub('attnmerge','attnquant')end
end
local body=assert(base:match('(attnmerge_SETUP:.-)\nplerows_SETUP:')):gsub('attnmerge','attnquant')
local anchor='01:-:-:-:f      BRA attnquant_END;'
local at=assert(body:find(anchor,1,true))
local setup=[=[--:-:-:-:6 MOV %_v0, attnquant_o;
--:-:-:-:6 ISCADD %_index, vid, tid, 0x8;]=]
body=body:sub(1,at-1)..quant('attnq',1,0xbfb0,setup,'--:-:-:-:6 BRA attnquant_END;\n')..body:sub(at+#anchor)
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x65, PT;','ISETP.GE.U32.AND P6, PT, op, 0x66, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA attnquant_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
