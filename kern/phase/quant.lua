-- Quantize where the producer has its FP32 result in registers/shared memory,
-- then publish codes and scales directly to the six receiving dies.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local source=read(assert(arg[1]));local s=read(assert(arg[2]));local output=assert(arg[3])
local maps,bodies={},{}
local function clone(tag,nexttag,newtag)
 local map={}
 for line in assert(source:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(tag..'_',1,true)then map[#map+1]=line:gsub(tag,newtag)end
 end
 maps[#maps+1]=table.concat(map,'\n')
 return assert(source:match('('..tag..'_SETUP:.-)\n'..nexttag..'_SETUP:')):gsub(tag,newtag)
end
local quant=dofile("phase/quant_emit.lua")(maps)
local dn=clone('dnnormx','router','dnquant')
dn=dn:gsub('[^\n]*LDG%.E%.64 dnquant_dq%d+, %[dnquant_yp%+0x%x+%];\n','')
local a=assert(dn:find('04:-:-:-:6      IADD dnquant_dq00.CC',1,true))
local b=assert(dn:find('dnquant_END:',a,true))
local setup=[=[--:-:-:-:6 MOV %_v0, dnquant_x0;
--:-:-:-:6 MOV %_v1, dnquant_x1;
--:-:-:-:6 MOV %_v2, dnquant_x2;
--:-:-:-:6 MOV %_v3, dnquant_x3;
--:-:-:-:6 LOP32I.AND %_index, dnquant_xo, 0xffff;
--:-:-:-:6 SHR.U32 %_index, %_index, 0x2;]=]
dn=dn:sub(1,a-1)..quant('dnq',4,0xbfa0,setup,'--:-:-:-:6 BRA dnquant_END;\n')..dn:sub(b)
bodies[#bodies+1]=dn
local hc=clone('bf16hcup','bf16hc','hcquant')
local old='--:-:-:-:6      IADD hcquant_xl, hcquant_gg, 0xf;\n--:-:-:-:6      SHR.U32 hcquant_xl, hcquant_xl, 0x4;'
local i,j=assert(hc:find(old,1,true));j=i+#old-1;hc=hc:sub(1,i-1)..'--:-:-:-:6      MOV32I hcquant_xl, 0x20;'..hc:sub(j+1)
local anchor='--:-:-:-:f      BRA hcquant_END;\nhcquant_END:'
local setup_hc=[=[--:-:-:-:6 ISETP.GE.U32.AND P4, PT, tid, 0x20, PT;
--:-:1:-:1 LDS %_cols, [RZ+0xbfa0];
--:-:1:-:2 LDS %_index, [RZ+0xbfb0];
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 BRA hcquant_END;
01:-:-:-:6 SHR.U32 %_cols, %_cols, 0x2;
--:-:-:-:6 SHL %_temp, cta, 0x5;
--:-:-:-:6 ISETP.GE.U32.AND P4, PT, %_temp, %_cols, PT;
--:-:-:-:6 LOP32I.AND %_index, %_index, 0xffff;
--:-:-:-:6 SHR.U32 %_index, %_index, 0x2;
--:-:-:-:6 IADD %_index, %_index, %_temp;
--:-:-:-:6 @P4 BRA hcquant_END;
--:-:-:-:6 IADD %_index, %_index, tid;
--:-:-:-:6 SHL %_temp, tid, 0x2;
--:-:1:-:2 LDS %_v0, [%_temp+0x600];
01:-:-:-:6 NOP;]=]
i,j=assert(hc:find(anchor,1,true));j=i+#anchor-1
hc=hc:sub(1,i-1)..quant('hcq',1,0xbfd8,setup_hc,'--:-:-:-:6 BRA hcquant_END;\nhcquant_END:')..hc:sub(j+1)
bodies[#bodies+1]=hc
local function once(a,b)local i,j=assert(s:find(a,1,true),'missing '..a);j=i+#a-1;assert(not s:find(a,j+1,true),'duplicate '..a);s=s:sub(1,i-1)..b..s:sub(j+1)end
local count=assert(tonumber(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;'),16));assert(count==95)
once('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
once('ISETP.GE.U32.AND P6, PT, op, 0x5f, PT;','ISETP.GE.U32.AND P6, PT, op, 0x61, PT;')
once('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6      BRA dnquant_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA hcquant_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
once('POSITION_END:',table.concat(bodies,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'wb'));f:write(s);f:close()
