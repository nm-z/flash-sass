-- Keep the gather arithmetic in warp 0, then hand the gathered activation to
-- all eight warps of the same CTA for quantization. No grid phase is added.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3]);local maps={}
for _,pair in ipairs({{'plerows','plegatherq'},{'actq256','plequant'}})do
 for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(pair[1]..'_',1,true)then maps[#maps+1]=line:gsub(pair[1],pair[2])end
 end
end
local gather=assert(base:match('(plerows_SETUP:.-)\npleconv_SETUP:')):gsub('plerows','plegatherq')
local start=assert(gather:find('plegatherq_BEGIN:',1,true))
gather=[=[plegatherq_SETUP:
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, cta, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA POSITION_END;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 MOV vid, RZ;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, warp, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA PLEGATHER_QUANT_JOIN;
]=]..gather:sub(start)
local quant=assert(base:match('(actq256_BEGIN:.-)\nrms2560_SETUP:')):gsub('actq256','plequant')
local offsets={[0xbf88]=0xbfa0,[0xbf8c]=0xbfa4,[0xbf90]=0xbfd0,[0xbf94]=0xbfd4,
 [0xbf98]=0xbfd8,[0xbf9c]=0xbfdc,[0xbfa0]=0xbfe0,[0xbfa4]=0xbfe4,[0xbfa8]=0xbfe8,[0xbfac]=0xbfec,[0xbfb0]=0xbff0}
quant=quant:gsub('%[RZ%+0x(%x+)%]',function(h)local x=tonumber(h,16);return string.format('[RZ+0x%x]',offsets[x] or x)end)
quant=quant:gsub('IADD vid, vid, 0x80;','IADD vid, vid, 0x8;')
local at=assert(gather:find('plegatherq_END:',1,true))
gather=gather:sub(1,at-1)..[=[plegatherq_END:
PLEGATHER_QUANT_JOIN:
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 MOV vid, warp;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 MOV32I units, 0xa;
]=]..quant
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x6a, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6b, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA plegatherq_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',gather..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
