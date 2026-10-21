-- Each PLE gate CTA owns one residual branch. Its warp0 normalizes that same
-- branch after a local rendezvous, preserving the original RMS reduction order.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local maps={}
for _,pair in ipairs({{'plegate','plenormgate'},{'rms2560','plenorm'}})do
 for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(pair[1]..'_',1,true)then maps[#maps+1]=line:gsub(pair[1],pair[2])end
 end
end
local gate=assert(base:match('(plegate_SETUP:.-)\nargmax_SETUP:')):gsub('plegate','plenormgate')
local norm=assert(base:match('(rms2560_BEGIN:.-)\nrms10240_SETUP:')):gsub('rms2560','plenorm')
local offsets={ [0xbf88]=0xbfa0,[0xbf8c]=0xbfa4,[0xbf90]=0xbfb0,[0xbf94]=0xbfb4,
 [0xbf98]=0xbfb8,[0xbf9c]=0xbfbc,[0xbfa0]=0xbfc0 }
norm=norm:gsub('%[RZ%+0x(%x+)%]',function(h)local n=tonumber(h,16);return string.format('[RZ+0x%x]',offsets[n] or n)end)
local at=assert(gate:find('plenormgate_END:',1,true))
gate=gate:sub(1,at-1)..[=[plenormgate_END:
3f:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, warp, RZ, PT;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA POSITION_END;
]=]..norm
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x69, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6a, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA plenormgate_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',gate..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
