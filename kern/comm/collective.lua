-- Coalesce each die's sixteen publication flags into one half-warp store per
-- destination, after all producer CTAs have fenced their data and arrived.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2]);local publication=read(assert(arg[3]))
s=assert(loadfile('mtp/schedule.lua'))()(s)
local function once(a,b)
 local i,j=assert(s:find(a,1,true),'missing collective anchor '..a);j=i+#a-1
 assert(not s:find(a,j+1,true),'duplicate collective anchor '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local a=assert(s:find('# a send phase (descriptor word 30 bit 0):',1,true))
local b=assert(s:find('BSPIN:\n',a,true))
s=s:sub(1,a-1)..s:sub(b)
once('PHASE_END:\n',[=[PHASE_END:
# Every actual producer fences its own writes before publishing a local arrival.
--:-:1:-:2      LDS bar_old, [RZ+0xbf58];
01:-:-:-:6      LOP32I.AND bar_old, bar_old, 0x1;
--:-:-:-:6      ISETP.NE.U32.AND P2, PT, bar_old, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P2 MEMBAR.SYS;
]=])
once('--:2:-:-:6  @P4 STG.E [bar_cp+0x4], bar_g;\n--:-:-:-:6      BRA BDONE;',
     '--:2:-:-:6  @P4 STG.E [bar_cp+0x4], bar_g;\n'..publication..'\n--:-:-:-:6      BRA BDONE;')
local f=assert(io.open(output,'wb'));assert(f:write(s));assert(f:close())
