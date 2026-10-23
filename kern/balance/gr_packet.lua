-- Keep the complete GR-read arithmetic and four-row CTA ownership. Collect
-- each die's <=64 outputs locally, then let CTA0 send two contiguous chunks
-- per peer. Other CTAs proceed to the existing phase footer; only CTA0 waits
-- for their packet-ready words. Its ordinary peer release covers the payload.
-- Descriptor words20..21 point to position-private ldown:64 floats +16 flags.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(text,a,b)
 local p,q=text:find(a,1,true);assert(p,a);assert(not text:find(a,q+1,true),'ambiguous '..a)
 return text:sub(1,p-1)..b..text:sub(q+1)
end
local maps=[=[
    16,17 : gp_packet<0-1>
    18,19 : gp_address<0-1>
    20 : gp_byte
    21 : gp_epoch
    22 : gp_ready
    23 : gp_temp
    24,25 : gp_table<0-1>
    26,27 : gp_peer<0-1>
    28 : gp_rows
    29 : gp_offset
    30 : gp_row
    31 : gp_value
]=]
for _,tag in ipairs({'bf16hcrx','bf16hcrwx','grreadsum'})do
 local first=assert(s:find(tag..'_XCOAL_ROW_STORE:',1,true))
 local last=assert(s:find('\n[%w_]+_SETUP:',first))
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function label(x)out[#out+1]=tag..'_PACK_'..x..':'end
 local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
 out[#out+1]=tag..'_XCOAL_ROW_STORE:'
 -- P2 is native valid-row AND active-warp AND lane0. Dot association and
 -- residual/RMS instructions above this point are unchanged.
 e('NOP','3f:-:-:Y:d')
 e('LDS.64 gp_packet0, [RZ+0xbfd0]','--:-:1:-:2')
 e('LDS gp_epoch, [RZ+0xbf74]','--:-:2:-:2')
 e('SHL gp_byte, '..tag..'_row, 0x2')
 e('IADD gp_address0.CC, gp_packet0, gp_byte','01:-:-:-:6')
 e('IADD.X gp_address1, gp_packet1, RZ')
 e('@P2 STG.E [gp_address], '..tag..'_ysum','--:3:-:-:2')
 e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
 e('IADD gp_epoch, gp_epoch, 0x1')
 e('SHL gp_byte, cta, 0x2');e('IADD gp_byte, gp_byte, 0x100')
 e('IADD gp_address0.CC, gp_packet0, gp_byte');e('IADD.X gp_address1, gp_packet1, RZ')
 pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
 e('@P5 STG.E.WT [gp_address], gp_epoch','--:3:-:-:2');e('NOP','04:-:-:-:6')
 pred('ISETP.NE.U32.AND P6, PT, cta, RZ, PT')
 e('@P6 BRA POSITION_END','--:-:-:Y:d')
 pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT')
 e('@P6 BRA '..tag..'_PACK_JOIN','--:-:-:Y:d')
 e('LOP32I.AND gp_byte, lane, 0xf');e('SHL gp_byte, gp_byte, 0x2')
 e('IADD gp_byte, gp_byte, 0x100')
 e('IADD gp_address0.CC, gp_packet0, gp_byte');e('IADD.X gp_address1, gp_packet1, RZ')
 label('WAIT')
 e('LDG.E.CV gp_ready, [gp_address]','--:-:1:-:2');e('NOP','01:-:-:-:6')
 pred('ISETP.GE.U32.AND P5, PT, gp_ready, gp_epoch, PT')
 pred('VOTE.ALL P5, P5');e('@!P5 BRA '..tag..'_PACK_WAIT','--:-:-:Y:d')
 label('JOIN');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
 pred('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT')
 e('@P6 BRA '..tag..'_PACK_DONE','--:-:-:Y:d')
 e('LDS.64 gp_table0, [RZ+0xbf98]','--:-:1:-:2')
 e('LDS gp_rows, [RZ+0xbfa0]','--:-:1:-:2')
 e('LDS gp_offset, [RZ+0xbfb0]','--:-:1:-:2')
 e('SHL gp_byte, warp, 0x3')
 e('IADD gp_table0.CC, gp_table0, gp_byte','01:-:-:-:6');e('IADD.X gp_table1, gp_table1, RZ')
 e('LDG.E.CG.64 gp_peer0, [gp_table]','--:-:1:-:2')
 e('SHL gp_byte, lane, 0x2')
 e('IADD gp_packet0.CC, gp_packet0, gp_byte');e('IADD.X gp_packet1, gp_packet1, RZ')
 e('IADD gp_byte, gp_byte, gp_offset')
 e('IADD gp_peer0.CC, gp_peer0, gp_byte','01:-:-:-:6');e('IADD.X gp_peer1, gp_peer1, RZ')
 e('MOV gp_row, lane')
 for half=0,1 do
  pred('ISETP.LT.U32.AND P5, PT, gp_row, gp_rows, PT')
  e(string.format('@P5 LDG.E.CG gp_value, [gp_packet+0x%x]',half*128),'--:-:1:-:2')
  e(string.format('@P5 STG.E.WT [gp_peer+0x%x], gp_value',half*128),'01:3:-:-:2')
  e('NOP','04:-:-:-:6')
  if half==0 then e('IADD gp_row, gp_row, 0x20')end
 end
 label('DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
 e('BRA POSITION_END','--:-:-:Y:d')
 s=s:sub(1,first-1)..table.concat(out,'\n')..'\n'..s:sub(last)
end
s=rep(s,'</REGISTER_MAPPING>',maps..'\n</REGISTER_MAPPING>')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('gr_packet ops=3 rows_max=64 ctas=16 peer_publishers=1 peer_exchanges_added=0\n')
