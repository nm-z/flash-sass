-- Preserve the producer's CTA-owned rows, but assign one peer to each of six
-- copying warps. The original publication serializes six destination loads
-- and stores in every active warp. All row tiles and all six peers remain;
-- source rereads are local and the existing CTA fence/handoff is preserved.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local tags={'q3kk6144','q4kk6144','q5kk6144','iq4k6144','q2moedn','q2acc640',
 'bf16k2560','q2k2560','q3kk2560','q4kk2560','q5kk2560','q6kk2560','iq4k2560',
 'q6kk1536acc','b32q40acc','b32q50acc','b32q80acc','b32iq4nlacc'}
for _,tag in ipairs(tags)do
 local region=assert(s:find('\n'..tag..'_END:',1,true))
 local first=assert(s:find('3f:-:-:-:6 NOP;\n--:-:1:-:2 LDS pub_flags, [RZ+0xbff8];',region,true))
 local last=assert(s:find('\n'..tag..'_SEND_DONE:',first,true))
 local old=s:sub(first,last-1)
 assert(old:find(tag..'_SEND_PEER:',1,true),'unexpected publisher '..tag)
 local shift=tonumber(assert(old:match('SHL pub_base, cta, 0x(%x+);')),16)
 assert(shift>=1 and shift<=6,'unsupported producer tile: '..tag)
 local rows=assert(old:match('LDS pub_rows, %[RZ%+0x(%x+)%]'))
 local src=assert(old:match('LDS%.64 pub_src0, %[RZ%+0x(%x+)%]'))
 local tab=assert(old:match('LDS%.64 pub_tab0, %[RZ%+0x(%x+)%]'))
 local offset=assert(old:match('LDS pub_offset, %[RZ%+0x(%x+)%]'))
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:Y:d')..' '..x..';'end
 e('NOP','3f:-:-:Y:d')
 e('LDS pub_flags, [RZ+0xbff8]','--:-:1:-:2')
 e('LOP32I.AND pub_flags, pub_flags, 0x80','01:-:-:Y:d')
 e('ISETP.EQ.U32.AND P6, PT, pub_flags, RZ, PT');e('@P6 BRA '..tag..'_SEND_DONE')
 e('MEMBAR.GL');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
 e('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT');e('@P6 BRA '..tag..'_SEND_DONE')
 e('LDS pub_rows, [RZ+0x'..rows..']','--:-:1:-:2')
 e('LDS pub_offset, [RZ+0x'..offset..']','--:-:2:-:2')
 e('LDS.64 pub_src0, [RZ+0x'..src..']','--:-:3:-:2')
 e('LDS.64 pub_tab0, [RZ+0x'..tab..']','--:-:4:-:2')
 -- For tiles no wider than32 rows, a warp covers32/R CTA-owned tiles.
 -- Advancing512 global rows advances32 owned rows on every CTA. Wider
 -- tiles use the two-half traversal below.
 e(string.format('SHL pub_base, cta, 0x%x',shift))
 if shift==6 then
  -- A64-row producer tile takes two32-lane copies. Alternate +32 and
  -- +(1024-32) rows to visit both halves before the next owned tile.
  e('IADD pub_row, pub_base, lane');e('MOV pub_step, 0x20')
 else
  e(string.format('SHR.U32 pub_tmp, lane, 0x%x',shift))
  e(string.format('SHL pub_row, pub_tmp, 0x%x',shift+4))
  e('IADD pub_row, pub_row, pub_base')
  e(string.format('LOP32I.AND pub_tmp, lane, 0x%x',(1<<shift)-1))
  e('IADD pub_row, pub_row, pub_tmp')
 end
 e('SHL pub_byte, pub_row, 0x2')
 e('SHL pub_peer, warp, 0x3')
 e('IADD pub_tab0.CC, pub_tab0, pub_peer','0f:-:-:Y:d')
 e('IADD.X pub_tab1, pub_tab1, RZ')
 e('LDG.E.CG.64 pub_dst0, [pub_tab]','--:-:1:-:2')
 e('IADD pub_src0.CC, pub_src0, pub_byte');e('IADD.X pub_src1, pub_src1, RZ')
 e('IADD pub_byte, pub_byte, pub_offset')
 e('IADD pub_dst0.CC, pub_dst0, pub_byte','01:-:-:Y:d');e('IADD.X pub_dst1, pub_dst1, RZ')
 out[#out+1]=tag..'_SEND_ROW:'
 e('ISETP.GE.U32.AND P6, PT, pub_base, pub_rows, PT');e('@P6 BRA '..tag..'_SEND_DONE')
 e('ISETP.LT.U32.AND P2, PT, pub_row, pub_rows, PT')
 e('@P2 LDG.E.CG pub_value, [pub_src]','--:-:1:-:2')
 e('@P2 STG.E.WT [pub_dst], pub_value','01:2:-:-:2')
 if shift==6 then
  e('SHL pub_tmp, pub_step, 0x2','02:-:-:Y:d')
  e('IADD pub_src0.CC, pub_src0, pub_tmp');e('IADD.X pub_src1, pub_src1, RZ')
  e('IADD pub_dst0.CC, pub_dst0, pub_tmp');e('IADD.X pub_dst1, pub_dst1, RZ')
  e('IADD pub_row, pub_row, pub_step');e('IADD pub_base, pub_base, pub_step')
  e('MOV pub_tmp, 0x400');e('IADD pub_step, pub_tmp, -pub_step')
 else
  e('IADD pub_src0.CC, pub_src0, 0x800','02:-:-:Y:d');e('IADD.X pub_src1, pub_src1, RZ')
  e('IADD pub_dst0.CC, pub_dst0, 0x800');e('IADD.X pub_dst1, pub_dst1, RZ')
  e('IADD pub_row, pub_row, 0x200');e('IADD pub_base, pub_base, 0x200')
 end
 e('BRA '..tag..'_SEND_ROW')
 s=s:sub(1,first-1)..table.concat(out,'\n')..s:sub(last)
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
