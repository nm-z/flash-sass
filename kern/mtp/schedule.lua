-- Scheduling corrections for the scalar dispatcher embedded in the MTP image.
-- No mathematical operation, descriptor ABI or opcode ordering changes.
return function(s)
 local function once(a,b)
  local i,j=assert(s:find(a,1,true),'missing schedule anchor '..a);j=i+#a-1
  assert(not s:find(a,j+1,true),'duplicate schedule anchor '..a)
  s=s:sub(1,i-1)..b..s:sub(j+1)
 end
 once('--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, cta, RZ, P4;',
      '--:-:-:Y:f      ISETP.EQ.U32.AND P2, PT, cta, RZ, P4;')
 once('20:-:-:-:6  @P4 STG.E [bar_cp+0x100], bar_g2;',
      '20:2:-:-:6  @P4 STG.E [bar_cp+0x100], bar_g2;')
 once('01:-:-:-:6      IADD pidx, pidx, bar_t;',
      '03:-:-:-:6      IADD pidx, pidx, bar_t;')
 for _,offset in ipairs{'8','20','4'} do
  once('LDG.E op, [fw+0x'..offset..'];','LDG.E.CG op, [fw+0x'..offset..'];')
 end
 return assert(loadfile('mtp/prefetch.lua'))()(s)
end
