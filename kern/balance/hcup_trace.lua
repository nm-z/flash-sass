-- Diagnostic only: per-CTA stage clocks for the last native HCUP5 phase.
-- Slots0..4 hold local globaltimer.lo values; slot5 holds the real SM id.
-- The five-position descriptor image ends below0x6000 in CTL. The host checks
-- that bound before readback. None of these clocks are subtracted across dies.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function insert(anchor,code)
 local p,q=assert(s:find(anchor,1,true));assert(not s:find(anchor,q+1,true),'ambiguous '..anchor)
 s=s:sub(1,q)..'\n'..code..s:sub(q+1)
end
local function stamp(slot,sm)
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
 e('ISETP.EQ.U32.AND P6, PT, tid, RZ, PT','--:-:-:Y:d');e('NOP')
 e('SHL h5_offset, cta, 0x5')
 e(string.format('IADD h5_offset, h5_offset, 0x%x',0x6000+4*slot))
 e('IADD h5_descriptor0.CC, ctl0, h5_offset');e('IADD.X h5_descriptor1, ctl1, RZ')
 e('@P6 S2R h5_trace, '..(sm and 'SR_VIRTID' or 'SR_GLOBALTIMERLO'),'--:-:6:-:2')
 if sm then e('@P6 BFE.U32 h5_trace, h5_trace, 0x914','20:-:-:-:6')end
 e('@P6 STG.E [h5_descriptor], h5_trace',sm and '--:2:-:-:2' or '20:2:-:-:2');e('NOP','02:-:-:-:6')
 return table.concat(out,'\n')
end
insert('hcup5_SETUP:\n3f:-:-:Y:d NOP;',stamp(5,true)..'\n'..stamp(0))
insert('H5_SILU_DONE:\n3f:-:-:Y:d BAR.SYNC 0x0;',stamp(1))
insert('H5_MATRIX_DONE:\n3f:-:-:Y:d MEMBAR.GL;\n--:-:-:-:6 BAR.SYNC 0x0;\n--:-:-:-:6 MEMBAR.GL.IVALLD;',stamp(2))
insert('H5_PUBLISH_JOIN:\n3f:-:-:Y:d BAR.SYNC 0x0;',stamp(3))
insert('H5_PUBLISH_DONE:\n3f:-:-:Y:d MEMBAR.GL;\n--:-:-:-:6 BAR.SYNC 0x0;\n--:-:-:-:6 MEMBAR.GL.IVALLD;',stamp(4))
local p=assert(s:find('</REGISTER_MAPPING>',1,true))
s=s:sub(1,p-1)..'    247 : h5_trace\n'..s:sub(p)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
