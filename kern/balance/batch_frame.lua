-- Nest complete balanced phase bundles inside one cross-die publication.
-- Native pidx is the real inner primary index, so composite QSA/GDN/router
-- child descriptors retain their exact layout. The wrapper's saved framework
-- state lives below the eight legacy parameter banks, away from bf00 scratch.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function once(old,new)
 local p,q=assert(s:find(old,1,true));assert(not s:find(old,q+1,true),'ambiguous '..old)
 s=s:sub(1,p-1)..new..s:sub(q+1)
end
local function gen()
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function l(x)out[#out+1]=x..':'end
 local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
 return out,e,l,pred
end
local maps=[=[
    180 : bb_active
    181 : bb_parent
    182 : bb_flags
    183 : bb_step
    184 : bb_mode
    185 : bb_cursor
    186 : bb_count
    187 : bb_position
]=]
once('</REGISTER_MAPPING>',maps..'</REGISTER_MAPPING>')
once('ISETP.GE.U32.AND P6, PT, op, 0x6c, PT;',
     'ISETP.GE.U32.AND P6, PT, op, 0x6d, PT;')
once('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',[=[--:-:-:-:6 BRA batch_SETUP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6      BRA POSITION_END;

embrow_SETUP:]=])
once('02:2:-:-:6  @P5 STS [RZ+0xbf60], RZ;', [=[02:2:-:-:6  @P5 STS [RZ+0xbf60], RZ;
02:2:-:-:6 @P5 STS [RZ+0xba00], RZ;]=])
local out,e,l,pred=gen()
l('batch_SETUP')
e('NOP','3f:-:-:Y:d')
for _,x in ipairs({{'bb_flags',0xbf58},{'bb_step',0xbf50},{'bb_mode',0xbf54},
                  {'bb_cursor',0xbf4c},{'bb_count',0xbf88}})do
 e(string.format('LDS %s, [RZ+0x%x]',x[1],x[2]),'--:-:1:-:2')
end
e('MOV bb_parent, pidx','01:-:-:-:6');e('MOV bb_active, 0x1')
pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
for _,x in ipairs({{'bb_active',0xba00},{'bb_parent',0xba04},{'bb_flags',0xba08},
                  {'bb_step',0xba0c},{'bb_mode',0xba10},{'bb_cursor',0xba14},
                  {'bb_count',0xba18},{'RZ',0xba20}})do
 e(string.format('@P5 STS [RZ+0x%x], %s',x[2],x[1]),'--:2:-:-:2')
end
e('BAR.SYNC 0x1','02:-:-:Y:d')
e('IADD pidx, pidx, 0x1')
e('SHL bar_t, pidx, 0x7')
e('IADD dsp0.CC, prog0, bar_t');e('IADD.X dsp1, prog1, RZ')
e('SHL op, lane, 0x2');e('IADD fw0.CC, dsp0, op');e('IADD.X fw1, dsp1, RZ')
pred('ISETP.GE.U32.AND P6, PT, tid, 0x20, PT')
e('@!P6 LDG.E.CG vtid, [fw]','--:3:4:-:2')
e('@!P6 LDG.E.CG vid, [dsp+0x7c]','--:3:5:-:2')
e('BRA PHASE','--:-:-:Y:d')
once('POSITION_END:',table.concat(out,'\n')..'\nPOSITION_END:')
-- After an inner footer, the ordinary next-phase path advances by its own
-- descriptor span. At the last position, restore the wrapper and execute its
-- real release/publication footer. No speculative token is committed here.
out,e,l,pred=gen()
e('LDS bb_active, [RZ+0xba00]','--:-:1:-:2')
e('NOP','01:-:-:-:6')
pred('ISETP.EQ.U32.AND P5, PT, bb_active, RZ, PT')
e('@P5 BRA BATCH_NEXT_DONE','--:-:-:Y:d')
e('LDS bb_position, [RZ+0xba20]','--:-:1:-:2')
e('LDS bb_count, [RZ+0xba18]','--:-:1:-:2')
e('BAR.SYNC 0x1','01:-:-:Y:d') -- all warps sample the old position
e('IADD bb_position, bb_position, 0x1')
pred('ISETP.GE.U32.AND P6, PT, bb_position, bb_count, PT')
e('@P6 BRA BATCH_RESTORE','--:-:-:Y:d')
pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
e('@P5 STS [RZ+0xba20], bb_position','--:2:-:-:2')
e('BAR.SYNC 0x1','02:-:-:Y:d')
e('BRA BATCH_NEXT_DONE','--:-:-:Y:d')
l('BATCH_RESTORE')
e('NOP','3f:-:-:Y:d') -- retire the inner footer's unused descriptor prefetch
for _,x in ipairs({{'bb_parent',0xba04},{'bb_flags',0xba08},{'bb_step',0xba0c},
                  {'bb_mode',0xba10},{'bb_cursor',0xba14}})do
 e(string.format('LDS %s, [RZ+0x%x]',x[1],x[2]),'--:-:1:-:2')
end
e('MOV pidx, bb_parent','01:-:-:-:6')
pred('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
for _,x in ipairs({{'RZ',0xba00},{'bb_flags',0xbf58},{'bb_step',0xbf50},
                  {'bb_mode',0xbf54},{'bb_cursor',0xbf4c}})do
 e(string.format('@P5 STS [RZ+0x%x], %s',x[2],x[1]),'--:2:-:-:2')
end
e('BAR.SYNC 0x1','02:-:-:Y:d');e('BRA PHASE_END','--:-:-:Y:d')
l('BATCH_NEXT_DONE')
once('# ---- next phase ----',table.concat(out,'\n')..'\n# ---- next phase ----')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('balanced_batch_frame opcode=108 context=ba00..ba20 nested_native_phases=1\n')
