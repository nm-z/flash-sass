-- Hierarchical six-die publication. Every producer CTA fences its payload and
-- records a local arrival. CTA0 publishes one epoch per destination after all
-- sixteen arrivals. Consumers poll six sender epochs instead of96 CTA epochs.
-- Layout: flags[0..5] peer epochs; flags[64..79] local producer arrivals.
-- The ordinary local grid barrier and program/descriptor lifetime remain.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(old,new)
 local a,b=assert(s:find(old,1,true));assert(not s:find(old,b+1,true),'ambiguous '..old)
 s=s:sub(1,a-1)..new..s:sub(b+1)
end
rep('</REGISTER_MAPPING>',[=[    232,233 : uc_self<0-1>
    234,235 : uc_ptr<0-1>
    236,237 : uc_dest<0-1>
    238,239 : uc_control<0-1>
    240 : uc_epoch
    241 : uc_offset
    242 : uc_value
    243 : uc_flags
    244 : uc_timer
    245 : uc_start
</REGISTER_MAPPING>]=])
local begin=assert(s:find('# ---- exchange wait (descriptor word 30 bit 1)',1,true))
local finish=assert(s:find('\nXWDONE:',begin,true))
local wait=s:sub(begin,finish-1)
local old='--:-:-:-:6      SHL bar_old, lane, 0x2;'
local p,q=assert(wait:find(old,1,true))
wait=wait:sub(1,p-1)..[=[--:-:-:-:6 IMNMX.U32 bar_old, lane, 0x5, PT;
--:-:-:-:6 SHL bar_old, bar_old, 0x2;]=]..wait:sub(q+1)
p=assert(wait:find('XWSPIN:',1,true))
q=assert(wait:find('--:-:-:-:6      ISETP.GE.U32.AND P4, PT, bar_t, bar_g, PT;',p,true))
wait=wait:sub(1,p-1)..[=[XWSPIN:
--:-:3:-:2 LDG.E.CV bar_t, [xe];
04:-:-:-:6 NOP;
]=]..wait:sub(q)
s=s:sub(1,begin-1)..wait..s:sub(finish)
begin=assert(s:find('# a send phase (descriptor word 30 bit 0)',1,true))
finish=assert(s:find('# Drain the publication stores',begin,true))
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function label(x)out[#out+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
e('NOP','27:-:-:Y:d') -- keep descriptor-prefetch scoreboards4/5 independent
e('LDS uc_flags, [RZ+0xbf58]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('LOP32I.AND uc_flags, uc_flags, 0x1')
pred('ISETP.EQ.U32.AND P4, PT, uc_flags, RZ, PT')
e('@P4 BRA UC_PUBLISH_DONE','--:-:-:Y:d')
e('MOV uc_control0, ctl0');e('MOV uc_control1, ctl1')
e('LDS uc_epoch, [RZ+0xbf74]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('IADD uc_epoch, uc_epoch, 0x1')
e('LDG.E.CG.64 uc_self0, [uc_control+0x70]','--:-:1:-:2')
e('SHL uc_offset, cta, 0x2');e('IADD uc_offset, uc_offset, 0x100')
e('IADD uc_ptr0.CC, uc_self0, uc_offset','01:-:-:-:6');e('IADD.X uc_ptr1, uc_self1, RZ')
pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT')
e('@P0 MEMBAR.SYS','--:-:-:Y:d')
e('@P0 STG.E.WT [uc_ptr], uc_epoch','--:2:-:-:2');e('NOP','02:-:-:-:6')
pred('ISETP.EQ.U32.AND P1, PT, cta, RZ, PT')
e('@!P1 BRA UC_STORE_EPOCH','--:-:-:Y:d')
e('LOP32I.AND uc_offset, lane, 0xf');e('SHL uc_offset, uc_offset, 0x2')
e('IADD uc_offset, uc_offset, 0x100')
e('IADD uc_ptr0.CC, uc_self0, uc_offset');e('IADD.X uc_ptr1, uc_self1, RZ')
label('UC_LOCAL_ARRIVALS')
e('LDG.E.CV uc_value, [uc_ptr]','--:-:3:-:2');e('NOP','04:-:-:-:6')
pred('ISETP.GE.U32.AND P4, PT, uc_value, uc_epoch, PT')
pred('VOTE.ALL P4, P4')
e('@P4 BRA UC_PEER_PUBLISH','--:-:-:Y:d')
e('S2R uc_timer, SR_GLOBALTIMERHI','--:-:6:-:2')
e('LDS uc_start, [RZ+0xbf5c]','--:-:1:-:2')
e('IADD uc_timer, uc_timer, -uc_start','21:-:-:-:6')
pred('ISETP.GE.U32.AND P5, PT, uc_timer, 0x46, PT')
pred('VOTE.ANY P5, P5');e('@P5 BRA BEXP','--:-:-:Y:d')
e('BRA UC_LOCAL_ARRIVALS','--:-:-:Y:d')
label('UC_PEER_PUBLISH')
e('MEMBAR.GL.IVALLD')
pred('ISETP.LT.U32.AND P2, PT, lane, 0x6, PT')
e('SHL uc_offset, lane, 0x3');e('IADD uc_offset, uc_offset, 0x40')
e('IADD uc_ptr0.CC, uc_control0, uc_offset');e('IADD.X uc_ptr1, uc_control1, RZ')
e('@P2 LDG.E.CG.64 uc_dest0, [uc_ptr]','--:-:1:-:2')
e('@P2 STG.E.WT [uc_dest], uc_epoch','01:2:-:-:2');e('NOP','02:-:-:-:6')
label('UC_STORE_EPOCH')
e('@P0 STS [RZ+0xbf74], uc_epoch','--:2:-:-:2')
pred('PSETP.AND.AND P2, PT, P0, P1, PT')
e('@P2 STG.E [uc_control+0x20], uc_epoch','--:2:-:-:2');e('NOP','02:-:-:-:6')
label('UC_PUBLISH_DONE')
s=s:sub(1,begin-1)..table.concat(out,'\n')..'\n'..s:sub(finish)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
