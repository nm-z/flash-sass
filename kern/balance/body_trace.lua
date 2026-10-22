-- Diagnostic image: all physical bodies, plus explicit local producer waits.
-- Reuse unused words/child records in the exchange trace allocation. R249..254
-- are outside every production body (highest live register R248). Preserve PR
-- so instrumentation at a join cannot change its predicated control flow.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function insert(anchor,code)
 local a=assert(s:find(anchor,1,true),'missing '..anchor)
 assert(not s:find(anchor,a+#anchor,true),'ambiguous '..anchor)
 s=s:sub(1,a-1)..anchor..'\n'..code..s:sub(a+#anchor)
end
local function before(anchor,code)
 local a=assert(s:find(anchor,1,true),'missing '..anchor)
 assert(not s:find(anchor,a+#anchor,true),'ambiguous '..anchor)
 s=s:sub(1,a-1)..code..s:sub(a)
end
local serial=0
local function stamp(aux,slot,composite,nonzero)
 serial=serial+1;local tag='BT_'..serial
 local out={};local function e(x)out[#out+1]='--:-:-:Y:d '..x..';'end
 local function load(x,n)
  out[#out+1]='--:-:'..n..':-:2 '..x..';'
 end
 e('P2R bt_pred, PR, RZ, 0x7f')
 e('ISETP.NE.U32.AND P6, PT, tid, RZ, PT');e('@P6 BRA '..tag..'_DONE')
 if nonzero then e('ISETP.EQ.U32.AND P6, PT, cta, RZ, PT');e('@P6 BRA '..tag..'_DONE')end
 e('MOV bt_ptr0, ctl0');e('MOV bt_ptr1, ctl1')
 load('LDG.E.64 bt_clock0, [bt_ptr+0xf0]',1)
 out[#out+1]='01:-:-:Y:d NOP;'
 load('LDG.E.64 bt_ptr0, [bt_ptr+0xe8]',2)
 load('LDS bt_temp, [RZ+0xbf70]',3)
 out[#out+1]='06:-:-:Y:d NOP;'
 e('IADD bt_temp, bt_temp, -bt_clock0')
 e('ISETP.GE.U32.AND P6, PT, bt_temp, bt_clock1, PT');e('@P6 BRA '..tag..'_DONE')
 if composite then
  load('LDS bt_clock0, [RZ+0xbf0c]',1);out[#out+1]='01:-:-:Y:d NOP;'
  e('IADD bt_clock0, bt_clock0, -0x1')
 else e('MOV bt_clock0, pidx')end
 if aux~=0 then e(string.format('IADD bt_clock0, bt_clock0, 0x%x',aux))end
 e('ISCADD bt_temp, bt_temp, bt_clock0, 0xc');e('SHL bt_temp, bt_temp, 0xb')
 e('IADD bt_ptr0.CC, bt_ptr0, bt_temp');e('IADD.X bt_ptr1, bt_ptr1, RZ')
 e('SHL bt_temp, cta, 0x6');e('IADD bt_ptr0.CC, bt_ptr0, bt_temp');e('IADD.X bt_ptr1, bt_ptr1, RZ')
 out[#out+1]=tag..'_CLOCK:'
 load('S2R bt_clock1, SR_GLOBALTIMERHI',1)
 load('S2R bt_clock0, SR_GLOBALTIMERLO',2)
 load('S2R bt_temp, SR_GLOBALTIMERHI',3)
 out[#out+1]='07:-:-:Y:d NOP;'
 e('ISETP.NE.U32.AND P6, PT, bt_clock1, bt_temp, PT');e('@P6 BRA '..tag..'_CLOCK')
 out[#out+1]=string.format('--:2:-:-:2 STG.E.64 [bt_ptr+0x%x], bt_clock0;',8*slot)
 out[#out+1]='02:-:-:Y:d NOP;'
 if aux==0 and slot==6 then
  -- The former optional GR-hash region stores CTA residency, not a claim
  -- that every resident CTA performs the same amount of useful arithmetic.
  e('SHL bt_temp, cta, 0x4');e('SHL bt_clock0, cta, 0x6')
  e('IADD bt_temp, bt_temp, -bt_clock0')
  e('IADD bt_ptr0.CC, bt_ptr0, bt_temp');e('IADD.X bt_ptr1, bt_ptr1, RZ')
  load('S2R bt_clock0, SR_SMID',1);out[#out+1]='01:-:-:Y:d NOP;'
  out[#out+1]='--:2:-:-:2 STG.E [bt_ptr+0x700], bt_clock0;'
  out[#out+1]='02:-:-:Y:d NOP;'
 end
 out[#out+1]=tag..'_DONE:';e('R2P PR, bt_pred, 0x7f')
 return '3f:-:-:Y:d NOP;\n'..table.concat(out,'\n')..'\n'
end
before('</REGISTER_MAPPING>',[=[
    249 : bt_temp
    250,251 : bt_ptr<0-1>
    252,253 : bt_clock<0-1>
    254 : bt_pred
]=])
-- XWDONE is entered once after the incoming acquire; child replays branch
-- straight to OP_DISPATCH. PHASE_END follows completion of the entire group.
insert('XWDONE:','3f:-:-:Y:d BAR.SYNC 0x1;\n'..stamp(0,6,false)..
 '3f:-:-:Y:d BAR.SYNC 0x1;\n')
local at=assert(s:find('\nPHASE_END:',1,true))
local _,ending=assert(s:find('[^\n]*BAR%.SYNC 0x1;\n',at))
s=s:sub(1,ending)..stamp(0,7,false)..s:sub(ending+1)
local function pair(first,last,event,composite)
 local aux=1+event//4;local slot=2*(event%4)
 -- Enter once before the polling label; loop backedges must not reset time.
 before(first..':',stamp(aux,slot,composite))
 insert(last..':',stamp(aux,slot+1,composite))
end
pair('BAL_SELECT_READY_WAIT','BAL_SELECT_READY_BARRIER',0,true)
pair('bscore_SCORE_JOIN_WAIT','bscore_SCORE_JOIN_READY',1,true)
pair('BAL_SELECT_RETURN_WAIT','BAL_SELECT_RETURN_BARRIER',2,true)
pair('BAL_ATTN_GATHER_READY_WAIT','BAL_ATTN_GATHER_READY_BARRIER',3,true)
pair('BAL_ATTN_PART_READY_WAIT','BAL_ATTN_PART_READY_BARRIER',4,true)
pair('BM_PREP_WAIT','BM_PREP_BARRIER',0,false)
before('ROUTE_JOIN:',stamp(1,0,false))
insert('@!P5 BRA ROUTE_JOIN;',stamp(1,1,false))
-- Final publisher clocks and flag arrival observations remain in the normal
-- exchange tracer. No trace points alter model descriptors or dispatch count.
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
