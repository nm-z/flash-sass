-- Diagnostic image: all physical bodies, plus explicit local producer waits.
-- Reuse unused words/child records in the exchange trace allocation. R234..239
-- are dead at body boundaries and all selected internal poll sites (router,
-- PLE, score join and MoE preparation use lower registers there). Keeping the
-- production R248 ceiling avoids changing its register allocation. P6 is dead
-- at these exact sites: body dispatch/footer overwrite it, and each selected
-- polling loop uses P3/P5 instead. Other predicates remain untouched.
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
 local function put(x)
  e('ISETP.EQ.U32.AND P6, PT, tid, RZ, PT')
  out[#out+1]='--:2:-:-:2 @P6 '..x..';'
 end
 -- Every branch must be warp-uniform. Branching 31 lanes around the tracer
 -- without SSY/SYNC leaves the warp divergent when model dispatch resumes.
 -- All lanes of warp0 compute the record; only its lane0 stores it.
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA '..tag..'_DONE')
 if nonzero then e('ISETP.EQ.U32.AND P6, PT, cta, RZ, PT');e('@P6 BRA '..tag..'_DONE')end
 e('MOV bt_ptr0, ctl0');e('MOV bt_ptr1, ctl1')
 load('LDG.E bt_clock0, [bt_ptr+0xf0]',1)
 load('LDG.E bt_clock1, [bt_ptr+0xf4]',2)
 load('LDS bt_temp, [RZ+0xbf70]',3)
 out[#out+1]='07:-:-:Y:d NOP;'
 e('IADD bt_temp, bt_temp, -bt_clock0')
 e('ISETP.GE.U32.AND P6, PT, bt_temp, bt_clock1, PT');e('@P6 BRA '..tag..'_DONE')
 -- Keep the control address intact until the window is known to be active;
 -- the trace pointer load must not overwrite its own address register pair.
 load('LDG.E.64 bt_clock0, [bt_ptr+0xe8]',1)
 out[#out+1]='01:-:-:Y:d NOP;'
 e('MOV bt_ptr0, bt_clock0');e('MOV bt_ptr1, bt_clock1')
 if composite then
  load('LDS bt_clock0, [RZ+0xbf0c]',1);out[#out+1]='01:-:-:Y:d NOP;'
  e('IADD bt_clock0, bt_clock0, -0x1')
 else e('MOV bt_clock0, pidx')end
 if aux~=0 then e(string.format('IADD bt_clock0, bt_clock0, 0x%x',aux))end
 e('ISCADD bt_temp, bt_temp, bt_clock0, 0xc');e('SHL bt_temp, bt_temp, 0xb')
 e('IADD bt_ptr0.CC, bt_ptr0, bt_temp');e('IADD.X bt_ptr1, bt_ptr1, RZ')
 if aux==0 and slot==6 then
  -- The former optional GR-hash region stores CTA residency, not a claim
  -- that every resident CTA performs the same amount of useful arithmetic.
  -- Derive both destinations from the unchanged phase base. A negative u32
  -- adjustment with IADD.X high,RZ adds an erroneous 4GiB carry for CTA>0.
  e('SHL bt_temp, cta, 0x4')
  e('IADD bt_clock0.CC, bt_ptr0, bt_temp');e('IADD.X bt_clock1, bt_ptr1, RZ')
  -- ptxas11.4 lowers %smid on sm_52 to VIRTID[28:20]; see smid.ptx.
  -- SR_SMID is absent from the legacy MaxAs table and encodes as LANEID.
  load('S2R bt_temp, SR_VIRTID',1);out[#out+1]='01:-:-:Y:d NOP;'
  e('BFE.U32 bt_temp, bt_temp, 0x914')
  put('STG.E [bt_clock+0x700], bt_temp')
  out[#out+1]='02:-:-:Y:d NOP;'
  e('ISETP.NE.U32.AND P6, PT, cta, RZ, PT');e('@P6 BRA '..tag..'_RESIDENT')
  load('LDS bt_temp, [RZ+0xbf80]',1);out[#out+1]='01:-:-:Y:d NOP;'
  e('ISETP.NE.U32.AND P6, PT, bt_temp, 0x3d, PT');e('@P6 BRA '..tag..'_RESIDENT')
  -- The routed gate starts the four-member gate/up group. Its second
  -- auxiliary record is unused by poll clocks; retain the actual selected
  -- global IDs here rather than assuming ten useful experts from static rows.
  load('LDS.64 bt_clock0, [RZ+0xbfb0]',1);out[#out+1]='01:-:-:Y:d NOP;'
  for i=0,9 do
   load(string.format('LDG.E.CG bt_temp, [bt_clock+0x%x]',4*i),1)
   out[#out+1]='01:-:-:Y:d NOP;'
   put(string.format('STG.E [bt_ptr+0x%x], bt_temp',0x1000+4*i))
   out[#out+1]='02:-:-:Y:d NOP;'
  end
  load('LDS.64 bt_clock0, [RZ+0xbfc8]',1);out[#out+1]='01:-:-:Y:d NOP;'
  load('LDG.E.CG bt_temp, [bt_clock]',1);out[#out+1]='01:-:-:Y:d NOP;'
  put('STG.E [bt_ptr+0x1028], bt_temp')
  out[#out+1]='02:-:-:Y:d NOP;'
  out[#out+1]=tag..'_RESIDENT:'
 end
 -- Start the body clock after metadata capture, at the CTA's own record.
 e('SHL bt_temp, cta, 0x6');e('IADD bt_ptr0.CC, bt_ptr0, bt_temp');e('IADD.X bt_ptr1, bt_ptr1, RZ')
 out[#out+1]=tag..'_CLOCK:'
 load('S2R bt_clock1, SR_GLOBALTIMERHI',1)
 load('S2R bt_clock0, SR_GLOBALTIMERLO',2)
 load('S2R bt_temp, SR_GLOBALTIMERHI',3)
 out[#out+1]='07:-:-:Y:d NOP;'
 e('ISETP.NE.U32.AND P6, PT, bt_clock1, bt_temp, PT');e('@P6 BRA '..tag..'_CLOCK')
 put(string.format('STG.E.64 [bt_ptr+0x%x], bt_clock0',8*slot))
 out[#out+1]='02:-:-:Y:d NOP;'
 out[#out+1]=tag..'_DONE:'
 return '3f:-:-:Y:d NOP;\n'..table.concat(out,'\n')..'\n'
end
before('</REGISTER_MAPPING>',[=[
    239 : bt_temp
    234,235 : bt_ptr<0-1>
    236,237 : bt_clock<0-1>
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
-- PLE has no auxiliary descriptor. The second half of the diagnostic arena
-- is reserved for its host-prefetch wait, and is guarded by the host builder.
before('plegatherq_WAITRDY:',stamp(2048,0,false))
insert('@P3 BRA plegatherq_WAITRDY;',stamp(2048,1,false))
-- Final publisher clocks and flag arrival observations remain in the normal
-- exchange tracer. No trace points alter model descriptors or dispatch count.
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
