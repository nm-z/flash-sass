-- Complete local QSA stages. Child descriptors are loaded from PROG with
-- their own +TOK masks; preloaded B8 banks are not consumed. The bf00 bank is
-- therefore private scratch for these balanced composites, never live B8 data.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local maps={[=[
    16,17 : bc_pointer<0-1>
    18 : bc_word
    19 : bc_mask
    20 : bc_token
    21 : bc_temp
    22 : bc_epoch
    23 : bc_saved
]=]}
local function rep(x,a,b)
 local i=assert(x:find(a,1,true),'missing '..a);assert(not x:find(a,i+#a,true),'ambiguous '..a)
 return x:sub(1,i-1)..b..x:sub(i+#a)
end
local function extract(tag)
 local a=assert(s:find(tag..'_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
 return s:sub(a,b-1)
end
local function clone(tag,new,other,return_to)
 local body=extract(tag);local rename={[tag]=new}
 local function instructions(text)
  local n=0;for line in text:gmatch('[^\n]+')do
   if line:match('^[%x%-]+:[%d%-]+:[%d%-]+:[Y%-]:%x%s')then n=n+1 end
  end;return n
 end
 local count=instructions(body)
 for a,b in pairs(other or {})do rename[a]=b end
 for a,b in pairs(rename)do
  for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
   if line:find(': '..a..'_',1,true)then maps[#maps+1]=line:gsub(a..'_',b..'_')end
  end
  body=body:gsub(a..'_',b..'_')
 end
 -- Control words also begin with a hexadecimal token and colon (01:, 3f:).
 -- Rename only complete label lines. Treating those wait masks as labels
 -- causes MaxAs to discard the cloned completion waits and their operations.
 local labels={};for line in body:gmatch('[^\n]+')do
  local name=line:match('^([%a_][%w_]*):%s*$')
  if name and name:sub(1,#new+1)~=new..'_'then labels[#labels+1]=name end
 end
 for _,name in ipairs(labels)do body=body:gsub('%f[%w_]'..name..'%f[^%w_]',new..'_'..name)end
 body=body:gsub('BRA POSITION_END;','BRA '..return_to..';')
 assert(instructions(body)==count,'cloning lost SASS instructions: '..tag..' -> '..new)
 return body
end
local function generator()
 local out={};local function e(ins,ctl)out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'end
 local function l(x)out[#out+1]=x..':'end
 local function w(mask)e('NOP',mask..':-:-:-:6')end
 return out,e,l,w
end
local function initialize(tag)
 local out,e,l,w=generator();l(tag..'_SETUP')
 e('LDS.64 bc_pointer0, [RZ+0xbf88]','3f:-:1:-:2');w('01')
 e('LDS bc_epoch, [RZ+0xbf90]','--:-:1:-:2');w('01')
 e('LDS bc_token, [RZ+0xbf70]','--:-:1:-:2');w('01')
 e('IADD bc_token, bc_token, 0x1');e('ISCADD bc_epoch, bc_token, bc_epoch, 0x8')
 e('IADD bc_saved, pidx, 0x1');e('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
 e('@P5 STS.64 [RZ+0xbf00], bc_pointer0','--:2:-:-:2')
 e('@P5 STS [RZ+0xbf08], bc_epoch','--:2:-:-:2')
 e('@P5 STS [RZ+0xbf0c], bc_saved','--:2:-:-:2');w('02');e('BAR.SYNC 0x0')
 return table.concat(out,'\n')..'\n'
end
local function child(tag,offset)
 local out,e,l,w=generator()
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA '..tag..'_COPIED')
 e(string.format('IADD bc_temp, pidx, 0x%x',offset));e('SHL bc_temp, bc_temp, 0x7')
 e('IADD bc_pointer0.CC, prog0, bc_temp');e('IADD.X bc_pointer1, prog1, RZ')
 e('LDG.E.CG bc_mask, [bc_pointer+0x7c]','--:-:2:-:2')
 e('SHL bc_temp, lane, 0x2');e('IADD bc_pointer0.CC, bc_pointer0, bc_temp');e('IADD.X bc_pointer1, bc_pointer1, RZ')
 e('LDG.E.CG bc_word, [bc_pointer]','--:-:1:-:2')
 e('LDS bc_token, [RZ+0xbf70]','--:-:3:-:2');w('07')
 e('SHR.U32 bc_mask, bc_mask, lane');e('LOP32I.AND bc_mask, bc_mask, 0x1')
 e('ISETP.NE.U32.AND P5, PT, bc_mask, RZ, PT');e('@P5 IADD bc_word, bc_word, bc_token')
 e('STS [bc_temp+0xbf80], bc_word','--:4:-:-:2');w('08')
 l(tag..'_COPIED');e('BAR.SYNC 0x0')
 e('LDS units, [RZ+0xbf84]','--:-:1:-:2');w('01')
 return table.concat(out,'\n')..'\n'
end
local function join(tag,increment,single_producer)
 local out,e,l,w=generator();l(tag)
 e('MEMBAR.GL','3f:-:-:-:6');e('BAR.SYNC 0x0')
 e('LDS.64 bc_pointer0, [RZ+0xbf00]','--:-:1:-:2');w('01')
 e('LDS bc_epoch, [RZ+0xbf08]','--:-:1:-:2');w('01')
 if increment then
  e('IADD bc_epoch, bc_epoch, 0x1')
  e('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
  e('@P5 STS [RZ+0xbf08], bc_epoch','--:2:-:-:2');w('02')
 end
 e('SHL bc_temp, cta, 0x2');e('IADD bc_pointer0.CC, bc_pointer0, bc_temp');e('IADD.X bc_pointer1, bc_pointer1, RZ')
 if single_producer then
  e('IADD bc_temp, tid, cta')
  e('ISETP.EQ.U32.AND P5, PT, bc_temp, RZ, PT')
 else e('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')end
 e('@P5 STG.E.WT [bc_pointer], bc_epoch','--:2:-:-:2');w('02')
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA '..tag..'_BARRIER')
 e('LDS.64 bc_pointer0, [RZ+0xbf00]','--:-:1:-:2');w('01')
 if not single_producer then
  e('LOP32I.AND bc_temp, lane, 0xf');e('ISCADD bc_pointer0.CC, bc_temp, bc_pointer0, 0x2');e('IADD.X bc_pointer1, bc_pointer1, RZ')
 end
 l(tag..'_WAIT')
 e('LDG.E.CV bc_word, [bc_pointer]','--:-:1:-:2');w('01')
 -- A later stage may publish while a slower CTA still observes this one.
 -- Payloads are distinct, and monotonic epochs make >= the causal acquire.
 e('ISETP.GE.U32.AND P5, PT, bc_word, bc_epoch, PT')
 e('VOTE.ALL P5, P5');e('@!P5 BRA '..tag..'_WAIT')
 l(tag..'_BARRIER');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
 return table.concat(out,'\n')..'\n'
end
local function finish(tag)
 local out,e,l,w=generator();l(tag)
 e('LDS pidx, [RZ+0xbf0c]','3f:-:1:-:2');w('01')
 e('IADD pidx, pidx, -0x1');e('BRA POSITION_END')
 return table.concat(out,'\n')..'\n'
end
local pieces={initialize('bal_select')}
pieces[#pieces+1]=[=[
--:-:-:Y:d ISETP.LT.U32.AND P6, PT, cta, 0x5, PT;
--:-:-:Y:d @P6 BRA BAL_SELECT_NORM;
--:-:-:Y:d ISETP.EQ.U32.AND P6, PT, cta, 0x5, PT;
--:-:-:Y:d @P6 BRA BAL_SELECT_INDEX;
--:-:-:Y:d BRA BAL_SELECT_READY;
BAL_SELECT_NORM:
]=]..child('BAL_SELECT_NORM_PARAMS',1)..'--:-:-:Y:d BRA bnorm_SETUP;\n'
pieces[#pieces+1]='BAL_SELECT_INDEX:\n'..child('BAL_SELECT_INDEX_PARAMS',2)..'--:-:-:Y:d BRA bindex_SETUP;\n'
pieces[#pieces+1]=join('BAL_SELECT_READY',false)..'--:-:-:Y:d IADD pidx, pidx, 0x3;\n'..
 child('BAL_SELECT_SCORE_PARAMS',0)..'--:-:-:Y:d BRA bscore_SETUP;\n'
pieces[#pieces+1]=clone('attnqkq','bnorm',nil,'BAL_SELECT_READY')
local index=clone('idxqk','bindex',nil,'BAL_SELECT_READY')
index=rep(index,'SHL vid, cta, 0x3;','IADD vid, cta, -0x5;\n--:-:-:Y:d SHL vid, vid, 0x3;');pieces[#pieces+1]=index
local score=clone('scoreselect','bscore',{scorejoin='bsjoin'},'BAL_SELECT_RETURN')
score=score:gsub('selectone_','bchoose_');pieces[#pieces+1]=score
pieces[#pieces+1]=clone('selectone','bchoose',nil,'BAL_SELECT_RETURN')
-- Nonzero score CTAs arrive here before selection completes. Only CTA0
-- publishes the next epoch, after all selected indices have been stored.
-- Waiting for that one producer replaces the old phase-footer grid join.
pieces[#pieces+1]=join('BAL_SELECT_RETURN',true,true)..[=[
3f:-:1:-:2 LDS pidx, [RZ+0xbf0c];
01:-:-:Y:d IADD pidx, pidx, -0x1;
]=]..child('BAL_FULL_GATHER_PARAMS',5)..'--:-:-:Y:d BRA bgather_SETUP;\n'
pieces[#pieces+1]=clone('kvgather','bgather',nil,'BAL_ATTN_GATHER_READY')
pieces[#pieces+1]=join('BAL_ATTN_GATHER_READY',true)..child('BAL_ATTN_PART_PARAMS',6)..'--:-:-:Y:d BRA bpart_SETUP;\n'
pieces[#pieces+1]=clone('attnpartn','bpart',nil,'BAL_ATTN_PART_READY')
pieces[#pieces+1]=join('BAL_ATTN_PART_READY',true)..child('BAL_ATTN_MERGE_PARAMS',7)..'--:-:-:Y:d BRA bmerge_SETUP;\n'
pieces[#pieces+1]=clone('attnquant','bmerge',{attnq='bmergeq'},'BAL_ATTN_RETURN')
pieces[#pieces+1]=finish('BAL_ATTN_RETURN')
s=rep(s,'ISETP.GE.U32.AND P6, PT, op, 0x6b, PT;','ISETP.GE.U32.AND P6, PT, op, 0x6c, PT;')
s=rep(s,'--:-:-:-:6      BRA POSITION_END;\nembrow_SETUP:',[=[--:-:-:-:6 BRA bal_select_SETUP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6      BRA POSITION_END;

embrow_SETUP:]=])
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=rep(s,'\nPOSITION_END:','\n'..table.concat(pieces,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
