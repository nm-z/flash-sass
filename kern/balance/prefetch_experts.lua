-- Pipeline the next selected expert's immutable weight row behind the current
-- row's arithmetic. Slot6 owns prefetched results; slot4 protects address
-- reads and is retired before the native arithmetic reuses it. Native result
-- registers, arithmetic order and output stores are unchanged.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local maps=[=[
    180,181 : ep_wq<0-1>
    182,183 : ep_ws<0-1>
    184-193 : ep_q<0-9>
    194-198 : ep_d<0-4>
    199 : ep_id
    200 : ep_rank
    201 : ep_tmp
    202,203 : ep_idx<0-1>
    204 : ep_weight
    205 : ep_owner
    206,207 : ep_weptr<0-1>
    208,209 : ep_ownerptr<0-1>
]=]
local function once(body,a,b)
 local i=assert(body:find(a,1,true),'missing '..a);assert(not body:find(a,i+#a,true),'ambiguous '..a)
 return body:sub(1,i-1)..b..body:sub(i+#a)
end
local function gen()
 local out={};local function e(x,c)
  -- Integer address operations use the native six-cycle dependent schedule.
  -- Keep predicate writes, XMAD and branches at the conservative spacing.
  local op=x:gsub('^@!?P%d%s+',''):match('^(%w+)')
  local integer=op=='MOV' or op=='IADD' or op=='SHL' or op=='ISCADD' or op=='NOP'
  out[#out+1]=(c or (integer and '--:-:-:-:6' or '--:-:-:Y:d'))..' '..x..';'
 end
 return out,e
end
assert(arg[3]==nil or arg[3]=='gate_only','unknown prefetch scope')
for _,tag in ipairs(arg[3]=='gate_only' and {'q2moegn'} or {'q2moegn','q2moedn'})do
 local down=tag=='q2moedn';local stride=down and 1 or 2;local lanes=down and 4 or 16
 local a=assert(s:find(tag..'_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
 local body=s:sub(a,b-1)
 local start=assert(body:find(tag..'_EXPERT:',1,true));local stop=assert(body:find('\n'..tag..'_END:',start,true))
 local core=body:sub(start,stop-1)
 assert(core:find(string.format('IADD %s_e, %s_e, 0x%x;',tag,tag,stride),1,true),
        'expert rank stride changed')
 assert(not core:find(' P5,',1,true),'native expert core uses P5')
 for line in core:gmatch('[^\n]+')do
  local rd,wr=line:match('^%S-:([^:]+):([^:]+):[^:]+:%x%s')
  assert(rd~='6' and wr~='6','native expert core uses dependency slot6')
 end
 local out,e=gen()
 e('NOP','20:-:-:Y:d')
 e(string.format('ISETP.GE.U32.AND P5, PT, %s_e, 0x%x, PT',tag,stride))
 e(string.format('IADD ep_rank, %s_e, 0x%x',tag,stride))
 e('@P5 MOV '..tag..'_eoff, ep_id')
 if down then e('@P5 MOV '..tag..'_we, ep_weight');e('@P5 MOV bd_owner, ep_owner') end
 e('ISETP.LT.U32.AND P4, PT, ep_rank, '..tag..'_ne, PT')
 e('SHL ep_tmp, ep_rank, 0x2')
 e('IADD ep_idx0.CC, '..tag..'_param_D0, ep_tmp');e('IADD.X ep_idx1, '..tag..'_param_D1, RZ')
 e('@P4 LDG.E.CG ep_id, [ep_idx]','--:4:6:-:2')
 e('NOP','08:-:-:Y:d')
 body=once(body,tag..'_EXPERT:',tag..'_EXPERT:\n'..table.concat(out,'\n'))
 body=once(body,'LDG.E.CG '..tag..'_eoff, ['..tag..'_ixp];','@!P5 LDG.E.CG '..tag..'_eoff, ['..tag..'_ixp];')
 if down then
  body=once(body,'LDG.E.CG '..tag..'_we, ['..tag..'_wep];','@!P5 LDG.E.CG '..tag..'_we, ['..tag..'_wep];')
  body=once(body,'LDG.E.CG bd_owner, [bd_address];','@!P5 LDG.E.CG bd_owner, [bd_address];')
 end
 local first=assert(body:find('[^\n]*LDG%.E%.CI%.U16 '..tag..'_d0, %['..tag..'_ws%+0x0%];'))
 local _,last=assert(body:find(string.format('LDG.E.CI.64 %s_q8, [%s_wq+0x%x];',tag,tag,4*lanes*8),first,true))
 local loads=body:sub(first,last);local count
 loads,count=loads:gsub('LDG%.E%.CI','@!P5 LDG.E.CI');assert(count==10,'weight-load coverage')
 out,e=gen();out[#out+1]=loads
 for i=0,4 do e(string.format('@P5 MOV %s_d%d, ep_d%d',tag,i,i),'--:-:-:-:1')end
 for i=0,9 do e(string.format('@P5 MOV %s_q%d, ep_q%d',tag,i,i),'--:-:-:-:1')end
 e('NOP','20:-:-:Y:d')
 e('@!P4 BRA '..tag..'_PREFETCH_DONE')
 if down then
  e('SHL ep_tmp, ep_rank, 0x2')
  e('IADD ep_weptr0.CC, '..tag..'_param_WE0, ep_tmp');e('IADD.X ep_weptr1, '..tag..'_param_WE1, RZ')
  e('LDG.E.CG ep_weight, [ep_weptr]','--:4:6:-:2')
  e('SHL ep_tmp, ep_id, 0x2')
  e('IADD ep_ownerptr0.CC, bd_bounds0, ep_tmp');e('IADD.X ep_ownerptr1, bd_bounds1, RZ')
  e('LDG.E.CG ep_owner, [ep_ownerptr]','--:4:6:-:2')
 end
 e('XMAD ep_tmp, ep_id, '..tag..'_param_rows, '..tag..'_row')
 local bits=down and {6,7} or {4,6,7,9}
 e(string.format('SHL ep_wq0, ep_tmp, 0x%x',bits[1]))
 for i=2,#bits do e(string.format('ISCADD ep_wq0, ep_tmp, ep_wq0, 0x%x',bits[i]))end
 e('ISCADD ep_tmp, '..tag..'_blk0, ep_wq0, 0x1')
 e('IADD ep_ws0.CC, '..tag..'_param_W0, ep_tmp');e('IADD.X ep_ws1, '..tag..'_param_W1, RZ')
 e('ISCADD ep_tmp, '..tag..'_blk0, ep_wq0, 0x4')
 e('ISCADD ep_tmp, '..tag..'_half, ep_tmp, 0x3')
 e(string.format('IADD ep_tmp, ep_tmp, 0x%x',down and 32 or 80))
 e('IADD ep_wq0.CC, '..tag..'_param_W0, ep_tmp');e('IADD.X ep_wq1, '..tag..'_param_W1, RZ')
 for i=0,4 do e(string.format('LDG.E.CI.U16 ep_d%d, [ep_ws+0x%x]',i,lanes*i),'--:4:6:-:1')end
 for i=0,4 do e(string.format('LDG.E.CI.64 ep_q%d, [ep_wq+0x%x]',2*i,lanes*8*i),'--:4:6:-:2')end
 e('NOP','08:-:-:Y:d')
 out[#out+1]=tag..'_PREFETCH_DONE:'
 body=body:sub(1,first-1)..table.concat(out,'\n')..body:sub(last+1)
 -- Prefetched iterations already have the complete weight row. Avoid
 -- rebuilding the unused current-expert weight pointers on those iterations.
 local function first_only(first_ins,last_ins,label)
  local p=assert(body:find(first_ins,1,true))
  local line=body:sub(1,p-1):match('.*()\n') or 0
  local _,q=assert(body:find(last_ins,p,true))
  body=body:sub(1,line)..'--:-:-:Y:d @P5 BRA '..label..';\n'..
       body:sub(line+1,q)..'\n'..label..':'..body:sub(q+1)
 end
 first_only('XMAD '..tag..'_t0, '..tag..'_eoff, '..tag..'_param_rows, RZ;',
   string.format('ISCADD %s_eoff, %s_t0, %s_eoff, 0x%x;',tag,tag,tag,down and 7 or 9),
   tag..'_PREFETCH_SKIP_OFFSET')
 first_only('SHL '..tag..'_off, '..tag..'_row, '..(down and '0x6;' or '0x4;'),
   'IADD.X '..tag..'_wq1, RZ, '..tag..'_param_W1;',tag..'_PREFETCH_SKIP_POINTERS')
 s=s:sub(1,a-1)..body..s:sub(b)
 io.stderr:write(string.format('expert_prefetch op=%s rank_stride=%d complete_weight_rows=1\n',tag,stride))
end
local at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..maps..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
