-- Instrument a single full-kernel GR operation. Original arithmetic and
-- predicates remain in order; trace stores use otherwise-dead registers.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local output=assert(arg[2]);local phase=tonumber(arg[3] or '153')
local aliases={}
for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 local registers,name=line:match('^%s*([%d,%s%-]+)%s*:%s*(%S+)')
 if registers then
  local first=tonumber(registers:match('%d+'));local stem,lo,hi=name:match('^(.-)<(%d+)%-(%d+)>$')
  if stem then
   aliases[stem]=first
   for i=tonumber(lo),tonumber(hi)do aliases[stem..string.format('%0'..#lo..'d',i)]=first+i-tonumber(lo)end
  else aliases[name]=first end
 end
end
local first=assert(s:find('grreadsum_SETUP:',1,true));local last=assert(s:find('\n[%w_]+_SETUP:',first+1))
local body=s:sub(first,last-1);local site=0;local listing={};local lines={}
for line in body:gmatch('[^\n]+')do
 local control,ins=line:match('^(%S+)%s+(.+);')
 if control and (ins:find('LDG.',1,true) or ins:find('STG.',1,true)) then
  local base=assert(ins:match('%[([%w_]+)'),'memory base')
  local reg=base:match('^R(%d+)$') or aliases[base];assert(reg,'unresolved address '..base);reg=tonumber(reg)
  assert(reg<237,'diagnostic scratch overlaps original memory operand')
  local predicate=ins:match('^(@!?P[0-6])%s+') or ''
  local mask=assert(control:match('^([^:]+):'))
  site=site+1;listing[#listing+1]=string.format('%d\t%s\tR%d/R%d\t%s',site,predicate,reg,reg+1,ins)
  lines[#lines+1]=string.format('%s:-:-:-:6 MOV grdiag_record0, R%d;',mask,reg)
  lines[#lines+1]=string.format('--:-:-:-:6 MOV grdiag_record1, R%d;',reg+1)
  lines[#lines+1]=string.format('--:-:-:-:6 MOV32I grdiag_record2, 0x%x;',site)
  lines[#lines+1]='--:-:-:-:6 MOV grdiag_record3, pidx;'
  lines[#lines+1]='--:6:-:-:2 '..predicate..' STG.E.WT.128 [grdiag_slot], grdiag_record0;'
  lines[#lines+1]='20:-:-:-:6 NOP;'
 end
 lines[#lines+1]=line
end
body=table.concat(lines,'\n')
local labels={};for line in body:gmatch('[^\n]+')do local label=line:match('^([%a_][%w_]*):%s*$');if label then labels[#labels+1]=label end end
for _,label in ipairs(labels)do body=body:gsub('%f[%w_]'..label..'%f[^%w_]','GRPROBE_'..label)end
local anchor='GRPROBE_grreadsum_BEGIN:'
local at=assert(body:find(anchor,1,true))
local init=[=[
3f:-:-:-:6 MOV grdiag_slot0, ctl0;
--:-:-:-:6 MOV grdiag_slot1, ctl1;
--:2:1:-:2 LDG.E.CG.64 grdiag_slot0, [grdiag_slot+0xd0];
03:-:-:-:6 ISCADD grdiag_record2, cta, tid, 0x8;
--:-:-:-:6 SHL grdiag_record2, grdiag_record2, 0x4;
--:-:-:-:6 IADD grdiag_slot0.CC, grdiag_slot0, grdiag_record2;
--:-:-:-:6 IADD.X grdiag_slot1, grdiag_slot1, RZ;
]=]
body=body:sub(1,at+#anchor-1)..'\n'..init..body:sub(at+#anchor)
local before=string.format([=[grreadsum_SETUP:
--:-:-:Y:d ISETP.EQ.U32.AND P6, PT, pidx, 0x%x, PT;
--:-:-:-:6 NOP;
--:-:-:Y:d @P6 BRA GRPROBE_grreadsum_SETUP;
]=],phase)
s=s:sub(1,first-1)..before..s:sub(first+#'grreadsum_SETUP:')
at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..[=[
    238,239 : grdiag_slot<0-1>
    244-247 : grdiag_record<0-3>
]=]..s:sub(at)
at=assert(s:find('\nPOSITION_END:',1,true));s=s:sub(1,at-1)..'\n'..body..s:sub(at)
f=assert(io.open(output,'w'));f:write(s);f:close()
f=assert(io.open(output..'.sites.tsv','w'));f:write(table.concat(listing,'\n')..'\n');f:close()
io.stderr:write(string.format('gr_address_probe phase=%d sites=%d memory_operations_unchanged=1 diagnostic_only=1\n',phase,site))
