-- A persistent launch rewrites activations, scales, KV and residuals. Route
-- mutable global loads through coherent L2 (.CG), retaining .CI for immutable
-- model weights. Pointer classes below are the generated weight-address names;
-- the audit CSV records every decision for review against the source generator.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local weights={wp=true,wq=true,ws=true,np=true,pp=true,hp=true}
local audit={'pointer,policy,instruction_before_completion_scheduling'};local mutable,immutable=0,0
s=s:gsub('[^\n]+',function(line)
 -- Compact PLE rows are CPU-written mapped system memory. Invalidate cached
 -- system-memory lines on each read after RDY; adjacent 1440-byte positions
 -- share cache lines, so reading RDY with CV alone does not refresh the payload.
 if line:match('LDG%.E%.U16 ple[%w]*_hw%d+, %[ple[%w]*_tp')then
  mutable=mutable+1;line=line:gsub('LDG%.E%.U16','LDG.E.CV.U16')
  audit[#audit+1]=string.format('%s,CV,"%s"',assert(line:match('%[([%w_]+)')),line:gsub('"','""'))
  return line
 end
 if not line:find('LDG.E.CI',1,true) then
  if line:find('LDG.E.CV',1,true) and (line:find('grreadsum_',1,true) or line:find('grcache_',1,true))then
   mutable=mutable+1;line=line:gsub('LDG%.E%.CV','LDG.E.CG')
   audit[#audit+1]=string.format('%s,CG,"%s"',assert(line:match('%[([%w_]+)')),line:gsub('"','""'))
   return line
  end
  return line
 end
 local ptr=assert(line:match('%[([%w_]+)'),'unclassified global load address')
 local suffix=ptr:match('_([^_]+)$');local policy=weights[suffix] and 'CI' or 'CG'
 if policy=='CI' then immutable=immutable+1 else mutable=mutable+1;line=line:gsub('LDG%.E%.CI','LDG.E.CG')end
 audit[#audit+1]=string.format('%s,%s,"%s"',ptr,policy,line:gsub('"','""'))
 return line
end)
-- Host reads may return out of issue order. Six independent completions per
-- packet replace the45-word burst sharing one completion. No CTA/grid join.
local ordered,words={},0
for line in s:gmatch('[^\n]+')do
 local word=line:match('LDG%.E%.CV%.U16 ple[%w]*_hw(%d+), %[ple[%w]*_tp')
 if word then
  local index=tonumber(word);local barrier=index%6+1;local last=index%6==5 or index==44
  if index==0 then ordered[#ordered+1]='3f:-:-:-:6 NOP; # begin host payload completion packets' end
  line=line:gsub('^(%S+)(%s+)',function(ctl,space)
   local wait,read,write,yield,stall=assert(ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$'))
   return table.concat({wait,read,tostring(barrier),yield,last and '2' or stall},':')..space
  end)
  ordered[#ordered+1]=line;words=words+1
  if last then ordered[#ordered+1]=string.format('%02x:-:-:-:6 NOP; # all host payload words in this packet complete',(1<<barrier)-1)end
 else ordered[#ordered+1]=line end
end
assert(words==45 or words==90,'unexpected host PLE payload geometry')
s=table.concat(ordered,'\n')..'\n'
-- Retire the distributed residual stores before their source registers are
-- normalized/reused. A burst sharing read barrier4 must not rely on the last
-- store's operand-read completion to protect every earlier vector store.
for _,op in ipairs({'bf16hcrwx','grreadsum'})do
 local anchor='--:4:-:-:1  @P6 STG.E.128 ['..op..'_ob+0x90], '..op..'_act36;'
 local i=s:find(anchor,1,true)
 if i then
  assert(not s:find(anchor,i+#anchor,true))
  s=s:sub(1,i+#anchor-1)..'\n08:-:-:-:6 MEMBAR.GL;\n--:-:-:-:6 BAR.SYNC 0x0; # residual store lifetime handoff'..s:sub(i+#anchor)
 end
end
assert(mutable>1000 and immutable>500,'unexpected global-load geometry '..mutable..'/'..immutable)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
f=assert(io.open(assert(arg[3]),'w'));f:write(table.concat(audit,'\n'),'\n');f:close()
io.stderr:write(string.format('mutable_loads coherent=%d immutable_cached=%d\n',mutable,immutable))
