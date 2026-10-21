-- Offline descriptor audit. No model execution, timing estimates or GPU calls.
-- usage: lua wait_report.lua SCHEDULE_TEXT OUTPUT_CSV
local dies={};local row
for line in io.lines(assert(arg[1]))do
 local kind=line:match('^(wait_%w+) ')
 if kind=='wait_phase' or kind=='wait_body'then
  local t={};for k,v in line:gmatch('([%w_]+)=([^ ]+)')do t[k]=tonumber(v)or v end
  if kind=='wait_phase'then dies[t.die]=dies[t.die]or {};table.insert(dies[t.die],t);t.members={};row=t
  else assert(row.die==t.die and row.phase==t.phase);row.members[#row.members+1]=t.name end
 end
end
local function outputs(p)
 local op=p.name
 if op=='grwrite_read' or op=='bf16hcrx' or op=='bf16hcrwx'then return 'GR down/injection shards; normalized residual xh; residual update where applicable' end
 if op=='hcupq_x'then return 'Z activation codes/scales/sums and FP32 GR-up/mix output' end
 if op=='gdn_full'then return 'normalized gated GDN output codes/scales/sums; local recurrent and convolution state' end
 if op=='router_full'then return 'local expert indices/weights/count; cleared partial buffer; router logits' end
 if op=='q2moegn'then return 'routed and optional shared expert activation codes/scales/sums' end
 if op=='q2moedn'then return 'this die MoE partial in all six inboxes' end
 if op=='attnqkq'then return 'appended Q/K/V cache; normalized indexer query; pooled key/ring; histogram initialization' end
 if op=='qsa_score_select'then return 'selected key-block IDs; score and histogram state' end
 if op=='kvgather'then return 'compact K/V caches consumed by attention partitions' end
 if op=='qsa_part_merge'then return 'merged gated attention output codes/scales/sums in all dies' end
 if op=='ple_rowsq'then return 'PLE gathered rows and activation codes/scales/sums' end
 if op=='rms2560'then return 'PLE normalized key/query vectors (not consumed by following PLE value GEMV)' end
 if op=='plegate_norm'then return 'four normalized PLE gate branches px' end
 if op=='pleconv'then return 'updated residual S and PLE convolution state' end
 if op=='hcwsum'then return 'updated residual S from six MoE partials' end
 if op=='argmax'then return 'per-die winning token/value pairs in all six inboxes' end
 if op=='amax'then return 'selected token ID; output tokens; stop flag; next-token dependency' end
 if op=='embrow'then return 'embedding residual S on all dies' end
 if op=='xsend'then return 'Q-shard or attention-output exchange participation; may carry no payload on this die' end
 if p.send==1 then return 'projection output shards and/or exchange participation; exact matrices listed in members' end
 return 'matrix projection outputs or retained local operation outputs; members identify producers'
end
local out=assert(io.open(assert(arg[2]),'w'))
out:write('die,phase,layer,block_type,operation,members,next_operation,grid_boundary,exchange_wait,publication,decision,producer_data\n')
local function q(v)return '"'..tostring(v):gsub('"','""')..'"'end
for d=0,5 do
 local rows=assert(dies[d]);assert(#rows==499,'requires full candidate schedule')
 for i,p in ipairs(rows)do
  local nextp=rows[i+1];local block=p.layer<0 and 'embedding/head' or p.layer%4==3 and 'QSA' or 'DeltaNet'
  local decision='retain: cross-block data or token boundary'
  if p.send==1 and nextp and nextp.exchange_wait==1 then decision='remove local grid join: next exchange acquire already joins all 96 producer flags'
  elseif p.name=='rms2560' and nextp and nextp.name=='bf16k2560' then decision='remove local grid join: independent PLE value projection; later exchange acquire covers normalized key/query producer'
  elseif p.grid==0 then decision='already no local grid join' end
  local values={d,p.phase,p.layer,block,p.name,table.concat(p.members,' + '),nextp and nextp.name or 'next-token',p.grid,p.exchange_wait,p.send,decision,outputs(p)}
  for j,v in ipairs(values)do if j>1 then out:write(',')end;out:write(q(v))end;out:write('\n')
 end
end
out:close()
