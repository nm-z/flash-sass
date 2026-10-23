-- Reuse canonical GPUq streams; do not submit another profiling job.
-- Usage: lua spec_profile_report.lua OUTPUT_PREFIX LABEL=JOB_ID ...
local prefix=assert(arg[1]);local samples={};local names={}
for i=2,#arg do
 local label,id=assert(arg[i]:match('^([%w_]+)=([%x%-]+)$'))
 -- assert returns both captures here because the match is its only argument.
 assert(id and #id==36,'job id')
 local command='ssh archy /home/nate/.cache/gpuq-readonly-stream-logs '..
  '/home/nate/.local/state/gpuq/history.sqlite3 '..id..' stderr'
 local pipe=assert(io.popen(command));local sample={label=label,id=id,ops={}}
 for line in pipe:lines()do
  local d,op,count,native,us=line:match('^specprofile start=%d+ width=5 die=(%d+) op=(%S+) phases=(%d+) native=(%d+) us=(%S+)$')
  if d then
   d=tonumber(d);assert(d<6);sample.ops[op]=sample.ops[op] or {};names[op]=true
   assert(not sample.ops[op][d],'duplicate profile')
   sample.ops[op][d]={count=tonumber(count),native=tonumber(native),us=assert(tonumber(us))}
  end
 end
 assert(pipe:close());samples[#samples+1]=sample
end
assert(#samples>=2)
local order={};for op in pairs(names)do order[#order+1]=op end
table.sort(order,function(a,b)return samples[1].ops[a][0].us>samples[1].ops[b][0].us end)
local csv=assert(io.open(prefix..'.csv','w'));csv:write('label,job,die,op,phases,native,us_per_five_position_pass\n')
local md=assert(io.open(prefix..'.md','w'))
md:write('# Saved first five-position verification profile\n\n')
for _,s in ipairs(samples)do md:write('`',s.label,'`: canonical job `',s.id,'`.\n\n')end
md:write('Full48-layer model;64 prompt tokens and128 committed outputs; four-draft prompt lookup with greedy acceptance and recovery. Each profile covers the first five-position target call (start66). Values are inclusive device phase intervals, averaged over six dies, in **µs per five-position pass**. They include waits and are not isolated arithmetic or request averages. Phase0 has no preceding timestamp. These are separate observations, not a randomized paired experiment.\n\n| Operation')
for _,s in ipairs(samples)do md:write('| ',s.label,' µs')end
md:write('|\n|---');for _ in ipairs(samples)do md:write('|---:')end;md:write('|\n')
local totals={}
for _,op in ipairs(order)do
 md:write('| `',op,'`')
 for i,s in ipairs(samples)do
  local sum=0
  for d=0,5 do
   local v=assert(s.ops[op][d],op..' missing die');sum=sum+v.us
   csv:write(string.format('%s,%s,%d,%s,%d,%d,%.3f\n',s.label,s.id,d,op,v.count,v.native,v.us))
  end
  totals[i]=(totals[i] or 0)+sum/6;md:write(string.format('| %.1f',sum/6))
 end
 md:write('|\n')
end
md:write('| **All491 intervals**');for _,v in ipairs(totals)do md:write(string.format('| **%.1f**',v))end;md:write('|\n')
md:write('\nThe receiver-local quantization prefix is charged to the following consumer (often `router_full`), while its producer omits quantized peer payloads. Compare the complete pass as well as `hcupq_x`. CSV retains each die and phase count.\n')
md:close();csv:close()
