-- Extract flattened canonical GPUQ text, not a second measurement database.
-- Usage: lua trace_to_tsv.lua FLATTENED_TEXT OUTPUT_TSV [warm]
-- The optional20th column carries request identity across repeated positions.
local warm=arg[3]=='warm';assert(not arg[3]or warm)
local out=assert(io.open(assert(arg[2]),'w'));local arm,die,run;local count=0
local function number(line,key)
 local value=tonumber(line:match('"'..key..'"%s*:%s*(%d+)'));assert(value,'missing '..key);return value
end
local function string_value(line,key)
 local value=line:match('"'..key..'"%s*:%s*"([^"]+)"');assert(value,'missing '..key);return value
end
for line in io.lines(assert(arg[1]))do
 local a,d=line:match('^xtrace arm=([%w_%-]+) die=(%d+)')
 if a then arm=a;die=tonumber(d);run=tonumber(line:match('run=(%-?%d+)'))or -1 end
 if line:find('"kind"%s*:%s*"exchange_trace"')then
  assert(arm,'trace without benchmark arm header')
  if warm then assert(run>=0,'warm extraction requires BENCH_XTRACE_EACH=1')end
  if not warm or run>0 then
   assert(line:match('"valid"%s*:%s*(%a+)')=='true','invalid trace record')
   local actual_die=number(line,'die');assert(actual_die==die,'trace/header die mismatch')
   local arrivals=assert(line:match('"arrival_ns"%s*:%s*%[([^%]]*)%]'),'missing arrivals')
   local row={arm,number(line,'position'),actual_die,number(line,'epoch'),
    number(line,'producer_phase'),number(line,'consumer_phase'),string_value(line,'producer_op'),string_value(line,'consumer_op'),
    number(line,'ready_ns'),number(line,'published_ns'),number(line,'wait_start_min_ns'),number(line,'wait_start_max_ns'),number(line,'acquire_ns')}
   local n=0;for v in arrivals:gmatch('%d+')do row[#row+1]=assert(tonumber(v));n=n+1 end
   assert(n==6,'expected six producer arrivals');row[#row+1]=number(line,'request')
   for i,v in ipairs(row)do out:write(tostring(v),i==#row and '\n'or '\t')end;count=count+1
  end
 end
end
out:close();assert(count>0,'no trace records extracted')
print(string.format('trace_to_tsv records=%d warm_only=%s request_column=1',count,tostring(warm)))
