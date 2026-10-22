-- Audit six-die descriptor geometry, including every compound-phase member.
-- This proves geometry bounds, not elapsed-time bounds or dynamic input equality.
-- Usage: lua work_report.lua PROGRAM_DUMP BALANCED_SASS [expected_phases]
local f=assert(io.open(assert(arg[2])));local sass=f:read('a');f:close()
local expected=tonumber(arg[3]) or 492
assert(expected>0 and expected<500)
local phases,records={},0
for line in io.lines(assert(arg[1])) do
 local d,p,m,index,name,words=line:match('^workdesc die=(%d+) phase=(%d+) member=(%d+) index=(%d+) name=(%S+) words=(%S+)$')
 if d then
  d,p,m,index=tonumber(d),tonumber(p),tonumber(m),tonumber(index)
  assert(d<6 and p<expected,'out-of-range die/phase')
  phases[p]=phases[p] or {};phases[p][m]=phases[p][m] or {}
  assert(not phases[p][m][d],'duplicate descriptor')
  local w={};local n=0
  for x in words:gmatch('[^,]+') do assert(x:match('^%x%x%x%x%x%x%x%x$'));w[n]=tonumber(x,16);n=n+1 end
  assert(n==32,'descriptor width')
  phases[p][m][d]={w=w,name=name,index=index};records=records+1
 elseif line:match('^workdesc') then error('malformed descriptor') end
end
local tiles={}
local function geometry(a)
 local w,n=a.w,a.name;local v={};local function put(k,x)assert(x and x>=0,k);v[k]=x end
 put('descriptor_units',w[1])
 local row8=w[30]&0x80000~=0
 local matrix=n:match('^q[23456]k') or n:match('^iq4k') or n:match('^b32') or
              n=='q2moegn' or n=='q2moedn' or n=='q2acc640' or n=='bf16k2560' or n=='bf16bias'
 if matrix then
  local rows=w[(n=='bf16k2560' or n=='bf16bias') and 8 or 10]
  local k=tonumber(n:match('(%d%d%d+)$')) or tonumber(n:match('(%d%d%d+)acc$')) or
    (n=='q2moegn' and 2560) or ((n=='q2moedn' or n:match('^b32')) and 640) or
    (n=='bf16bias' and 2560)
  assert(k,'unknown matrix width '..n)
  if not row8 and not tiles[n] then
   tiles[n]=2^tonumber(assert(sass:match('SHL '..n..'_rowstep, '..n..'_rowstep, 0x(%x+);'),'tile '..n),16)
  end
  local wide=w[30]&0x2000000~=0
  local tile=row8 and (wide and 32 or 8) or tiles[n]
  local ranks=(n=='q2moegn' or n=='q2moedn') and 10 or 1
  put('rows',rows);put('K',k);put('ranks',ranks)
  put('dot_values',rows*k*ranks);put('issued_dot_values',math.ceil(rows/tile)*tile*k*ranks)
  put('row_tiles',math.ceil(rows/tile))
  if row8 then
   assert(w[1]==math.ceil(rows/tile) and rows%8==0,'row-stripe geometry')
   if wide then assert(k==640 and w[22]==2560,'wide down tile geometry') end
   put('global_rows',w[22]);put('published_ranks',w[23])
  else assert(w[1]==16,'dense dispatch geometry') end
  if w[30]&0x100000~=0 then put('published_values',rows*w[23]*6) end
  if w[30]&0x80~=0 then put('published_values',rows*6) end
  if w[30]&0x200000~=0 then put('activation_values',11*640);put('quant_groups',11*20) end
  if w[30]&0x800000~=0 then put('reduced_values',rows*6) end
  if w[30]&0x1000000~=0 then put('head_candidates',rows);put('cta_winners',16) end
 elseif n=='embrow' then
  put('columns',w[1]*32);put('packed_row_bytes',w[17]);put('peers',w[19]);put('published_values',w[1]*32*4*w[19])
 elseif n=='bf16hcrx' or n=='bf16hcrwx' or n=='grwrite_read' then
  put('dot_values',w[8]*10240);put('rows',w[8]);put('normalized_values',10240)
  local tag=n=='grwrite_read' and 'grreadsum' or n
  assert(sass:find('SHL '..tag..'_rowstep, '..tag..'_rowstep, 0x2;',1,true),'GR read row tile')
  put('issued_dot_values',math.ceil(w[8]/4)*4*10240)
  put('published_values',w[8]*6)
  if n~='bf16hcrx' then put('residual_values',10240) end
  if n=='grwrite_read' then put('sum_values',2560*(w[30]&0x400000~=0 and 1 or 6)) end
 elseif n=='hcupq_x' then
  assert(w[8]%128==0,'GR quantization alignment')
  assert(sass:find('MOV32I hcquant_xl, 0x20;',1,true),'GR up column tile')
  put('dot_values',w[8]*320);put('up_rows',w[8]);put('mix_columns',w[8]/4)
  put('issued_dot_values',math.ceil(w[8]/128)*128*320)
  local copies=sass:find('ISETP.GE.U32.AND P4, PT, warp, 0x6, PT;\n--:-:1:-:1 LDS hcq_cols,',1,true) and 6 or 1
  put('quant_evaluations',w[8]/128*copies)
  put('quant_groups',w[8]/128);put('silu_and_injection_values',w[20]);put('published_columns',w[8]/4*6)
 elseif n=='gdn_full' then
  put('heads',w[1]);put('state_cells',w[1]*128*128);put('norm_and_quant_values',w[1]*128)
 elseif n=='conv1d' then put('channels',w[11]);put('time_base',w[10])
 elseif n=='router_full' then put('rows',w[8]);put('dot_values',w[8]*2560)
 elseif n=='router_select' then put('experts_scanned',512);put('selected_ranks',10)
 elseif n=='hcwsum' then put('columns',w[8]);put('residual_values',w[9]);put('owners',w[11])
 elseif n=='ple_rowsq' then put('heads',16);put('packed_bytes',16*90);put('quant_values',2560);put('time_base',w[12])
 elseif n=='rms2560' then put('normalized_values',w[1]*2560)
 elseif n=='plegate_norm' then put('gates',w[1]);put('gate_and_norm_values',w[1]*2560)
 elseif n=='pleconv' then put('channels',10240);put('time_base',w[12])
 elseif n=='qsa_full_bal' then put('child_stages',7);put('resident_ctas',w[1])
 elseif n=='attnqkq' then put('q_heads',w[1]-1);put('kv_heads',1);put('head_width',256);put('time_base',w[22])
 elseif n=='idxqk' then put('indexer_vectors',w[1]);put('width',128);put('time_base',w[20])
 elseif n=='qsa_score_select' then put('score_partitions',w[1]);put('block_limit',w[11]);put('time_base',w[10])
 elseif n=='qsa_select' then put('block_limit',w[9]);put('selection_pass_start',w[10]);put('time_base',w[8])
 elseif n=='kvgather' then put('block_limit',w[17]);put('gather_units',w[1]);put('time_base',w[16])
 elseif n=='attnpartn' then put('head_partitions',w[1]);put('block_limit',w[15]);put('time_base',w[14])
 elseif n=='attnmergeq_x' then put('heads',w[1]);put('head_width',256);put('published_values',w[1]*256*6)
 elseif n=='amax' then put('die_winners',w[6]);put('time_base',w[14]);put('prompt_length',w[15])
 else error('unhandled operation '..n) end
 return v
end
local count,metrics,bad,worst=0,0,0,1
local index={0,0,0,0,0,0};local opnames={}
print('phase\tmember\top\tcomponent\tdie0\tdie1\tdie2\tdie3\tdie4\tdie5\tmax_over_min\tpass')
for p=0,expected-1 do
 local phase=assert(phases[p],'missing phase '..p);local members=0
 for m in pairs(phase)do members=members+1 end
 local flags=assert(assert(phase[0])[0]).w[30]
 local width=flags&8==0 and 1 or (flags>>8)&255
 if width==0 then width=2 end
 assert(members==width,'compound phase member coverage')
 for m=0,members-1 do
  local group=assert(phase[m],'member gap');local dims={};local ref=assert(group[0]);opnames[ref.name]=true
  for d=0,5 do
   local a=assert(group[d],'missing die');assert(a.name==ref.name and a.w[0]==ref.w[0],'opcode mismatch')
   assert(a.index==index[d+1],'descriptor index gap');index[d+1]=a.index+1
   assert(a.w[30]==ref.w[30] and a.w[31]==ref.w[31],'different dispatch paths/token masks')
   dims[d]=geometry(a)
  end
  local keys={};for k in pairs(dims[0])do keys[#keys+1]=k end;table.sort(keys)
  for _,k in ipairs(keys)do
   local lo,hi=math.huge,0;local values={}
   for d=0,5 do local x=assert(dims[d][k]);lo=math.min(lo,x);hi=math.max(hi,x);values[#values+1]=x end
   local ratio=hi==0 and 1 or (lo==0 and math.huge or hi/lo)
   local pass=hi*10<=lo*11;bad=bad+(pass and 0 or 1);worst=math.max(worst,ratio);metrics=metrics+1
   print(string.format('%d\t%d\t%s\t%s\t%s\t%.9f\t%d',p,m,ref.name,k,table.concat(values,'\t'),ratio,pass and 1 or 0))
  end
  count=count+1
 end
end
assert(count*6==records,'coverage mismatch')
local types=0;for _ in pairs(opnames)do types=types+1 end
io.stderr:write(string.format('work_geometry phases=%d members=%d records=%d operation_types=%d components=%d failing_components=%d worst_pairwise_ratio=%.9f dynamic_input_equality=requires_separate_evidence timing=not_tested\n',expected,count,records,types,metrics,bad,worst))
if bad>0 then os.exit(1) end
