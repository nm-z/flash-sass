-- Read grhash records exported from the canonical GPUQ log. Compare each die
-- against its own first request and compare replicated residuals across dies.
local dir=assert(arg[2]);local reference,spatial,counts,coordinates={},{},{},{}
local first_request;local first_temporal,first_spatial={},{}
local differences=assert(io.open(dir..'/gr-hash-differences.csv','w'))
differences:write('comparison,request,position,die,layer,operation,phase,reference_hash,actual_hash\n')
local function before(a,b)
 if not b then return true end
 local ar=a.layer<0 and 1000 or 2*a.layer+(a.op=='bf16hcrwx' and 1 or 0)
 local br=b.layer<0 and 1000 or 2*b.layer+(b.op=='bf16hcrwx' and 1 or 0)
 return a.position<b.position or (a.position==b.position and ar<br)
end
local total,bad_t,bad_s=0,0,0
for line in io.lines(assert(arg[1]))do
 if line:match('^grhash ') and line:find('point=updated_residual',1,true)then
  local r={};for k,v in line:gmatch('([%w_]+)=([^ ]+)')do r[k]=k=='hash' and v or tonumber(v)or v end
  assert(type(r.hash)=='string' and #r.hash==16)
  first_request=first_request or r.request
  local coord=r.position..':'..r.layer..':'..r.op
  local dk=r.die..':'..coord;local rk=r.request..':'..r.die;local sk=r.request..':'..coord
  coordinates[rk]=coordinates[rk]or {};assert(not coordinates[rk][coord],'duplicate coordinate');coordinates[rk][coord]=true
  counts[rk]=(counts[rk]or 0)+1;total=total+1
  if r.request==first_request then reference[dk]=r.hash
  else
   local ref=assert(reference[dk],'missing first-request coordinate')
   if ref~=r.hash then
    bad_t=bad_t+1;if before(r,first_temporal[rk])then first_temporal[rk]=r end
    differences:write(string.format('repeat,%d,%d,%d,%d,%s,%d,%s,%s\n',r.request,r.position,r.die,r.layer,r.op,r.phase,ref,r.hash))
   end
  end
  if r.die==0 then spatial[sk]=r.hash
  else
   local ref=assert(spatial[sk],'missing GPU0 coordinate')
   if ref~=r.hash then
    bad_s=bad_s+1;if before(r,first_spatial[rk])then first_spatial[rk]=r end
    differences:write(string.format('die,%d,%d,%d,%d,%s,%d,%s,%s\n',r.request,r.position,r.die,r.layer,r.op,r.phase,ref,r.hash))
   end
  end
 end
end
differences:close();assert(total>0,'no updated residual fingerprints')
local expected=assert(coordinates[first_request..':0']);local n=counts[first_request..':0']
for key,set in pairs(coordinates)do
 assert(counts[key]==n,'unequal record coverage')
 for coord in pairs(expected)do assert(set[coord],'missing residual coordinate')end
end
local md=assert(io.open(dir..'/gr-hash-first-differences.md','w'))
md:write(string.format('# Updated residual fingerprint comparison\n\n%d records; %d residual coordinates per request/die. All request/die coordinate sets match. %d differences across repeated fixed-input requests; %d differences from GPU0 within the same request. These hashes fingerprint all10240 updated residual values already computed by the original operation. They are diagnostics, not an independent output-correctness oracle.\n\n',total,n,bad_t,bad_s))
md:write('| Comparison | Request | Die | First position | GR consumer layer | Operation | Phase |\n|---|---:|---:|---:|---:|---|---:|\n')
for _,kind in ipairs({'repeat','die'})do local rows=kind=='repeat' and first_temporal or first_spatial
 local keys={};for k in pairs(rows)do keys[#keys+1]=k end;table.sort(keys)
 for _,k in ipairs(keys)do local r=rows[k];md:write(string.format('| %s | %d | %d | %d | %d | %s | %d |\n',kind,r.request,r.die,r.position,r.layer,r.op,r.phase))end
end
md:write('\nLayer -1 denotes the output head. `grwrite_read` applies the preceding MoE residual update before the named consumer layer; `bf16hcrwx` applies that layer’s attention residual update. The first differing hash brackets a producer/consumer boundary; it does not by itself prove which side is defective.\n')
md:close();print(string.format('gr_hash_report records=%d repeat_differences=%d cross_die_differences=%d',total,bad_t,bad_s))
