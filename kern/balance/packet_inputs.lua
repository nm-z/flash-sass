-- K640 down projections consume32-value records with128-byte stride:
-- codes0..63, S32 at64, two S16 sums at68/72, FP32 scale at76.
-- Weight tiles, row ownership, dot arithmetic and owner accumulation stay intact.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function once(body,a,b)
 local i=assert(body:find(a,1,true),'missing '..a)
 assert(not body:find(a,i+#a,true),'ambiguous '..a)
 return body:sub(1,i-1)..b..body:sub(i+#a)
end
for _,tag in ipairs({'q2moedn','q2acc640','b32q40acc','b32q50acc','b32q80acc','b32iq4nlacc'})do
 local a=assert(s:find(tag..'_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
 local body=s:sub(a,b-1);local q2=tag:sub(1,2)=='q2';local temp=tag..(q2 and '_t1' or '_t1p')
 body=once(body,'SHL '..temp..', '..tag..'_lane, 0x6;','SHL '..temp..', '..tag..'_lane, 0x7;')
 local n;body,n=body:gsub('SHL '..temp..', '..tag..'_lane, 0x2;',
                          'SHL '..temp..', '..tag..'_lane, 0x7;')
 assert(n>=1 and n<=2,'sum/scale lane geometry '..tag)
 if tag=='q2moedn' then
  body=once(body,'XMAD q2moedn_t0, q2moedn_e, 0x500, RZ;','XMAD q2moedn_t0, q2moedn_e, 0xa00, RZ;')
  body=once(body,'XMAD q2moedn_t0, q2moedn_e, 0x50, RZ;','XMAD q2moedn_t0, q2moedn_e, 0xa00, RZ;')
 end
 body,n=body:gsub('%['..tag..'_xp%+0x(%x+)%]',function(x)
  local old=tonumber(x,16);assert(old<0x500 and old%16==0)
  return string.format('[%s_xp+0x%x]',tag,(old//256)*512+old%256)
 end)
 assert(n==20,'activation load coverage '..tag..' '..n)
 for _,kind in ipairs({q2 and 'sp' or 'zp','ap'})do
  body,n=body:gsub('%['..tag..'_'..kind..'%+0x(%x+)%]',function(x)
   local old=tonumber(x,16);assert(old<=64 and old%16==0)
   return string.format('[%s_%s+0x%x]',tag,kind,old*32)
  end)
  assert(n==5 or (kind=='zp' and n==0),'metadata load coverage '..tag..' '..kind..' '..n)
 end
 -- Packet data is produced in this persistent launch; it is not immutable
 -- weight data eligible for the read-only cache.
 body=body:gsub('[^\n]*LDG[^\n]+',function(line)
  if line:find('['..tag..'_xp+',1,true) or line:find('['..tag..'_sp+',1,true) or
     line:find('['..tag..'_zp+',1,true) or line:find('['..tag..'_ap+',1,true) then
   return line:gsub('%.CI','.CG')
  end
  return line
 end)
 s=s:sub(1,a-1)..body..s:sub(b)
 io.stderr:write('packet_inputs '..tag..' groups=20 stride=128\n')
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
