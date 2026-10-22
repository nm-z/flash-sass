-- Four local query heads, one persistent K/V head. The compact gather keeps
-- the production blocked strides but writes/reads only head0, avoiding a new
-- floating-point data layout or dequantization order.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(x,a,b)
 local i=assert(x:find(a,1,true),'missing '..a);assert(not x:find(a,i+#a,true),'ambiguous '..a)
 return x:sub(1,i-1)..b..x:sub(i+#a)
end
local function routine(tag,edit)
 local a=assert(s:find(tag..'_SETUP:',1,true));local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
 s=s:sub(1,a-1)..edit(s:sub(a,b-1))..s:sub(b)
end
routine('attnqkq',function(b)
 b=rep(b,'attnqkq_head, 0x18, PT;','attnqkq_head, 0x4, PT;')
 b=rep(b,'attnqkq_kh, attnqkq_head, -0x18;','attnqkq_kh, attnqkq_head, -0x4;')
 b=rep(b,'SHL attnqkq_ee, attnqkq_ss, 0x9;','SHL attnqkq_ee, attnqkq_ss, 0x8;')
 b=rep(b,'SHL attnqkq_ee, attnqkq_ss, 0x5;','SHL attnqkq_ee, attnqkq_ss, 0x4;')
 return b
end)
routine('kvgather',function(b)
 b=rep(b,'SHL kvgather_tmp, kvgather_src, 0xb;','SHL kvgather_tmp, kvgather_src, 0xa;')
 b=rep(b,'SHL kvgather_tmp, kvgather_src, 0x7;','SHL kvgather_tmp, kvgather_src, 0x6;')
 local points={};local at=1
 while true do local i,j=b:find('[^\n]*MOV kvgather_kd0, RZ;',at);if not i then break end
  points[#points+1]=i;at=j+1
 end
 assert(#points==8,'gather head/block geometry '..#points)
 local ending=assert(b:find('kvgather_END:',points[8],true))
 local out={b:sub(1,points[1]-1)}
 for j=0,3 do
  -- The expanded production loop is KV-major: head0 tokens0..3, then
  -- head1 tokens0..3. Taking alternating blocks drops odd tokens and keeps
  -- half of head1. Retain all four consecutive head0 blocks instead.
  local a=points[j+1];local last=points[j+2]-1
  local block=b:sub(a,last)
  assert(block:find(string.format('[kvgather_kdp+0x%x]',j*512),1,true),
   'unexpected gather source order for token '..j)
  assert(block:find(string.format('[kvgather_kcp+0x%x]',j*16),1,true),
   'unexpected gather destination head/order for token '..j)
  -- The previous unit's tail predicate is immaterial; establish this exact key.
  local insert=assert(block:find('[^\n]*@!P5 LDG%.E'))
  block=block:sub(1,insert-1)..string.format([=[
--:-:-:Y:d ISETP.EQ.U32.AND P5, PT, kvgather_unit, kvgather_ns, PT;
--:-:-:Y:d ISETP.LE.U32.AND P5, PT, kvgather_tail, 0x%x, P5;
]=],j)..block:sub(insert)
  for _,ptr in ipairs({'kdp','vdp'})do
   block=block:gsub('%[kvgather_'..ptr..'%+0x(%x+)%]',function(h)
    return string.format('[kvgather_%s+0x%x]',ptr,tonumber(h,16)//2)
   end)
  end
  for _,ptr in ipairs({'ksp','vsp'})do
   block=block:gsub('%[kvgather_'..ptr..'%+0x(%x+)%]',function(h)
    return string.format('[kvgather_%s+0x%x]',ptr,tonumber(h,16)//2)
   end)
  end
  out[#out+1]=block
 end
 out[#out+1]=b:sub(ending)
 return table.concat(out,'\n')
end)
routine('attnpartn',function(b)
 b=rep(b,'SHR.U32 attnpartn_tmp, attnpartn_cta, 0x2;','SHR.U32 attnpartn_tmp, attnpartn_cta, 0x1;')
 b=rep(b,'LOP32I.AND attnpartn_u3, attnpartn_cta, 0x1;','MOV attnpartn_u3, RZ;')
 local a=assert(b:find('[^\n]*SHR%.U32 attnpartn_u2, attnpartn_cta, 0x1;'))
 local _,z=assert(b:find('[^\n]*BAR%.SYNC 0x0;',a))
 local copy=[=[
--:-:-:Y:d LOP32I.AND attnpartn_u1, attnpartn_cta, 0x1;
--:-:-:Y:d SHL attnpartn_u1, attnpartn_u1, 0xb;
--:-:-:Y:d SHL attnpartn_u2, attnpartn_tid, 0x4;
--:-:-:Y:d IADD attnpartn_u1, attnpartn_u1, attnpartn_u2;
--:-:-:Y:d IADD attnpartn_qp0.CC, attnpartn_u1, attnpartn_param_QO0;
--:-:-:Y:d IADD.X attnpartn_qp1, RZ, attnpartn_param_QO1;
--:-:-:Y:d ISETP.LT.U32.AND P6, PT, attnpartn_tid, 0x80, PT;
--:-:1:-:2 @P6 LDG.E.CG.128 attnpartn_q00, [attnpartn_qp];
01:1:-:-:2 @P6 STS.128 [attnpartn_u2], attnpartn_q00;
01:-:-:-:6 BAR.SYNC 0x0;
]=]
 b=b:sub(1,a-1)..copy..b:sub(z+1)
 local final=assert(b:find('attnpartn_FINAL:',1,true))
 a=assert(b:find('[^\n]*LOP32I%.AND attnpartn_u1, attnpartn_cta, 0x1;',final))
 z=assert(b:find('[^\n]*@P0 STG%.E%.64 %[attnpartn_pop',a))
 local pointers=[=[
--:-:-:Y:d LOP32I.AND attnpartn_u1, attnpartn_cta, 0x1;
--:-:-:Y:d SHL attnpartn_u1, attnpartn_u1, 0x1;
--:-:-:Y:d SHR.U32 attnpartn_u3, attnpartn_cta, 0x1;
--:-:-:Y:d SHL attnpartn_pop0, attnpartn_u1, 0xd;
--:-:-:Y:d ISCADD attnpartn_pop0, attnpartn_u3, attnpartn_pop0, 0xa;
--:-:-:Y:d SHR.U32 attnpartn_tmp, attnpartn_tid, 0x2;
--:-:-:Y:d ISCADD attnpartn_pop0, attnpartn_tmp, attnpartn_pop0, 0x4;
--:-:-:Y:d IADD attnpartn_pop0.CC, attnpartn_pop0, attnpartn_param_PO0;
--:-:-:Y:d IADD.X attnpartn_pop1, RZ, attnpartn_param_PO1;
--:-:-:Y:d SHL attnpartn_pmp0, attnpartn_u1, 0x5;
--:-:-:Y:d ISCADD attnpartn_pmp0, attnpartn_u3, attnpartn_pmp0, 0x2;
--:-:-:Y:d IADD attnpartn_pdp0.CC, attnpartn_pmp0, attnpartn_param_PD0;
--:-:-:Y:d IADD.X attnpartn_pdp1, RZ, attnpartn_param_PD1;
--:-:-:Y:d IADD attnpartn_pmp0.CC, attnpartn_pmp0, attnpartn_param_PM0;
--:-:-:Y:d IADD.X attnpartn_pmp1, RZ, attnpartn_param_PM1;
]=]
 b=b:sub(1,a-1)..pointers..b:sub(z)
 return dofile('balance/attention_heads.lua')(b)
end)
-- Merge dequantization/gating remains local-head indexed; publication is global.
routine('attnquant',function(b)
 return rep(b,'ISCADD attnq_index, vid, tid, 0x8;', [=[ISCADD attnq_index, vid, tid, 0x8;
--:-:1:-:2 LDS attnq_cols, [RZ+0xbfb8];
01:-:-:Y:d IADD attnq_index, attnq_index, attnq_cols;]=])
end)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
