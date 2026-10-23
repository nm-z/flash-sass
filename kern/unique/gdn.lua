-- Each key head and its three value heads have one matrix owner. Recurrent
-- state remains local-head-major; only Q/K addressing and final publication
-- map the grouped local value heads back to global model order.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(old,new)
 local a,b=assert(s:find(old,1,true));assert(not s:find(old,b+1,true),'ambiguous '..old)
 s=s:sub(1,a-1)..new..s:sub(b+1)
end
rep('</REGISTER_MAPPING>',[=[    240 : ug_keys
    241 : ug_one
    242 : ug_diff
    243 : ug_group
</REGISTER_MAPPING>]=])
local function group(head)
 return string.format([=[--:-:1:-:2 LDS ug_keys, [RZ+0xbff4];
01:-:-:-:6 SHR.U32 ug_keys, ug_keys, 0x8;
--:-:-:-:6 MOV ug_one, 0x1;
--:-:-:-:6 IADD ug_diff, %s, -ug_keys;
--:-:-:Y:d ICMP.GE ug_group, ug_one, RZ, ug_diff;
--:-:-:-:6 IADD ug_diff, ug_diff, -ug_keys;
--:-:-:Y:d ICMP.GE ug_diff, ug_one, RZ, ug_diff;
--:-:-:-:6 IADD ug_group, ug_group, ug_diff;
]=],head)
end
for _,tag in ipairs({'dnlocal','dnstep'})do
 rep('--:-:-:-:6      LOP32I.AND '..tag..'_kh, '..tag..'_head, 0xf;',
  group(tag..'_head')..'--:-:-:-:6 XMAD ug_diff, ug_group, ug_keys, RZ;\n'..
  '--:-:-:-:6 IADD '..tag..'_kh, '..tag..'_head, -ug_diff;')
end
for _,tag in ipairs({'dnquant','dnlocalnorm','dnnormx'})do
 local old='--:-:-:-:6      IADD '..tag..'_xo, '..tag..'_t, '..tag..'_param_xoff;'
 rep(old,group(tag..'_head')..string.format([=[--:-:-:-:6 MOV ug_one, 0x10;
--:-:-:-:6 IADD ug_one, ug_one, -ug_keys;
--:-:-:-:6 XMAD ug_diff, ug_group, ug_one, RZ;
--:-:-:-:6 SHL ug_diff, ug_diff, 0x9;
--:-:-:-:6 IADD %s_xo, %s_t, %s_param_xoff;
--:-:-:-:6 IADD %s_xo, %s_xo, ug_diff;]=],tag,tag,tag,tag,tag))
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
