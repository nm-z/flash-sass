-- Fingerprint all10240 updated residual values already in registers. No new
-- input loads, phase, CTA join, or early cache invalidation is introduced.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,i+#a,true));s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',[=[    240,241 : gw_ptr<0-1>
    242,243 : gw_window<0-1>
    242 : gw_first
    243 : gw_count
    244 : gw_token
    245 : gw_offset
    246,247 : gw_hash<0-1>
    248 : gw_tmp
</REGISTER_MAPPING>]=])
for _,op in ipairs({'bf16hcrwx','grreadsum'})do
 local tag='GW_'..op
 local code=[=[
--:-:1:-:1 LDS gw_offset, [RZ+0xbfd4];
--:-:2:-:1 LDS gw_token, [RZ+0xbf70];
01:-:-:-:6 ISETP.EQ.U32.AND P0, PT, gw_offset, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P0 BRA @_DONE;
--:-:-:-:6 MOV gw_ptr0, ctl0;
--:-:-:-:6 MOV gw_ptr1, ctl1;
--:-:1:-:2 LDG.E.64 gw_window0, [gw_ptr+0xf0];
03:-:-:-:6 IADD gw_token, gw_token, -gw_first;
--:-:-:-:6 ISETP.LT.U32.AND P0, PT, gw_token, gw_count, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P0 BRA @_DONE;
--:-:1:-:2 LDG.E.64 gw_ptr0, [gw_ptr+0xe8];
--:-:-:-:6 IADD gw_offset, gw_offset, -0x1;
--:-:-:-:6 ISCADD gw_offset, gw_token, gw_offset, 0x7;
--:-:-:-:6 SHL gw_offset, gw_offset, 0xb;
--:-:-:-:6 ISCADD gw_offset, tid, gw_offset, 0x3;
01:-:-:-:6 IADD gw_ptr0.CC, gw_ptr0, gw_offset;
--:-:-:-:6 IADD.X gw_ptr1, gw_ptr1, RZ;
--:-:-:-:6 MOV32I gw_hash0, 0x811c9dc5;
--:-:-:-:6 MOV32I gw_hash1, 0x9e3779b9;
]=]
 for i=0,39 do
  local v=string.format('%s_act%02d',op,i)
  code=code..string.format([=[--:-:-:-:6 LOP.XOR gw_hash0, gw_hash0, %s;
--:-:-:-:6 SHL gw_tmp, gw_hash0, 0x5;
--:-:-:-:6 SHR.U32 gw_hash0, gw_hash0, 0x1b;
--:-:-:-:6 LOP.OR gw_hash0, gw_hash0, gw_tmp;
--:-:-:-:6 IADD gw_hash1, gw_hash1, %s;
--:-:-:-:6 SHL gw_tmp, gw_hash1, 0x7;
--:-:-:-:6 SHR.U32 gw_hash1, gw_hash1, 0x19;
--:-:-:-:6 LOP.OR gw_hash1, gw_hash1, gw_tmp;
]=],v,v)
 end
 -- P6 selects16 lanes per CTA, not a whole warp. Keep control warp-uniform;
 -- predicate only the final store, matching the residual writer's ownership.
 code=code..'--:6:-:-:2 @P6 STG.E.64 [gw_ptr], gw_hash0;\n20:-:-:-:6 NOP;\n'..tag..'_DONE:\n'
 code=code:gsub('@_',tag..'_')
 local a='--:4:-:-:1  @P6 STG.E.128 ['..op..'_ob+0x90], '..op..'_act36;'
 rep(a,a..'\n'..code)
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
