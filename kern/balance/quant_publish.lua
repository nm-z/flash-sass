-- GR's completed 32-column tile is already in shared memory. Six warps
-- quantize that same tile and each publishes to one peer, replacing the
-- latency-dependent six-peer pointer/load/store loop on a single warp.
-- Keep all formats, columns and exact quantization arithmetic.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)
 local i=assert(s:find(a,1,true),'missing '..a)
 assert(not s:find(a,i+#a,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(i+#a)
end
rep('ISETP.GE.U32.AND P4, PT, tid, 0x20, PT;\n--:-:1:-:1 LDS hcq_cols,',
    'ISETP.GE.U32.AND P4, PT, warp, 0x6, PT;\n--:-:1:-:1 LDS hcq_cols,')
rep('IADD hcq_index, hcq_index, tid;','IADD hcq_index, hcq_index, lane;')
rep('SHL hcq_temp, tid, 0x2;','SHL hcq_temp, lane, 0x2;')
rep('01:-:-:-:6 MOV hcq_peer, RZ;',[[01:-:-:Y:d MOV hcq_peer, warp;
--:-:-:Y:d SHL hcq_temp, warp, 0x5;
--:-:-:Y:d IADD hcq_table0.CC, hcq_table0, hcq_temp;
--:-:-:Y:d IADD.X hcq_table1, hcq_table1, RZ;]])
local first=assert(s:find('--:-:-:-:6 IADD hcq_table0.CC, hcq_table0, 0x20;',1,true))
local _,last=assert(s:find('02:-:-:-:6 NOP;\n--:-:-:-:6 BRA hcquant_END;',first,true))
s=s:sub(1,first-1)..'02:-:-:Y:d NOP;\n--:-:-:Y:d BRA hcquant_END;'..s:sub(last+1)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
