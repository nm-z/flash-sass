-- Publish GEMV output from the CTA that produced each row. The original XSEND
-- grid phase disappears; only a local rendezvous precedes these remote stores.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
replace('</REGISTER_MAPPING>',[=[
    16,17 : pub_src<0-1>
    18,19 : pub_tab<0-1>
    20,21 : pub_dst<0-1>
    22 : pub_row
    23 : pub_rows
    24 : pub_offset
    25 : pub_value
    26 : pub_peer
    27 : pub_tmp
    28 : pub_flags
    29 : pub_step
    30 : pub_byte
    31 : pub_base
</REGISTER_MAPPING>]=])
local tags={}
for _,tag in ipairs({'q3kk6144','q4kk6144','q5kk6144','iq4k6144','q2moedn','q2acc640',
 'bf16k2560','q2k2560','q3kk2560','q4kk2560','q5kk2560','q6kk2560','iq4k2560',
 'q6kk1536acc','b32q40acc','b32q50acc','b32q80acc','b32iq4nlacc'})do
 local shift=assert(s:match('SHL '..tag..'_rowstep, '..tag..'_rowstep, 0x(%x+);'))
 tags[#tags+1]={tag,tonumber(shift,16)}
end
for _,item in ipairs(tags)do
 local tag,shift=item[1],item[2]
 local body=([=[
3f:-:-:-:6 BAR.SYNC 0x0;
--:-:1:-:2 LDS pub_flags, [RZ+0xbff8];
01:-:-:-:6 LOP32I.AND pub_flags, pub_flags, 0x80;
--:-:-:-:6 ISETP.EQ.U32.AND P6, PT, pub_flags, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA @_SEND_DONE;
--:-:-:-:6 MEMBAR.GL;
--:-:1:-:1 LDS pub_rows, [RZ+0xbfa8];
--:-:1:-:2 LDS pub_offset, [RZ+0xbfe0];
--:-:-:-:6 SHR.U32 pub_row, tid, 0x%GROUP%;
--:-:-:-:6 ISCADD pub_row, cta, pub_row, 0x%SHIFT%;
--:-:-:-:6 LOP32I.AND pub_tmp, tid, 0x%MASK%;
--:-:-:-:6 ISETP.EQ.U32.AND P4, PT, pub_tmp, RZ, PT;
--:-:-:-:6 MOV32I pub_step, 0x%STEP%;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 SHL pub_base, cta, 0x%SHIFT%;
01:-:-:-:6 NOP;
@_SEND_ROW:
--:-:-:-:6 ISETP.GE.U32.AND P6, PT, pub_base, pub_rows, PT;
--:-:1:-:1 LDS.64 pub_src0, [RZ+0xbfa0];
--:-:1:-:2 LDS.64 pub_tab0, [RZ+0xbfd8];
--:-:-:-:6 SHL pub_byte, pub_row, 0x2;
--:-:-:-:6 @P6 BRA @_SEND_DONE;
01:-:-:-:6 ISETP.LT.U32.AND P2, PT, pub_row, pub_rows, P4;
--:-:-:-:6 IADD pub_src0.CC, pub_src0, pub_byte;
--:-:-:-:6 IADD.X pub_src1, pub_src1, RZ;
--:-:-:-:6 NOP;
--:-:1:-:2 @P2 LDG.E.CI pub_value, [pub_src];
--:-:-:-:6 IADD pub_byte, pub_byte, pub_offset;
--:-:-:-:6 MOV pub_peer, RZ;
@_SEND_PEER:
--:-:3:-:2 LDG.E.64 pub_dst0, [pub_tab];
04:-:-:-:6 IADD pub_dst0.CC, pub_dst0, pub_byte;
--:-:-:-:6 IADD.X pub_dst1, pub_dst1, RZ;
01:2:-:-:2 @P2 STG.E.WT [pub_dst], pub_value;
--:-:-:-:6 IADD pub_tab0.CC, pub_tab0, 0x8;
--:-:-:-:6 IADD.X pub_tab1, pub_tab1, RZ;
--:-:-:-:6 IADD pub_peer, pub_peer, 0x1;
--:-:-:-:6 ISETP.LT.U32.AND P6, PT, pub_peer, 0x6, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
02:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA @_SEND_PEER;
--:-:-:-:6 IADD pub_row, pub_row, pub_step;
--:-:-:-:6 IADD pub_base, pub_base, pub_step;
--:-:-:-:6 BRA @_SEND_ROW;
@_SEND_DONE:
]=]):gsub('@_',tag..'_'):gsub('%%GROUP%%',string.format('%x',8-shift)):gsub('%%SHIFT%%',string.format('%x',shift)):gsub('%%MASK%%',string.format('%x',(1<<(8-shift))-1)):gsub('%%STEP%%',string.format('%x',16<<shift))
 if tag=='bf16k2560' then
  body=body:gsub('LDS pub_rows, %[RZ%+0xbfa8%];','LDS pub_rows, [RZ+0xbfa0];')
  body=body:gsub('LDS%.64 pub_src0, %[RZ%+0xbfa0%];','LDS.64 pub_src0, [RZ+0xbf98];')
 end
 -- Replace only label, leaving the existing CTA epilogue to rejoin all warps.
 replace(tag..'_END:',tag..'_END:\n'..body)
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
