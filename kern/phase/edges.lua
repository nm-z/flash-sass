-- Direct publication at the embedding and final reducer producers.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',[=[    180,181 : edge_table<0-1>
    182,183 : edge_dest<0-1>
    184 : edge_offset
    185 : edge_peer
    186 : edge_flags
</REGISTER_MAPPING>]=])
local function publication(tag,embedding,topk)
 local code=[=[
3f:-:1:-:2 LDS edge_flags, [RZ+0xbff8];
01:-:-:-:6 LOP32I.AND edge_flags, edge_flags, 0x80;
--:-:-:-:6 ISETP.EQ.U32.AND P6, PT, edge_flags, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA @_PUB_DONE;
--:-:1:-:2 LDS.64 edge_table0, [RZ+0xbfb8];
]=]
 if embedding then
  code=code..[=[--:-:-:-:6 ISCADD edge_offset, cta, tid, 0x8;
--:-:-:-:6 SHL edge_offset, edge_offset, 0x2;
]=]
 else
  code=code..'--:-:1:-:2 LDS edge_offset, [RZ+0xbfc0];\n'
  if topk then code=code..'01:-:-:-:6 ISCADD edge_offset, lane, edge_offset, 0x3;\n'
  else code=code..'--:-:-:-:6 ISETP.EQ.U32.AND P4, PT, lane, RZ, PT;\n'end
 end
 code=code..[=[01:-:-:-:6 MOV edge_peer, RZ;
@_PUB_LOOP:
--:-:1:-:2 LDG.E.64 edge_dest0, [edge_table];
01:-:-:-:6 IADD edge_dest0.CC, edge_dest0, edge_offset;
--:-:-:-:6 IADD.X edge_dest1, edge_dest1, RZ;
]=]
 if embedding then
  for _,offset in ipairs({0,0x2800,0x5000,0x7800})do code=code..string.format('--:2:-:-:1 STG.E.WT [edge_dest+0x%x], embrow_y;\n',offset)end
 else
  code=code..'--:2:-:-:1 '..(topk and '' or '@P4 ')..'STG.E.WT [edge_dest], '..(topk and 'topk_si' or 'argmax_im')..';\n'
  code=code..'--:2:-:-:1 '..(topk and '' or '@P4 ')..'STG.E.WT [edge_dest+0x4], '..(topk and 'topk_sm' or 'argmax_m')..';\n'
 end
 code=code..[=[--:-:-:-:6 IADD edge_table0.CC, edge_table0, 0x8;
--:-:-:-:6 IADD.X edge_table1, edge_table1, RZ;
--:-:-:-:6 IADD edge_peer, edge_peer, 0x1;
--:-:-:-:6 ISETP.LT.U32.AND P5, PT, edge_peer, 0x6, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
02:-:-:-:6 NOP;
--:-:-:-:6 @P5 BRA @_PUB_LOOP;
@_PUB_DONE:
]=]
 return (code:gsub('@_',tag..'_'))
end
rep('embrow_END:','embrow_END:\n'..publication('embrow',true,false))
for _,tag in ipairs({'argmax','topk'})do
 local anchor='--:1:-:-:6  @P4 STG.E ['..tag..'_cp], RZ;\n01:-:-:-:f      BRA '..tag..'_END;'
 rep(anchor,'--:1:-:-:6  @P4 STG.E ['..tag..'_cp], RZ;\n'..publication(tag,false,tag=='topk')..'--:-:-:-:6 BRA '..tag..'_END;')
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
