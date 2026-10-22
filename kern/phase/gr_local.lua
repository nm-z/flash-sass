-- Full-column HCUP for replicated GR. Each CTA owns a multiple of 32 columns;
-- quantize its independent 32-column groups with the original arithmetic.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,i+#a,true),'ambiguous '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',[=[    16 : hclo_offset
    17 : hclo_cols
    18 : hclo_span
    19 : hclo_base
    20 : hclo_tmp
</REGISTER_MAPPING>]=])
rep('--:-:-:-:6      MOV32I hcquant_xl, 0x20;', [=[--:-:-:-:6 IADD hcquant_xl, hcquant_gg, 0x1ff;
--:-:-:-:6 SHR.U32 hcquant_xl, hcquant_xl, 0x9;
--:-:-:-:6 SHL hcquant_xl, hcquant_xl, 0x5;]=])
local a=assert(s:find('3f:-:-:-:6 NOP;\n--:-:-:-:6 ISETP.GE.U32.AND P4, PT, tid, 0x20, PT;',1,true))
local b=assert(s:find('--:-:1:-:2 LDS.64 hcq_table0,',a,true))
local setup=[=[3f:-:-:-:6 NOP;
--:-:-:-:6 ISETP.GE.U32.AND P4, PT, tid, 0x20, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 BRA hcquant_END;
--:-:-:-:6 MOV hclo_offset, RZ;
HCLOCAL_GROUP:
--:-:1:-:1 LDS hclo_cols, [RZ+0xbfa0];
--:-:1:-:2 LDS hcq_index, [RZ+0xbfb0];
01:-:-:-:6 SHR.U32 hclo_cols, hclo_cols, 0x2;
--:-:-:-:6 IADD hclo_span, hclo_cols, 0x1ff;
--:-:-:-:6 SHR.U32 hclo_span, hclo_span, 0x9;
--:-:-:-:6 SHL hclo_span, hclo_span, 0x5;
--:-:-:-:6 XMAD hclo_base, cta, hclo_span, hclo_offset;
--:-:-:-:6 ISETP.GE.U32.AND P4, PT, hclo_base, hclo_cols, PT;
--:-:-:-:6 ISETP.GE.U32.AND P5, PT, hclo_offset, hclo_span, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 PSETP.OR.AND P4, PT, P4, P5, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 BRA hcquant_END;
--:-:-:-:6 LOP32I.AND hcq_index, hcq_index, 0xffff;
--:-:-:-:6 SHR.U32 hcq_index, hcq_index, 0x2;
--:-:-:-:6 IADD hcq_index, hcq_index, hclo_base;
--:-:-:-:6 IADD hcq_index, hcq_index, tid;
--:-:-:-:6 IADD hclo_tmp, hclo_offset, tid;
--:-:-:-:6 SHL hclo_tmp, hclo_tmp, 0x2;
--:-:1:-:2 LDS hcq_v0, [hclo_tmp+0x600];
01:-:-:-:6 NOP;
]=]
s=s:sub(1,a-1)..setup..s:sub(b)
rep('--:-:-:-:6 BRA hcquant_END;\nhcquant_END:',[=[--:-:-:-:6 IADD hclo_offset, hclo_offset, 0x20;
--:-:-:-:6 BRA HCLOCAL_GROUP;
hcquant_END:]=])
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
