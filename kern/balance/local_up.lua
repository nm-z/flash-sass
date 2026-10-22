-- HCUP keeps a 32-column-aligned interval on each CTA. The same body supports
-- sharded peer output and complete local output; only the descriptor's local
-- bit changes the number of quantization destinations.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(a,b)
 local i=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,i+#a,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(i+#a)
end
rep('</REGISTER_MAPPING>',[=[    16 : hlu_offset
    17 : hlu_cols
    18 : hlu_span
    19 : hlu_base
    20 : hlu_temp
    21 : hlu_flags
    22 : hlu_peers
</REGISTER_MAPPING>]=])
rep('MOV32I hcquant_xl, 0x20;',[[IADD hcquant_xl, hcquant_gg, 0x1ff;
--:-:-:Y:d SHR.U32 hcquant_xl, hcquant_xl, 0x9;
--:-:-:Y:d SHL hcquant_xl, hcquant_xl, 0x5;]])
local first=assert(s:find('3f:-:-:-:6 NOP;\n--:-:-:-:6 ISETP.GE.U32.AND P4, PT, warp, 0x6, PT;\n--:-:1:-:1 LDS hcq_cols,',1,true))
local last=assert(s:find('--:-:1:-:2 LDS.64 hcq_table0,',first,true))
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:Y:d')..' '..x..';'end
e('LDS hlu_flags, [RZ+0xbff8]','3f:-:1:-:2');e('NOP','01:-:-:Y:d')
e('LOP32I.AND hlu_flags, hlu_flags, 0x40000');e('MOV hlu_peers, 0x6')
e('ISETP.NE.U32.AND P5, PT, hlu_flags, RZ, PT');e('@P5 MOV hlu_peers, 0x1')
e('ISETP.GE.U32.AND P4, PT, warp, hlu_peers, PT');e('@P4 BRA hcquant_END')
e('MOV hlu_offset, RZ');out[#out+1]='BALANCED_LOCAL_UP_GROUP:'
e('LDS hlu_cols, [RZ+0xbfa0]','--:-:1:-:2');e('LDS hcq_index, [RZ+0xbfb0]','--:-:1:-:2');e('NOP','01:-:-:Y:d')
e('SHR.U32 hlu_cols, hlu_cols, 0x2');e('IADD hlu_span, hlu_cols, 0x1ff')
e('SHR.U32 hlu_span, hlu_span, 0x9');e('SHL hlu_span, hlu_span, 0x5')
e('XMAD hlu_base, cta, hlu_span, hlu_offset')
e('ISETP.GE.U32.AND P4, PT, hlu_base, hlu_cols, PT')
e('ISETP.GE.U32.AND P5, PT, hlu_offset, hlu_span, PT')
e('PSETP.OR.AND P4, PT, P4, P5, PT');e('@P4 BRA hcquant_END')
e('LOP32I.AND hcq_index, hcq_index, 0xffff');e('SHR.U32 hcq_index, hcq_index, 0x2')
e('IADD hcq_index, hcq_index, hlu_base');e('IADD hcq_index, hcq_index, lane')
e('IADD hlu_temp, hlu_offset, lane');e('SHL hlu_temp, hlu_temp, 0x2')
e('LDS hcq_v0, [hlu_temp+0x600]','--:-:1:-:2');e('NOP','01:-:-:Y:d')
s=s:sub(1,first-1)..table.concat(out,'\n')..'\n'..s:sub(last)
rep('02:-:-:Y:d NOP;\n--:-:-:Y:d BRA hcquant_END;',[[02:-:-:Y:d NOP;
--:-:-:Y:d IADD hlu_offset, hlu_offset, 0x20;
--:-:-:Y:d BRA BALANCED_LOCAL_UP_GROUP;]])
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
