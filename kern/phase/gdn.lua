-- Own convolution, recurrent state, RMS, gate and publication in one CTA/head.
-- The second descriptor supplies convolution parameters; it is data, not a
-- dispatched sub-operation. There are no inner grid barriers in this operator.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local function rep(x,a,b)local i=assert(x:find(a,1,true),'missing '..a);return x:sub(1,i-1)..b..x:sub(i+#a)end
local maps={}
local function mapping(src,a,b)
 for line in assert(src:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(a..'_',1,true)then maps[#maps+1]=line:gsub(a,b)end
 end
end
for _,x in ipairs({'dnlocal','dnlocalnorm','dnlocalq'})do mapping(s,x,x:gsub('dnlocal','gdn'))end
mapping(base,'conv1d','gdnconv')
maps[#maps+1]=[=[    16,17 : gdnfetch_ptr<0-1>
    18 : gdnfetch_offset
    19 : gdnfetch_word
    20 : gdnfetch_mask
    21 : gdnfetch_tok
]=]
local conv=assert(base:match('(conv1d_BEGIN:.-)\nconv1d_END:')):gsub('conv1d','gdnconv')
conv=conv:gsub('%[RZ%+0x(%x+)%]',function(h)local n=tonumber(h,16);return string.format('[RZ+0x%x]',n>=0xbf80 and n<=0xbfff and n-0xbf80+0x2800 or n)end)
conv=rep(conv,'20:-:-:-:6      ISCADD gdnconv_tg, gdnconv_bid, gdnconv_tid, 0x8;', [=[
20:-:-:-:6 ISETP.GE.U32.AND P6, PT, tid, 0x60, PT;
--:-:-:-:6 SHR.U32 gdnconv_tg, tid, 0x5;
--:-:-:-:6 XMAD gdnconv_tg, gdnconv_tg, units, vid;
--:-:-:-:6 ISCADD gdnconv_tg, gdnconv_tg, lane, 0x5;
--:-:-:-:6 @P6 BRA gdnconv_END;]=])
conv=rep(conv,'--:2:-:-:1      STG.E.128 [gdnconv_up], gdnconv_a0;', [=[
--:2:-:-:1 STG.E.128 [gdnconv_up], gdnconv_a0;
--:-:-:-:6 SHL gdnconv_t, tid, 0x4;
--:2:-:-:1 STS.128 [gdnconv_t+0x3000], gdnconv_a0;]=])
local dn=assert(s:match('(dnlocal_SETUP:.-)\nPOSITION_END:')):gsub('dnlocal','gdn')
-- dnlocal is the last appended body before POSITION_END in local.lua.
local prelude=[=[
--:-:-:-:6 MOV vtid, tid;
--:-:-:-:6 IADD gdnfetch_offset, pidx, 0x1;
--:-:-:-:6 SHL gdnfetch_offset, gdnfetch_offset, 0x7;
--:-:-:-:6 IADD gdnfetch_ptr0.CC, prog0, gdnfetch_offset;
--:-:-:-:6 IADD.X gdnfetch_ptr1, prog1, RZ;
--:-:-:-:6 SHL gdnfetch_offset, lane, 0x2;
--:-:-:-:6 IADD gdnfetch_ptr0.CC, gdnfetch_ptr0, gdnfetch_offset;
--:-:-:-:6 IADD.X gdnfetch_ptr1, gdnfetch_ptr1, RZ;
--:-:-:-:6 ISETP.LT.U32.AND P6, PT, tid, 0x20, PT;
--:-:1:-:2 LDS gdnfetch_tok, [RZ+0xbf70];
--:-:-:-:6 NOP;
--:-:2:-:2 @P6 LDG.E gdnfetch_word, [gdnfetch_ptr];
02:-:-:-:6 ISETP.EQ.U32.AND P5, PT, lane, 0xa, P6;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
01:-:-:-:6 NOP;
--:-:-:-:6 @P5 IADD gdnfetch_word, gdnfetch_word, gdnfetch_tok;
--:2:-:-:6 @P6 STS [gdnfetch_offset+0x2800], gdnfetch_word;
02:-:-:-:6 BAR.SYNC 0x0;
]=]..conv..'\n'..[=[
gdnconv_END:
3f:-:-:-:6 BAR.SYNC 0x0;
]=]
dn=rep(dn,'gdn_BEGIN:', 'gdn_BEGIN:\n'..prelude)
dn=dn:gsub('LDG%.E%.128 gdn_x0, %[gdn_vp%];','LDS.128 gdn_x0, [gdn_smaddr+0x3000];')
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=rep(s,'ISETP.GE.U32.AND P6, PT, op, 0x63, PT;','ISETP.GE.U32.AND P6, PT, op, 0x64, PT;')
s=rep(s,'--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA gdn_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
s=rep(s,'POSITION_END:',dn..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
