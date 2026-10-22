-- Complete balanced PLE gather/quantization in the existing physical phase.
-- One CTA owns one of the sixteen 160-value n-gram heads. Its first lane
-- retains the production hash/ready arithmetic; the warp cooperatively reads
-- the 90-byte IQ4_NL row, and five warps decode/quantize its five32-value
-- blocks. No cross-CTA dependency or additional exchange is introduced.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(text,a,b)
 local at=assert(text:find(a,1,true),'missing '..a)
 assert(not text:find(a,at+#a,true),'ambiguous '..a)
 return text:sub(1,at-1)..b..text:sub(at+#a)
end
local first=assert(s:find('plegatherq_SETUP:',1,true))
local last=s:find('\n[%w_]+_SETUP:',first+1) or assert(s:find('\nPOSITION_END:',first+1))
local old=s:sub(first,last-1)
local begin=assert(old:find('plegatherq_BEGIN:',1,true))
local data=assert(old:find('[^\n]*LDG%.E%.CV%.U16 plegatherq_hw00,'))
local hash=old:sub(begin,data-1)
assert(not hash:find('BAR.SYNC',1,true),'hash must not contain a divergent CTA barrier')
hash=rep(hash,'MOV plegatherq_h, vtid;','MOV plegatherq_h, cta;')
local maps={[=[
    200,201 : bple_src<0-1>
    202,203 : bple_pg<0-1>
    204 : bple_word
    205 : bple_index
    206 : bple_block
    207 : bple_shift
    208 : bple_code
    209 : bple_scale
    210 : bple_value
    211 : bple_out
    212 : bple_temp
    214,215 : bple_dest<0-1>
]=]}
local out={}
local function e(ins,ctl)out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'end
local function l(name)out[#out+1]=name..':'end
local function w(mask)e('NOP',mask..':-:-:-:6')end
l('plegatherq_SETUP')
e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA BPLE_ROW_READY')
-- Explicit reconvergence is required: only lane0 hashes the head, but all
-- lanes must resume together for the coalesced row loads and CTA rendezvous.
e('SSY BPLE_HASH_JOIN')
e('ISETP.NE.U32.AND P6, PT, tid, RZ, PT');e('@P6 BRA BPLE_HASH_SYNC')
out[#out+1]=hash
l('BPLE_HASH_SYNC');e('SYNC');l('BPLE_HASH_JOIN')
e('SHFL.IDX PT, bple_src0, plegatherq_tp0, 0x0, 0x1f','--:-:1:-:2')
e('SHFL.IDX PT, bple_src1, plegatherq_tp1, 0x0, 0x1f','--:-:2:-:2');w('03')
e('SHL bple_temp, lane, 0x1')
e('IADD bple_dest0.CC, bple_src0, bple_temp');e('IADD.X bple_dest1, bple_src1, RZ')
e('LDG.E.CV.U16 bple_word, [bple_dest]','--:-:1:-:2');w('01')
e('SHL bple_temp, lane, 0x2')
e('STS [bple_temp+0x1000], bple_word','--:2:-:-:2');w('02')
e('ISETP.LT.U32.AND P4, PT, lane, 0xd, PT')
e('@P4 LDG.E.CV.U16 bple_word, [bple_dest+0x40]','--:-:1:-:2');w('01')
e('@P4 STS [bple_temp+0x1080], bple_word','--:2:-:-:2');w('02')
l('BPLE_ROW_READY');e('BAR.SYNC 0x0','3f:-:-:-:6')
e('ISETP.GE.U32.AND P6, PT, warp, 0x5, PT');e('@P6 BRA POSITION_END')
e('LDS.64 bple_pg0, [RZ+0xbfa0]','--:-:1:-:2');w('01')
e('XMAD bple_block, warp, 0x9, RZ')
e('SHL bple_temp, bple_block, 0x2')
e('LDS bple_scale, [bple_temp+0x1000]','--:-:1:-:2')
e('LOP32I.AND bple_index, lane, 0xf');e('SHR.U32 bple_index, bple_index, 0x1')
e('IADD bple_index, bple_index, 0x1');e('IADD bple_index, bple_index, bple_block')
e('SHL bple_index, bple_index, 0x2')
e('LDS bple_word, [bple_index+0x1000]','--:-:2:-:2');w('03')
e('F2F.F32.F16 bple_scale, bple_scale.H0','--:-:3:-:2');w('04')
-- IQ4_NL stores the low nibbles for values0..15, then the high nibbles
-- for values16..31. The two adjacent bytes share one staged u16 word.
e('LOP32I.AND bple_shift, lane, 0x1');e('SHL bple_shift, bple_shift, 0x3')
e('LOP32I.AND bple_temp, lane, 0x10');e('SHR.U32 bple_temp, bple_temp, 0x2')
e('IADD bple_shift, bple_shift, bple_temp')
e('SHR.U32 bple_code, bple_word, bple_shift');e('LOP32I.AND bple_code, bple_code, 0xf')
e('SHL bple_code, bple_code, 0x2')
e('LDS bple_value, [bple_code]','--:-:1:-:2');w('01')
e('FMUL bple_value, bple_scale, bple_value')
e('XMAD bple_out, cta, 0xa0, RZ');e('IADD bple_out, bple_out, tid')
e('SHL bple_temp, bple_out, 0x2')
e('IADD bple_dest0.CC, bple_pg0, bple_temp');e('IADD.X bple_dest1, bple_pg1, RZ')
e('STG.E.WT [bple_dest], bple_value','--:2:-:-:2');w('02')
local quant=dofile('balance/../phase/quant_emit.lua')(maps)
local q=quant('bpleq',1,0xbfd0,
 '--:-:-:Y:d MOV %_v0, bple_value;\n--:-:-:Y:d MOV %_index, bple_out;',
 '--:-:-:Y:d BRA POSITION_END;')
q=rep(q,'LDS.64 bpleq_table0, [RZ+0xbfd0];','MOV bpleq_table0, RZ;')
q=rep(q,'--:-:-:-:6 MOV32I bpleq_mneg1, 0xbf800000;',
 '--:-:-:-:6 MOV bpleq_table1, RZ;\n--:-:-:-:6 MOV32I bpleq_mneg1, 0xbf800000;')
for _,p in ipairs({{'bpleq_qp0',0,0xbfd0},{'bpleq_sp320',8,0xbfd8},
                   {'bpleq_sp160',16,0xbfe0},{'bpleq_ap0',24,0xbfe8}})do
 q=rep(q,string.format('LDG.E.64 %s, [bpleq_table+0x%x];',p[1],p[2]),
         string.format('LDS.64 %s, [RZ+0x%x];',p[1],p[3]))
end
q=rep(q,'bpleq_peer, 0x6, PT;','bpleq_peer, 0x1, PT;')
out[#out+1]=q
s=s:sub(1,first-1)..table.concat(out,'\n')..'\n'..s:sub(last)
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
