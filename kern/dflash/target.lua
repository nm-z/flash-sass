-- Retained prefix state and target-feature capture in the production image.
-- Stateful operators resolve the selected input bank on the device. GDN writes
-- each new tile straight to its output bank; small rings copy only the channels
-- owned by the thread/warp that will update them, within that same operation.
local function read(path)local f=assert(io.open(path));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2])
local function replace(a,b)
 local i,j=assert(s:find(a,1,true),'missing '..a)
 assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local function gen()
 local o={}
 local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function label(x)o[#o+1]=x..':'end
 local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
 return o,e,label,pred
end
replace('</REGISTER_MAPPING>',[=[
    232,233 : dfbank_table<0-1>
    234,235 : dfbank_select<0-1>
    236,237 : dfbank_out<0-1>
    238,239 : dfbank_in<0-1>
    240 : dfbank_offset
    241 : dfbank_count
    242 : dfbank_index
    244-247 : dfbank_value<0-3>
    16-19 : dffeature_value<0-3>
    20,21 : dffeature_src<0-1>
    22,23 : dffeature_dst<0-1>
    24 : dffeature_position
    25 : dffeature_tap
    26 : dffeature_index
    27 : dffeature_offset
</REGISTER_MAPPING>]=])
-- Resolve an optional input table. The current parameter is the direct input
-- for positions 1..3 and for the ordinary scalar path.
local function resolve(e,l,pred,tag,param)
 e('LDS.64 dfbank_table0, [RZ+0xbfe8]','3f:-:1:-:2')
 e('LOP.OR dfbank_index, dfbank_table0, dfbank_table1','01:-:-:-:6')
 pred('ISETP.EQ.U32.AND P5, PT, dfbank_index, RZ, PT')
 e('@P5 BRA '..tag..'_DIRECT','--:-:-:Y:d')
 e('LDG.E.CG.64 dfbank_select0, [dfbank_table]','--:-:1:-:2');e('NOP','01:-:-:-:6')
 e('LDG.E.CV dfbank_index, [dfbank_select]','--:-:1:-:2');e('NOP','01:-:-:-:6')
 e('ISCADD dfbank_offset, dfbank_index, 0x8, 0x3')
 e('IADD dfbank_table0.CC, dfbank_table0, dfbank_offset');e('IADD.X dfbank_table1, dfbank_table1, RZ')
 e('LDG.E.CG.64 dfbank_in0, [dfbank_table]','--:-:1:-:2');e('NOP','01:-:-:-:6')
 e('MOV '..param..'0, dfbank_in0');e('MOV '..param..'1, dfbank_in1')
 l(tag..'_DIRECT')
end
-- Only the unfused DNSTEP carries bank fields: the host deliberately preserves
-- it while its checkpoint/output pointer is nonzero. Normal scalar fusion is
-- unchanged and never consumes these descriptor words.
do
 local o,e,l,pred=gen();resolve(e,l,pred,'DF_DN_INPUT','dnstep_param_S')
 replace('--:-:-:-:6      MOV dnstep_tid, vtid;',table.concat(o,'\n')..'\n--:-:-:-:6      MOV dnstep_tid, vtid;')
 o,e,l,pred=gen()
 e('LDS.64 dfbank_out0, [RZ+0xbfe0]','3f:-:1:-:2')
 e('LOP.OR dfbank_index, dfbank_out0, dfbank_out1','01:-:-:-:6')
 pred('ISETP.EQ.U32.AND P0, PT, dfbank_index, RZ, PT');e('@P0 BRA DF_DN_OUTPUT_READY','--:-:-:Y:d')
 e('SHL dfbank_offset, dnstep_head, 0x10');e('SHL dfbank_index, dnstep_g, 0xb')
 e('IADD dfbank_offset, dfbank_offset, dfbank_index');e('ISCADD dfbank_offset, dnstep_p, dfbank_offset, 0x6')
 e('IADD dnstep_sp0.CC, dfbank_out0, dfbank_offset');e('IADD.X dnstep_sp1, dfbank_out1, RZ')
 l('DF_DN_OUTPUT_READY')
 local anchor='--:2:-:-:1      STG.E.128 [dnstep_sp+0x0], dnstep_s00;'
 replace(anchor,table.concat(o,'\n')..'\n'..anchor)
end
local function ring(tag,param,slots,kind)
 local o,e,l,pred=gen();local label='DF_BANK_'..tag
 e('LDS.64 dfbank_out0, [RZ+0xbfe0]','3f:-:1:-:2')
 e('LOP.OR dfbank_index, dfbank_out0, dfbank_out1','01:-:-:-:6')
 pred('ISETP.EQ.U32.AND P5, PT, dfbank_index, RZ, PT');e('@P5 BRA '..label..'_DONE','--:-:-:Y:d')
 resolve(e,l,pred,label,param)
 e('MOV dfbank_in0, '..param..'0');e('MOV dfbank_in1, '..param..'1')
 e('MOV '..param..'0, dfbank_out0');e('MOV '..param..'1, dfbank_out1')
 if kind=='index' then
  pred('ISETP.NE.U32.AND P5, PT, vid, 0x4, PT');e('@P5 BRA '..label..'_DONE','--:-:-:Y:d')
  e('SHL dfbank_offset, vtid, 0x4');e('MOV dfbank_count, 0x200')
 else
  e('ISCADD dfbank_offset, vid, vtid, 0x8');e('SHL dfbank_offset, dfbank_offset, 0x4')
  if kind=='conv' then
   e('SHL dfbank_count, conv1d_param_n, 0x2')
   pred('ISETP.GE.U32.AND P5, PT, dfbank_offset, dfbank_count, PT')
   e('@P5 BRA '..label..'_DONE','--:-:-:Y:d')
  else e('MOV dfbank_count, 0xa000')end
 end
 e('IADD dfbank_in0.CC, dfbank_in0, dfbank_offset');e('IADD.X dfbank_in1, dfbank_in1, RZ')
 e('IADD dfbank_out0.CC, dfbank_out0, dfbank_offset');e('IADD.X dfbank_out1, dfbank_out1, RZ')
 for i=1,slots do
  e('LDG.E.CG.128 dfbank_value0, [dfbank_in]','--:2:1:-:2')
  e('STG.E.128 [dfbank_out], dfbank_value0','01:3:-:-:2');e('NOP','06:-:-:-:6')
  if i<slots then
   e('IADD dfbank_in0.CC, dfbank_in0, dfbank_count');e('IADD.X dfbank_in1, dfbank_in1, RZ')
   e('IADD dfbank_out0.CC, dfbank_out0, dfbank_count');e('IADD.X dfbank_out1, dfbank_out1, RZ')
  end
 end
 e('MEMBAR.GL.IVALLD','3f:-:-:Y:d');l(label..'_DONE')
 local a,b=assert(s:find('MOV '..tag..'_tid, vtid;',1,true))
 local start=(s:sub(1,a-1):match('.*()\n') or 0)+1
 s=s:sub(1,start-1)..table.concat(o,'\n')..'\n'..s:sub(start)
end
ring('conv1d','conv1d_param_S',4,'conv')
ring('pleconv','pleconv_param_ST',10,'ple')
ring('idxqk','idxqk_param_RING',4,'index')
ring('bindex','bindex_param_RING',4,'index')
-- Feature = mean of four residual branches, after the completed model block.
-- Descriptor: S, ring, absolute token (+TOK), tap. Only the draft owner runs.
local o,e,l,pred=gen();l('dffeature_SETUP');e('NOP','3f:-:-:-:6')
pred('ISETP.GE.U32.AND P6, PT, cta, units, PT');e('@P6 BRA POSITION_END','--:-:-:Y:d')
e('LDS.64 dffeature_src0, [RZ+0xbf88]','--:-:1:-:2')
e('LDS.64 dffeature_dst0, [RZ+0xbf90]','--:-:1:-:2')
e('LDS dffeature_position, [RZ+0xbf98]','--:-:1:-:2')
e('LDS dffeature_tap, [RZ+0xbf9c]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('ISCADD dffeature_index, cta, tid, 0x8');e('SHL dffeature_index, dffeature_index, 0x2')
e('IADD dffeature_src0.CC, dffeature_src0, dffeature_index');e('IADD.X dffeature_src1, dffeature_src1, RZ')
for j=0,3 do e(string.format('LDG.E.CG dffeature_value%d, [dffeature_src+0x%x]',j,j*10240),'--:-:1:-:2')end
e('LOP32I.AND dffeature_position, dffeature_position, 0x7ff')
e('XMAD dffeature_offset, dffeature_position, 0xc800, dffeature_index')
e('XMAD dffeature_offset, dffeature_tap, 0x2800, dffeature_offset')
e('IADD dffeature_dst0.CC, dffeature_dst0, dffeature_offset');e('IADD.X dffeature_dst1, dffeature_dst1, RZ')
e('FADD dffeature_value0, dffeature_value0, dffeature_value1','01:-:-:-:6')
e('FADD dffeature_value0, dffeature_value0, dffeature_value2')
e('FADD dffeature_value0, dffeature_value0, dffeature_value3');e('FMUL dffeature_value0, dffeature_value0, 0.25')
e('STG.E.WT [dffeature_dst], dffeature_value0','--:2:-:-:2');e('NOP','02:-:-:-:6');e('BRA POSITION_END','--:-:-:Y:d')
local count=tonumber(assert(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;')),16)
assert(count<=113,'feature opcode collision')
local jump={};for op=count,113 do
 jump[#jump+1]='--:-:-:-:6 BRA '..(op==113 and 'dffeature_SETUP' or 'POSITION_END')..';\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;'
end
replace(string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count),'ISETP.GE.U32.AND P6, PT, op, 0x72, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',table.concat(jump,'\n')..'\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'w'));f:write(s);f:close()
