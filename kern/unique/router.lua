-- Disjoint router rows -> local top10 -> one small peer packet per die.
-- The next routed-gate phase merges60 candidates on CTA0 and releases its
-- local consumers. All ten selected experts remain present on every die as
-- row shards; no extra peer exchange is needed for the final routing decision.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(old,new)
 local a,b=assert(s:find(old,1,true));assert(not s:find(old,b+1,true),'ambiguous '..old)
 s=s:sub(1,a-1)..new..s:sub(b+1)
end
local maps={}
for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('routerproj_',1,true)then maps[#maps+1]=line:gsub('routerproj_','urproj_')end
end
maps[#maps+1]=[=[
    16-18 : ur_v<0-2>
    20-22 : ur_id<0-2>
    24 : ur_max
    25 : ur_selected
    26 : ur_difference
    27 : ur_candidate
    28 : ur_other
    29 : ur_neginf
    30 : ur_invalid
    32,33 : ur_address<0-1>
    34,35 : ur_logits<0-1>
    36,37 : ur_peer<0-1>
    38,39 : ur_targets<0-1>
    40 : ur_count
    41 : ur_first
    42 : ur_die
    43 : ur_index
    44 : ur_offset
    45 : ur_word
    46 : ur_gate
    47 : ur_index2
    48 : ur_exp
    49 : ur_sum
    50 : ur_inverse
    51 : ur_error
    52 : ur_log2e
    53 : ur_negative_one
    54 : ur_value
    56,57 : ur_output<0-1>
    232,233 : ur_ready<0-1>
    234,235 : ur_config<0-1>
    236 : ur_epoch
    237 : ur_layer
    238 : ur_flags
    239 : ur_temp
]=]
local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function label(x)out[#out+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
local function address(dst,base,offset)
 e('IADD '..dst..'0.CC, '..base..'0, '..offset);e('IADD.X '..dst..'1, '..base..'1, RZ')
end
local function top10(values,shared)
 e('MOV32I ur_neginf, 0xff800000');e('MOV32I ur_invalid, 0xffffffff')
 pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT')
 for rank=0,9 do
  e('MOV ur_max, ur_v0')
  for j=1,values-1 do e('FMNMX ur_max, ur_max, ur_v'..j..', !PT')end
  for _,distance in ipairs({16,8,4,2,1})do
   e(string.format('SHFL.BFLY PT, ur_other, ur_max, 0x%x, 0x1f',distance),'--:-:5:-:2')
   e('FMNMX ur_max, ur_max, ur_other, !PT','10:-:-:-:6')
  end
  e('MOV ur_selected, ur_invalid')
  for j=0,values-1 do
   e('LOP.XOR ur_difference, ur_v'..j..', ur_max')
   e('ICMP.EQ ur_candidate, ur_id'..j..', ur_invalid, ur_difference','--:-:-:Y:d')
   e('IMNMX.U32 ur_selected, ur_selected, ur_candidate, PT')
  end
  for _,distance in ipairs({16,8,4,2,1})do
   e(string.format('SHFL.BFLY PT, ur_other, ur_selected, 0x%x, 0x1f',distance),'--:-:5:-:2')
   e('IMNMX.U32 ur_selected, ur_selected, ur_other, PT','10:-:-:-:6')
  end
  e(string.format('@P0 STS [RZ+0x%x], ur_selected',shared+rank*4),'--:2:-:-:2')
  e(string.format('@P0 STS [RZ+0x%x], ur_max',shared+40+rank*4),'--:2:-:-:2')
  e('NOP','02:-:-:-:6')
  for j=0,values-1 do
   e('IADD ur_difference, ur_id'..j..', -ur_selected')
   e('ICMP.EQ ur_v'..j..', ur_neginf, ur_v'..j..', ur_difference','--:-:-:Y:d')
  end
 end
end
-- Keep the known BF16 projection and local producer-arrival protocol. The
-- consumer tail is replaced by the local shortlist and parallel peer stores.
local a=assert(s:find('routerproj_SETUP:',1,true))
local b=assert(s:find('--:-:-:-:6 IADD routejoin_tmp, pidx, 0x1;',a,true))
local projection=s:sub(a,b-1):gsub('routerproj_','urproj_'):gsub('ROUTE_JOIN','UR_PROJECTION_JOIN')
projection=projection:gsub('@P6 BRA POSITION_END;','@P6 BRA UR_LOCAL_JOIN;')
-- The setup's inactive-CTA branch is irrelevant for the required16-CTA shape;
-- it nevertheless joins the same CTA epilogue rather than bypassing it.
out[#out+1]=projection
e('LDS.64 ur_logits0, [RZ+0xbf98]','--:-:1:-:2')
e('LDS ur_count, [RZ+0xbfa0]','--:-:1:-:2')
e('LDS ur_first, [RZ+0xbfb8]','--:-:1:-:2')
e('LDS ur_die, [RZ+0xbfbc]','--:-:1:-:2');e('NOP','01:-:-:-:6')
for j=0,2 do
 e('MOV32I ur_v'..j..', 0xff800000')
 e(string.format('IADD ur_index, lane, 0x%x',32*j))
 e('IADD ur_id'..j..', ur_index, ur_first')
 pred('ISETP.LT.U32.AND P4, PT, ur_index, ur_count, PT')
 pred('ISETP.LT.U32.AND P4, PT, ur_id'..j..', 0x200, P4')
 e('SHL ur_offset, ur_index, 0x2');address('ur_address','ur_logits','ur_offset')
 e('@P4 LDG.E.CG ur_v'..j..', [ur_address]','--:-:1:-:2');e('NOP','01:-:-:-:6')
end
top10(3,0xb200)
e('MOV ur_gate, RZ');e('MOV32I ur_offset, 0x200');e('IADD ur_offset, ur_offset, -ur_first')
e('SHL ur_offset, ur_offset, 0x2');address('ur_address','ur_logits','ur_offset')
pred('ISETP.EQ.U32.AND P4, PT, ur_die, 0x5, PT')
e('@P4 LDG.E.CG ur_gate, [ur_address]','--:-:1:-:2')
e('@P0 STS [RZ+0xb250], ur_gate','01:2:-:-:2');e('NOP','02:-:-:-:6')
label('UR_LOCAL_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d')
pred('ISETP.NE.U32.AND P6, PT, cta, RZ, PT');e('@P6 BRA POSITION_END','--:-:-:Y:d')
pred('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT');e('@P6 BRA UR_LOCAL_DONE','--:-:-:Y:d')
e('LDS.64 ur_targets0, [RZ+0xbfb0]','--:-:1:-:2')
e('LDS ur_die, [RZ+0xbfbc]','--:-:1:-:2')
e('SHL ur_offset, warp, 0x3');e('NOP','01:-:-:-:6')
address('ur_address','ur_targets','ur_offset')
e('LDG.E.CG.64 ur_peer0, [ur_address]','--:-:1:-:2')
e('SHL ur_index, lane, 0x2');e('ISCADD ur_offset, ur_die, ur_index, 0x7')
e('NOP','01:-:-:-:6');address('ur_peer','ur_peer','ur_offset')
pred('ISETP.LT.U32.AND P4, PT, lane, 0x15, PT')
e('@P4 LDS ur_word, [ur_index+0xb200]','--:-:1:-:2')
e('@P4 STG.E.WT [ur_peer], ur_word','01:2:-:-:2');e('NOP','02:-:-:-:6')
label('UR_LOCAL_DONE');e('BAR.SYNC 0x0','3f:-:-:Y:d');e('BRA POSITION_END','--:-:-:Y:d')
-- Prefix for the first gate projection. CTA0 copies the six shortlists with
-- six warps, merges on warp0 and publishes ordinary local routing buffers.
-- Other CTAs acquire one local epoch before entering the existing matrix body.
local local_body=table.concat(out,'\n');out={}
e('LDS ur_flags, [RZ+0xbff8]','3f:-:1:-:2');e('NOP','01:-:-:-:6')
e('LOP32I.AND ur_flags, ur_flags, 0x4000000')
pred('ISETP.EQ.U32.AND P6, PT, ur_flags, RZ, PT')
e('@P6 BRA UR_GATE_BEGIN','--:-:-:Y:d')
e('LDS.64 ur_ready0, [RZ+0xbfc8]','--:-:1:-:2')
e('LDS ur_epoch, [RZ+0xbf70]','--:-:2:-:2')
e('LDS ur_layer, [RZ+0xbff4]','--:-:2:-:2');e('NOP','03:-:-:-:6')
e('LDG.E.CG.64 ur_config0, [ur_ready+0x8]','--:-:1:-:2')
e('SHL ur_temp, ur_epoch, 0x5');e('ISCADD ur_temp, ur_epoch, ur_temp, 0x4')
e('IADD ur_epoch, ur_epoch, ur_temp');e('IADD ur_epoch, ur_epoch, ur_layer');e('IADD ur_epoch, ur_epoch, 0x1')
e('NOP','01:-:-:-:6')
pred('ISETP.NE.U32.AND P6, PT, cta, RZ, PT')
e('@P6 BRA UR_WAIT_LOCAL','--:-:-:Y:d')
pred('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT')
e('@P6 BRA UR_CANDIDATES_JOIN','--:-:-:Y:d')
e('LDG.E.CG.64 ur_logits0, [ur_config]','--:-:1:-:2')
e('SHL ur_offset, warp, 0x7');e('ISCADD ur_offset, lane, ur_offset, 0x2')
e('NOP','01:-:-:-:6');address('ur_logits','ur_logits','ur_offset')
pred('ISETP.LT.U32.AND P4, PT, lane, 0xa, PT')
e('@P4 LDG.E.CG ur_word, [ur_logits]','--:-:1:-:2')
e('@P4 LDG.E.CG ur_value, [ur_logits+0x28]','--:-:1:-:2')
e('XMAD ur_index, warp, 0xa, lane');e('SHL ur_index, ur_index, 0x2')
e('@P4 STS [ur_index+0xb200], ur_word','01:2:-:-:2')
e('@P4 STS [ur_index+0xb300], ur_value','--:2:-:-:2');e('NOP','02:-:-:-:6')
pred('ISETP.EQ.U32.AND P4, PT, warp, 0x5, PT')
pred('ISETP.EQ.U32.AND P4, PT, lane, RZ, P4')
e('@P4 LDG.E.CG ur_gate, [ur_logits+0x50]','--:-:1:-:2')
e('@P4 STS [RZ+0xb400], ur_gate','01:2:-:-:2');e('NOP','02:-:-:-:6')
label('UR_CANDIDATES_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT')
e('@P6 BRA UR_MERGE_JOIN','--:-:-:Y:d')
for j=0,1 do
 e('MOV32I ur_v'..j..', 0xff800000');e('MOV32I ur_id'..j..', 0xffffffff')
 e(string.format('IADD ur_index, lane, 0x%x',j*32));e('SHL ur_index, ur_index, 0x2')
 pred('ISETP.LT.U32.AND P4, PT, ur_index, 0xf0, PT')
 e('@P4 LDS ur_id'..j..', [ur_index+0xb200]','--:-:1:-:2')
 e('@P4 LDS ur_v'..j..', [ur_index+0xb300]','--:-:1:-:2');e('NOP','01:-:-:-:6')
end
top10(2,0xb500)
e('MOV ur_exp, RZ');e('MOV32I ur_log2e, 0x3fb8aa3b');e('MOV32I ur_negative_one, 0xbf800000')
e('SHL ur_index, lane, 0x2')
pred('ISETP.LT.U32.AND P4, PT, lane, 0xa, PT')
e('LDS ur_max, [RZ+0xb528]','--:-:1:-:2')
e('@P4 LDS ur_value, [ur_index+0xb528]','--:-:1:-:2')
e('@P4 LDS ur_selected, [ur_index+0xb500]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('@P4 FADD ur_value, ur_value, -ur_max');e('@P4 FMUL ur_value, ur_value, ur_log2e')
e('@P4 MUFU.EX2 ur_exp, ur_value','--:-:3:-:2');e('NOP','04:-:-:-:6')
e('MOV ur_sum, ur_exp')
for _,distance in ipairs({16,8,4,2,1})do
 e(string.format('SHFL.BFLY PT, ur_other, ur_sum, 0x%x, 0x1f',distance),'--:-:5:-:2')
 e('FADD ur_sum, ur_sum, ur_other','10:-:-:-:6')
end
e('MUFU.RCP ur_inverse, ur_sum','--:-:3:-:2')
e('FFMA ur_error, ur_sum, ur_inverse, ur_negative_one','04:-:-:-:6')
e('FADD.FTZ ur_error, -ur_error, -RZ');e('FFMA ur_inverse, ur_inverse, ur_error, ur_inverse')
e('FMUL ur_exp, ur_exp, ur_inverse')
for _,entry in ipairs({{8,'ur_selected'},{16,'ur_exp'},{24,'ur_selected'},{32,'ur_exp'}})do
 e(string.format('LDG.E.CG.64 ur_output0, [ur_config+0x%x]',entry[1]),'--:-:1:-:2')
 e('NOP','01:-:-:-:6');address('ur_output','ur_output','ur_index')
 e('@P4 STG.E [ur_output], '..entry[2],'--:2:-:-:2');e('NOP','02:-:-:-:6')
end
e('LDS ur_gate, [RZ+0xb400]','--:-:1:-:2')
e('LDG.E.CG.64 ur_output0, [ur_config+0x28]','--:-:2:-:2');e('NOP','03:-:-:-:6')
e('FADD ur_gate, -ur_gate, RZ');e('FMNMX ur_gate, ur_gate, 87, PT');e('FMNMX ur_gate, ur_gate, -87, !PT')
e('FMUL ur_gate, ur_gate, ur_log2e');e('MUFU.EX2 ur_gate, ur_gate','--:-:3:-:2')
e('FADD ur_gate, ur_gate, 1','04:-:-:-:6');e('MUFU.RCP ur_inverse, ur_gate','--:-:3:-:2')
e('FFMA ur_error, ur_gate, ur_inverse, ur_negative_one','04:-:-:-:6')
e('FADD.FTZ ur_error, -ur_error, -RZ');e('FFMA ur_gate, ur_inverse, ur_error, ur_inverse')
e('@P0 STG.E [ur_output], ur_gate','--:2:-:-:2')
e('MOV ur_count, 0xa');e('@P0 STG.E [ur_ready], ur_count','--:2:-:-:2')
e('MEMBAR.GL','3f:-:-:Y:d')
e('@P0 STG.E.WT [ur_ready+0x4], ur_epoch','--:2:-:-:2');e('NOP','02:-:-:-:6')
e('BRA UR_MERGE_JOIN','--:-:-:Y:d')
label('UR_WAIT_LOCAL')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA UR_MERGE_JOIN','--:-:-:Y:d')
label('UR_READY_SPIN')
e('LDG.E.CV ur_word, [ur_ready+0x4]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.EQ.U32.AND P4, PT, ur_word, ur_epoch, PT')
pred('VOTE.ALL P4, P4');e('@!P4 BRA UR_READY_SPIN','--:-:-:Y:d')
label('UR_MERGE_JOIN');e('BAR.SYNC 0x0','3f:-:-:Y:d');e('MEMBAR.GL.IVALLD')
label('UR_GATE_BEGIN')
rep('q2moegn_SETUP:','q2moegn_SETUP:\n'..table.concat(out,'\n'))
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
local count=assert(tonumber(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;'),16))
assert(count==108 or count==111,'unique router dispatch base')
rep(string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count),
 'ISETP.GE.U32.AND P6, PT, op, 0x70, PT;')
local table_entries={}
for opcode=count,111 do
 table_entries[#table_entries+1]='--:-:-:-:6 BRA '..(opcode==111 and 'urproj_SETUP' or 'POSITION_END')..';\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;'
end
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 table.concat(table_entries,'\n')..'\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('\nPOSITION_END:','\n'..local_body..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
