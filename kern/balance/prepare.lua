-- Generate the real gate/up -> activation-quantization -> down dependency.
-- All CTAs quantize owned32-value groups, publish their own stores, and join
-- once. This is local readiness inside the existing down dispatch.
return function(source,maps)
 maps[#maps+1]=[=[
    16 : bact_gate
    17 : bact_up
    18 : bact_value
    19 : bact_t1
    20 : bact_t2
    21 : bact_t3
    22 : bact_t4
    23 : bact_t5
    24 : bact_log2e
    25 : bact_ln2hi
    26 : bact_ln2lo
    27 : bact_c720
    28 : bact_c120
    29 : bact_c24
    30 : bact_c6
    31 : bact_chalf
    32 : bact_cone
    33 : bact_mneg1
    200,201 : mb_prepare<0-1>
    204,205 : mb_g<0-1>
    206,207 : mb_u<0-1>
    208,209 : mb_ready<0-1>
    210,211 : mb_targets<0-1>
    212 : mb_index
    213 : mb_token
    214 : mb_layer
    215 : mb_epoch
    216,217 : mb_address<0-1>
    218 : mb_value
    219 : mb_flags
    220 : mb_temp
]=]
 local out={};local function e(ins,ctl)out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'end
 local function l(x)out[#out+1]=x..':'end
 local function w(mask)e('NOP',mask..':-:-:-:6')end
 e('LDS mb_flags, [RZ+0xbff8]','3f:-:1:-:2');w('01')
 e('LOP32I.AND mb_flags, mb_flags, 0x200000')
 e('ISETP.EQ.U32.AND P6, PT, mb_flags, RZ, PT');e('@P6 BRA BM_PREP_DONE')
 e('LDS.64 mb_prepare0, [RZ+0xbfe8]','--:-:1:-:2');w('01')
 for _,x in ipairs({{'mb_g0',0},{'mb_u0',8},{'mb_ready0',48}})do
  e(string.format('LDG.E.CG.64 %s, [mb_prepare+0x%x]',x[1],x[2]),'--:-:1:-:2');w('01')
 end
 e('IADD mb_targets0.CC, mb_prepare0, 0x10');e('IADD.X mb_targets1, mb_prepare1, RZ')
 e('LDS mb_token, [RZ+0xbf70]','--:-:1:-:2');w('01')
 e('LDS mb_layer, [RZ+0xbff4]','--:-:1:-:2');w('01')
 e('IADD mb_epoch, mb_token, 0x1');e('ISCADD mb_epoch, mb_epoch, mb_layer, 0x6')
 local constants=assert(source:match('([^\n]*MOV32I ga_log2e,.-)\n[^\n]*MOV32I ga_c127,'))
 out[#out+1]=constants:gsub('ga_','bact_')
 e('ISCADD mb_index, cta, tid, 0x8')
 l('BM_PREP_LOOP')
 e('ISETP.GE.U32.AND P6, PT, mb_index, 0x1b80, PT');e('@P6 BRA BM_PREP_QUANT_DONE')
 e('SHL mb_temp, mb_index, 0x2')
 e('IADD mb_address0.CC, mb_g0, mb_temp');e('IADD.X mb_address1, mb_g1, RZ')
 e('LDG.E.CG bact_gate, [mb_address]','--:-:1:-:2');w('01')
 e('IADD mb_address0.CC, mb_u0, mb_temp');e('IADD.X mb_address1, mb_u1, RZ')
 e('LDG.E.CG bact_up, [mb_address]','--:-:1:-:2');w('01')
 local first=assert(source:find('FADD ga_t4, -ga_gate, RZ;',1,true))
 first=assert(source:sub(1,first):match('.*\n()'))
 local _,last=assert(source:find('FMUL ga_value, ga_value, q2moegn_accf;',first,true))
 out[#out+1]=source:sub(first,last):gsub('ga_','bact_'):gsub('q2moegn_accf','bact_up')
 local quant=dofile('balance/../phase/quant_emit.lua')(maps)
 local q=quant('bmq',1,0xbfe8,
   '--:-:-:Y:d MOV %_v0, bact_value;\n--:-:-:Y:d MOV %_index, mb_index;',
   '--:-:-:Y:d IADD mb_index, mb_index, 0x1000;\n--:-:-:Y:d BRA BM_PREP_LOOP;\n')
 local n
 q,n=q:gsub('[^\n]*LDS%.64 bmq_table0, %[%s*RZ%+0xbfe8%];',
   '--:-:-:Y:d MOV bmq_table0, mb_targets0;\n--:-:-:Y:d MOV bmq_table1, mb_targets1;')
 assert(n==1,'quantization pointer table geometry')
 q,n=q:gsub('bmq_peer, 0x6, PT','bmq_peer, 0x1, PT');assert(n==1)
 out[#out+1]=q
 l('BM_PREP_QUANT_DONE')
 e('MEMBAR.GL','3f:-:-:-:6');e('BAR.SYNC 0x0')
 e('SHL mb_temp, cta, 0x2')
 e('IADD mb_address0.CC, mb_ready0, mb_temp');e('IADD.X mb_address1, mb_ready1, RZ')
 e('ISETP.EQ.U32.AND P5, PT, tid, RZ, PT')
 e('@P5 STG.E.WT [mb_address], mb_epoch','--:2:-:-:2');w('02')
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA BM_PREP_BARRIER')
 e('LOP32I.AND mb_temp, lane, 0xf');e('SHL mb_temp, mb_temp, 0x2')
 e('IADD mb_address0.CC, mb_ready0, mb_temp');e('IADD.X mb_address1, mb_ready1, RZ')
 l('BM_PREP_WAIT')
 e('LDG.E.CV mb_value, [mb_address]','--:-:1:-:2');w('01')
 e('ISETP.EQ.U32.AND P5, PT, mb_value, mb_epoch, PT')
 e('VOTE.ALL P5, P5');e('@!P5 BRA BM_PREP_WAIT')
 l('BM_PREP_BARRIER')
 e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD')
 l('BM_PREP_DONE')
 return table.concat(out,'\n')..'\n'
end
