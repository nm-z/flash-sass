-- Row-striped routed/shared projections in the production balanced image.
-- The routed arithmetic is retained; expert strides and row ownership change.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function rep(text,a,b)
 local i=assert(text:find(a,1,true),'missing '..a)
 assert(not text:find(a,i+#a,true),'ambiguous '..a)
 return text:sub(1,i-1)..b..text:sub(i+#a)
end
local maps={[=[
    16,17 : br_src<0-1>
    18,19 : br_targets<0-1>
    20,21 : br_dest<0-1>
    22 : br_rows
    23 : br_global_rows
    24 : br_ranks
    25 : br_die
    26 : br_base
    27 : br_row
    28 : br_global
    29 : br_rank
    30 : br_offset
    31 : br_value
    32 : br_peer
    33 : br_tmp
    34 : br_flags
    35 : br_reduce
    36 : br_groups
    37 : br_group
    38 : br_piece
    39 : br_step
    248 : bm_flags
    240,241 : bd_low<0-1>
    242 : bd_owner
    243 : bd_seen
    244,245 : bd_bounds<0-1>
    246,247 : bd_address<0-1>
    30 : grcache_limit
]=]}
local wide_down={q2moedn=true,q2acc640=true,b32q40acc=true,b32q50acc=true,b32q80acc=true,b32iq4nlacc=true}
local function codegen()
 local out={};local function emit(ins,ctl)out[#out+1]=(ctl or '--:-:-:Y:d')..' '..ins..';'end
 local function label(x)out[#out+1]=x..':'end
 local function wait(x)emit('NOP',x..':-:-:-:6')end
 return out,emit,label,wait
end
local function publication(tag)
 local out,e,l,w=codegen()
 local rows_slot=tag=='bf16k2560' and 0xbfa0 or 0xbfa8
 local source_slot=tag=='bf16k2560' and 0xbf98 or 0xbfa0
 e('LDS br_flags, [RZ+0xbff8]','3f:-:1:-:2');w('01')
 e('LOP32I.AND br_flags, br_flags, 0x100000')
 e('ISETP.EQ.U32.AND P6, PT, br_flags, RZ, PT');e('@P6 BRA '..tag..'_STRIPE_DONE')
 e('MEMBAR.GL');e('BAR.SYNC 0x0')
 e('ISETP.GE.U32.AND P6, PT, warp, 0x6, PT')
 e('@P6 BRA '..tag..'_STRIPE_DONE')
 for _,x in ipairs({{'br_rows',rows_slot},{'br_global_rows',0xbfd8},{'br_ranks',0xbfdc},
                   {'br_die',0xbfe0},{'br_base',0xbfe4}})do
  e(string.format('LDS %s, [RZ+0x%x]',x[1],x[2]),'--:-:1:-:2');w('01')
 end
 if wide_down[tag] then e('ISCADD br_row, vid, lane, 0x5')
 else e('LOP32I.AND br_tmp, lane, 0x7');e('ISCADD br_row, vid, br_tmp, 0x3') end
 e('LDS br_reduce, [RZ+0xbff8]','--:-:1:-:2');w('01')
 e('LOP32I.AND br_reduce, br_reduce, 0x800000')
 e('ISETP.NE.U32.AND P5, PT, br_reduce, RZ, PT')
 e('MOV32I br_groups, 0x1');e('@P5 MOV32I br_groups, 0x6')
 e('SHL br_step, br_rows, 0x2')
 e('SHR.U32 br_global, br_row, 0x3')
 e('ISCADD br_global, br_global, br_global, 0x1')
 e('ISCADD br_global, br_global, br_die, 0x1')
 e('LOP32I.AND br_tmp, br_row, 0x7');e('ISCADD br_global, br_global, br_tmp, 0x3')
 -- One warp per peer. For eight-row stripes, each warp carries four ranks;
 -- the uniform rank-base loop avoids divergence before the CTA handoff.
 e('MOV br_peer, RZ')
 l(tag..'_STRIPE_RANK')
 if wide_down[tag] then e('MOV br_rank, br_peer')
 else e('SHR.U32 br_rank, lane, 0x3');e('IADD br_rank, br_rank, br_peer') end
 e('ISETP.LT.U32.AND P2, PT, br_row, br_rows, PT')
 e('ISETP.LT.U32.AND P2, PT, br_rank, br_ranks, P2')
 e('PSETP.AND.AND P0, PT, P2, P5, PT');e('PSETP.AND.AND P1, PT, P2, !P5, PT')
 e(string.format('@!P5 LDS.64 br_src0, [RZ+0x%x]',source_slot),'--:-:1:-:2')
 e('@P5 LDS.64 br_src0, [RZ+0xbfe8]','--:-:1:-:2');w('01')
 e('LDS.64 br_targets0, [RZ+0xbfd0]','--:-:1:-:2');w('01')
 e('SHL br_tmp, warp, 0x3');e('IADD br_targets0.CC, br_targets0, br_tmp');e('IADD.X br_targets1, br_targets1, RZ')
 e('XMAD br_offset, br_rank, br_rows, br_row');e('SHL br_offset, br_offset, 0x2')
 e('IADD br_src0.CC, br_src0, br_offset');e('IADD.X br_src1, br_src1, RZ')
 e('MOV br_value, RZ');e('MOV br_group, RZ')
 l(tag..'_STRIPE_PART')
 e('@P2 LDG.E.CG br_piece, [br_src]','--:-:1:-:2');w('01')
 e('@P0 FADD br_value, br_value, br_piece');e('@P1 MOV br_value, br_piece')
 e('IADD br_src0.CC, br_src0, br_step');e('IADD.X br_src1, br_src1, RZ')
 e('IADD br_group, br_group, 0x1')
 e('ISETP.LT.U32.AND P6, PT, br_group, br_groups, PT');e('@P6 BRA '..tag..'_STRIPE_PART')
 if tag=='q2moegn' then
  -- Remote addresses are contiguous across the four ranks in each warp.
  -- Global neuron order is restored by the local activation-read map.
  e('MOV br_tmp, 0x2');e('IMNMX.U32 br_tmp, br_die, br_tmp, PT')
  e('XMAD br_offset, br_die, 0xd, br_tmp');e('IADD br_offset, br_offset, vid')
  e('XMAD br_offset, br_offset, 0x50, RZ');e('ISCADD br_offset, br_rank, br_offset, 0x3')
  e('LOP32I.AND br_tmp, lane, 0x7');e('IADD br_offset, br_offset, br_tmp')
 else e('XMAD br_offset, br_rank, br_global_rows, br_global') end
 e('IADD br_offset, br_offset, br_base');e('SHL br_offset, br_offset, 0x2')
 e('LDG.E.CG.64 br_dest0, [br_targets]','--:-:1:-:2');w('01')
 e('IADD br_dest0.CC, br_dest0, br_offset');e('IADD.X br_dest1, br_dest1, RZ')
 e('@P2 STG.E.WT [br_dest], br_value','--:2:-:-:2');w('02')
 e('IADD br_peer, br_peer, '..(wide_down[tag] and '0x1' or '0x4'))
 e('ISETP.LT.U32.AND P6, PT, br_peer, br_ranks, PT')
 e('@P6 BRA '..tag..'_STRIPE_RANK')
 l(tag..'_STRIPE_DONE')
 return table.concat(out,'\n')..'\n'
end
local tags={'q2moegn','q2moedn','q2k2560','q2acc640','bf16k2560',
 'q3kk2560','q4kk2560','q5kk2560','q6kk2560','iq4k2560',
 'b32q40','b32q50','b32q80','b32iq4nl',
 'b32q40acc','b32q50acc','b32q80acc','b32iq4nlacc'}
for _,tag in ipairs(tags)do
 local start=assert(s:find(tag..'_SETUP:',1,true))
 local finish=assert(s:find('\n[%w_]+_SETUP:',start+1))
 local body=s:sub(start,finish-1)
 if tag=='q2moegn'then
  -- In this image bit32 denotes a same-token descriptor chain. It must not
  -- select the legacy whole-expert gate/up/quantization body.
  local n
  body,n=body:gsub('[^\n]*LDS ga_flag, %[%s*RZ%+0xbff8%];.-[^\n]*@P4 BRA GA_SETUP;\n','')
  assert(n==1,'legacy GA dispatch geometry')
 end
 local shift=assert(body:match('SHL '..tag..'_rowstep, '..tag..'_rowstep, 0x(%x+);'))
 local group_shift=8-tonumber(shift,16)
 local tile_shift=wide_down[tag] and 5 or 3
 local tile=1<<tile_shift
 -- Only descriptors marked ROW8 take this path. Preserve ordinary dense GEMVs.
 local prefix=([=[
3f:-:1:-:2 LDS bm_flags, [RZ+0xbff8];
01:-:-:Y:d LOP32I.AND bm_flags, bm_flags, 0x80000;
--:-:-:Y:d ISETP.EQ.U32.AND P6, PT, bm_flags, RZ, PT;
--:-:-:Y:d @P6 BRA @_ROW_DEFAULT;
--:-:-:Y:d ISETP.GE.U32.AND P6, PT, tid, 0x%THREADS%, PT;
--:-:-:Y:d @P6 BRA @_END;
@_ROW_DEFAULT:
]=]):gsub('@_',tag..'_'):gsub('%%THREADS%%',string.format('%x',tag=='q2moegn' and 256 or math.min(256,tile*(1<<group_shift))))
 if wide_down[tag] then
  -- A warp owns eight rows for every K640 format here. Tail rows are whole
  -- warps: bypass their loads before entering the native unpredicated loop.
  -- All warps still join the existing END/publication or chain barrier.
  assert(group_shift==2,'wide tail requires four lanes per row')
  local tail,te,tl,tw=codegen()
  te('LDS br_rows, [RZ+0xbfa8]','--:-:1:-:2');tw('01')
  te('SHL br_tmp, warp, 0x3');te('ISCADD br_tmp, vid, br_tmp, 0x5')
  te('ISETP.GE.U32.AND P6, PT, br_tmp, br_rows, PT');te('@P6 BRA '..tag..'_END')
  tl(tag..'_WIDE_TAIL_OK')
  prefix=rep(prefix,tag..'_ROW_DEFAULT:',table.concat(tail,'\n')..'\n'..tag..'_ROW_DEFAULT:')
 end
 body=rep(body,tag..'_BEGIN:',tag..'_BEGIN:\n'..prefix)
 -- A predicated rowstep/base change retains the original dense loop otherwise.
 body=body:gsub('(SHL '..tag..'_rowstep, '..tag..'_rowstep, 0x%x+;)',function(ins)
  return ins..'\n--:-:-:Y:d ISETP.NE.U32.AND P6, PT, bm_flags, RZ, PT;\n'..
   '--:-:-:Y:d @P6 SHL '..tag..'_rowstep, '..tag..'_gridDimX, 0x'..string.format('%x',tile_shift)..';'
 end,1)
 -- The flag lives outside all selected operators' registers (their maximum
 -- is240). Preserve the original group operand before overwriting row itself.
 local base_pat='([^\n]*ISCADD '..tag..'_row, '..tag..'_bid, ([%w_]+), 0x%x+;)'
 local n
 body,n=body:gsub(base_pat,function(old,group)
  local mask=assert(old:match('^(%S+):')):match('^[^:]+')
  return mask..':-:-:-:6 NOP;\n--:-:-:Y:d ISETP.NE.U32.AND P6, PT, bm_flags, RZ, PT;\n'..
   (tag=='q2moegn' and '--:-:-:Y:d @P6 LOP32I.AND '..group..', '..group..', 0x7;\n' or '')..
   old:gsub(' ISCADD',' @!P6 ISCADD',1)..'\n'..
   '--:-:-:Y:d @P6 ISCADD '..tag..'_row, '..tag..'_bid, '..group..', 0x'..string.format('%x',tile_shift)..';'
 end)
 assert(n==1,'row base geometry '..tag..' '..n)
 if tag=='q2moegn' then
  -- Two CTA halves visit even/odd ranks respectively. Each warp stays
  -- uniform, and the existing publication barrier joins all ten outputs.
  body=rep(body,'MOV q2moegn_e, RZ;','SHR.U32 q2moegn_e, tid, 0x7;')
  body=rep(body,'IADD q2moegn_e, q2moegn_e, 0x1;','IADD q2moegn_e, q2moegn_e, 0x2;')
 end
 -- Existing END publication is disabled on balanced descriptors (no0x80).
 body=rep(body,tag..'_END:',tag..'_END:\n'..publication(tag))
 s=s:sub(1,start-1)..body..s:sub(finish)
end
-- Routed matrices use a compact local row stride for all512 global experts.
local function routine(tag,edit)
 local a=assert(s:find(tag..'_SETUP:',1,true))
 local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
 s=s:sub(1,a-1)..edit(s:sub(a,b-1))..s:sub(b)
end
for _,x in ipairs({{'q2moegn','0x800','0x4','0x6','0x7','0x9'},
                  {'q2moedn','0x8000','0x6','0x7'}})do
 local tag=x[1]
 local old='02:-:-:-:6      XMAD '..tag..'_t0, '..tag..'_eoff, '..x[2]..', RZ;\n'..
  '--:-:-:-:6      XMAD.PSL '..tag..'_eoff, '..tag..'_eoff, 0x7, '..tag..'_t0;'
 local out,e=codegen()
 e('XMAD '..tag..'_t0, '..tag..'_eoff, '..tag..'_param_rows, RZ','02:-:-:Y:d')
 e('SHL '..tag..'_eoff, '..tag..'_t0, '..x[3])
 for i=4,#x do e('ISCADD '..tag..'_eoff, '..tag..'_t0, '..tag..'_eoff, '..x[i])end
 routine(tag,function(body)return rep(body,old,table.concat(out,'\n'))end)
end
local old='--:-:-:-:6      XMAD q2moegn_t0, q2moegn_e, 0xa00, RZ;'
routine('q2moegn',function(body)return rep(body,old,'--:-:-:Y:d XMAD q2moegn_t0, q2moegn_e, q2moegn_param_rows, RZ;\n--:-:-:Y:d SHL q2moegn_t0, q2moegn_t0, 0x2;')end)
-- Preserve the old six-part FP32 association while visiting each rank once.
-- An expert-to-owner map selects the same partial; a six-bit seen mask keeps
-- the original first-product versus accumulated-FMA choice for each owner.
routine('q2moedn',function(body)
 local out,e,l,w=codegen()
 e('SHR.U32 bd_low0, tid, 0x2');e('ISCADD bd_low0, vid, bd_low0, 0x5')
 e('LOP32I.AND bd_low1, tid, 0x3')
 e('ISETP.EQ.U32.AND P5, PT, bd_low1, RZ, PT')
 e('ISETP.LT.U32.AND P5, PT, bd_low0, q2moedn_param_rows, P5')
 e('SHL bd_low0, bd_low0, 0x2');e('MOV bd_owner, RZ')
 l('BALANCED_DOWN_ZERO')
 e('XMAD bd_low1, bd_owner, q2moedn_param_rows, RZ')
 e('ISCADD bd_low1, bd_low1, bd_low0, 0x2')
 e('IADD bd_address0.CC, q2moedn_param_Y0, bd_low1');e('IADD.X bd_address1, q2moedn_param_Y1, RZ')
 e('@P5 STG.E.WT [bd_address], RZ','--:2:-:-:2');w('02')
 e('IADD bd_owner, bd_owner, 0x1');e('ISETP.LT.U32.AND P6, PT, bd_owner, 0x6, PT')
 e('@P6 BRA BALANCED_DOWN_ZERO');e('MEMBAR.GL')
 e('LDS.64 bd_bounds0, [RZ+0xbfd0]','--:-:1:-:2');w('01')
 e('IADD bd_bounds0.CC, bd_bounds0, 0x1c');e('IADD.X bd_bounds1, bd_bounds1, RZ')
 e('MOV bd_seen, RZ');e('MOV q2moedn_e, RZ')
 body=rep(body,'--:-:-:-:6      MOV q2moedn_e, RZ;',table.concat(out,'\n'))
 out,e,l,w=codegen()
 e('SHL bd_low0, q2moedn_eoff, 0x2','02:-:-:Y:d')
 e('IADD bd_address0.CC, bd_bounds0, bd_low0');e('IADD.X bd_address1, bd_bounds1, RZ')
 e('LDG.E.CG bd_owner, [bd_address]','--:-:1:-:2');w('01')
 e('LDS q2moedn_param_Y0, [RZ+0xbfa0]','--:-:1:-:2')
 e('LDS q2moedn_param_Y1, [RZ+0xbfa4]','--:-:1:-:2');w('01')
 e('XMAD q2moedn_t0, bd_owner, q2moedn_param_rows, RZ')
 e('SHL q2moedn_t0, q2moedn_t0, 0x2')
 e('IADD q2moedn_param_Y0.CC, q2moedn_param_Y0, q2moedn_t0')
 e('IADD.X q2moedn_param_Y1, q2moedn_param_Y1, RZ')
 e('MOV bd_low0, 0x1');e('SHL bd_low0, bd_low0, bd_owner')
 e('LOP.AND bd_low1, bd_seen, bd_low0')
 e('ISETP.EQ.U32.AND P3, PT, bd_low1, RZ, PT')
 e('LOP.OR bd_seen, bd_seen, bd_low0')
 body=rep(body,'--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, q2moedn_e, RZ, PT;',table.concat(out,'\n'))
 return body
end)
-- The assembled MoE output is one vector, not six partials.
s=rep(s,'--:-:-:-:6 MOV grcache_peer, RZ;', [=[
--:-:-:Y:d MOV grcache_peer, RZ;
--:-:1:-:2 LDS grcache_limit, [RZ+0xbff8];
01:-:-:Y:d LOP32I.AND grcache_limit, grcache_limit, 0x400000;
--:-:-:Y:d ISETP.NE.U32.AND P6, PT, grcache_limit, RZ, PT;
--:-:-:Y:d MOV32I grcache_limit, 0x6;
--:-:-:Y:d @P6 MOV32I grcache_limit, 0x1;]=])
s=rep(s,'ISETP.LT.U32.AND P6, PT, grcache_peer, 0x6, PT;',
        'ISETP.LT.U32.AND P6, PT, grcache_peer, grcache_limit, PT;')
local prepare=dofile('balance/prepare.lua')(s,maps)
s=rep(s,'q2moedn_SETUP:','q2moedn_SETUP:\n'..prepare)
s=rep(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
