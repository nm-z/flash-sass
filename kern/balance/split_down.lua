-- Two physical lanes cooperate on each original Q2 down-projection lane.
-- Each computes sixteen of the original thirty-two integer products. Pair
-- sums happen before I2F, preserving all five FMAs and the final four-way
-- floating-point reduction. The weight/activation storage ABI is unchanged.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local a=assert(s:find('q2moedn_SETUP:',1,true))
local b=assert(s:find('\n[%w_]+_SETUP:',a+1))
local body=s:sub(a,b-1)
local function replace(old,new,expected)
 local n=0;local at=1;local out={}
 while true do
  local p,q=body:find(old,at,true)
  if not p then out[#out+1]=body:sub(at);break end
  out[#out+1]=body:sub(at,p-1);out[#out+1]=new;at=q+1;n=n+1
 end
 assert(n==(expected or 1),'split-down coverage '..old..': '..n)
 body=table.concat(out)
end
local function remove_line(ins)
 local p=assert(body:find(ins,1,true),'missing '..ins)
 assert(not body:find(ins,p+#ins,true),'ambiguous '..ins)
 local start=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local finish=assert(body:find('\n',p,true))
 body=body:sub(1,start-1)..body:sub(finish+1)
end
-- The CTA still owns32 rows, now with eight useful warps instead of four.
replace('ISETP.GE.U32.AND P6, PT, tid, 0x80, PT;',
        'ISETP.GE.U32.AND P6, PT, tid, 0x100, PT;')
replace('SHL br_tmp, warp, 0x3;\n--:-:-:Y:d ISCADD br_tmp, vid, br_tmp, 0x5;',
        'SHL br_tmp, warp, 0x2;\n--:-:-:Y:d ISCADD br_tmp, vid, br_tmp, 0x5;')
replace('SHR.U32 bd_low0, tid, 0x2;', 'SHR.U32 bd_low0, tid, 0x3;')
replace('LOP32I.AND bd_low1, tid, 0x3;', 'LOP32I.AND bd_low1, tid, 0x7;')
replace('SHL q2moedn_rowstep, q2moedn_rowstep, 0x6;',
        'SHL q2moedn_rowstep, q2moedn_rowstep, 0x5;')
replace('ISCADD q2moedn_row, q2moedn_bid, q2moedn_grp, 0x6;',
        'ISCADD q2moedn_row, q2moedn_bid, q2moedn_grp, 0x5;')
replace('LOP32I.AND q2moedn_lane, q2moedn_tid, 0x3;', [=[LOP32I.AND q2moedn_lane, q2moedn_tid, 0x7;
--:-:-:-:6 LOP32I.AND sd_part, q2moedn_lane, 0x1;
--:-:-:-:6 SHR.U32 sd_lane, q2moedn_lane, 0x1;]=],2)
replace('SHR.U32 q2moedn_grp, q2moedn_tid, 0x2;',
        'SHR.U32 q2moedn_grp, q2moedn_tid, 0x3;',2)
replace('LOP32I.AND q2moedn_half, q2moedn_lane, 0x1;',
        'LOP32I.AND q2moedn_half, sd_lane, 0x1;',2)
replace('SHR.U32 q2moedn_blk0, q2moedn_lane, 0x1;',
        'SHR.U32 q2moedn_blk0, sd_lane, 0x1;',2)
-- Each partner selects one32-bit word in the existing64-bit code payload.
replace('IADD q2moedn_t1, q2moedn_t1, 0x20;', [=[ISCADD q2moedn_t1, sd_part, q2moedn_t1, 0x2;
--:-:-:-:6 IADD q2moedn_t1, q2moedn_t1, 0x20;]=])
replace('IADD ep_tmp, ep_tmp, 0x20;', [=[ISCADD ep_tmp, sd_part, ep_tmp, 0x2;
--:-:-:-:6 IADD ep_tmp, ep_tmp, 0x20;]=])
-- The original activation half-block is64 bytes. Partners read32 bytes each.
replace('SHL q2moedn_t1, q2moedn_lane, 0x6;', [=[SHL q2moedn_t1, sd_lane, 0x6;
--:-:-:-:6 ISCADD q2moedn_t1, sd_part, q2moedn_t1, 0x5;]=])
replace('SHL q2moedn_t1, q2moedn_lane, 0x2;',
        'SHL q2moedn_t1, sd_lane, 0x2;')
for chunk=0,4 do
 for _,j in ipairs({8,12}) do
  remove_line(string.format('LDG.E.CG.128 q2moedn_act%02d, [q2moedn_xp+0x%x];',
                            16*chunk+j,256*chunk+4*j))
 end
 replace(string.format('LDG.E.CI.64 q2moedn_q%d,',2*chunk),
         string.format('LDG.E.CI q2moedn_q%d,',2*chunk))
 replace(string.format('LDG.E.CI.64 ep_q%d,',2*chunk),
         string.format('LDG.E.CI ep_q%d,',2*chunk))
 remove_line(string.format('@P5 MOV q2moedn_q%d, ep_q%d;',2*chunk+1,2*chunk+1))
 -- Drop the second sixteen-product body; this now executes on the partner.
 local first=assert(body:find('LOP32I.AND q2moedn_m0, q2moedn_q'..(2*chunk+1)..',',1,true))
 local start=(body:sub(1,first-1):match('.*()\n') or 0)+1
 local last=assert(body:find('SHR q2moedn_acc1,',first,true))
 local finish=assert(body:sub(1,last-1):match('.*()\n'))
 local removed=body:sub(start,finish);local _,masks=removed:gsub('LOP32I%.AND','')
 local _,muls=removed:gsub('XMAD%.U16%.S16','')
 assert(masks==8 and muls==16,'half-block split geometry')
 body=body:sub(1,start-1)..body:sub(finish+1)
end
-- Subtract the32-value activation sum once per pair, on the even partner.
local zero={'--:-:-:Y:d ISETP.NE.U32.AND P6, PT, sd_part, RZ, PT;',
            '--:-:-:-:6 NOP;'}
for i=0,4 do zero[#zero+1]=string.format('--:-:-:-:6 @P6 MOV q2moedn_nsum%d, RZ;',i)end
replace('q2moedn_LOOP:',table.concat(zero,'\n')..'\nq2moedn_LOOP:')
-- Integer dot bounds: |q*a|<=381 and32 products, well below signed32 range.
-- IADD is exact here, so pairing cannot change the subsequent FP32 inputs.
replace('I2F.F32.S32 q2moedn_dotf, q2moedn_dot;', [=[SHFL.BFLY PT, sd_peer, q2moedn_dot, 0x1, 0x1f;
04:-:-:-:6 IADD q2moedn_dot, q2moedn_dot, sd_peer;
--:-:3:-:6 I2F.F32.S32 q2moedn_dotf, q2moedn_dot;]=],5)
-- Original reduction order was XOR2 then XOR1. Double the distances to
-- reduce only equivalent partners in the new eight-lane row.
replace('SHFL.BFLY PT, q2moedn_t, q2moedn_accf, 0x2, 0x1f;',
        'SHFL.BFLY PT, q2moedn_t, q2moedn_accf, 0x4, 0x1f;')
replace('SHFL.BFLY PT, q2moedn_t, q2moedn_accf, 0x1, 0x1f;',
        'SHFL.BFLY PT, q2moedn_t, q2moedn_accf, 0x2, 0x1f;')
replace('q2moedn_BEGIN:', 'q2moedn_SPLIT_DOT:\nq2moedn_BEGIN:')
s=s:sub(1,a-1)..body..s:sub(b)
local maps=[=[
    210 : sd_part
    211 : sd_lane
    212 : sd_peer
]=]
local at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..maps..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write('split_down lanes_per_row=8 active_warps_per_full_tile=8 integer_products_per_lane=80 fp32_order=preserved\n')
