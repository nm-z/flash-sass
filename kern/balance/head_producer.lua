-- Accumulate CTA winners from the output GEMV's already-reduced logits.
-- Q5_K2560 uses R16..220; R222..229 are dead throughout its arithmetic.
-- No dot-product operation or accumulation order is changed. The generic
-- routine retains its old behavior unless descriptor bit24 is set.
return function(s)
 local function replace(a,b)
  if a:sub(1,1)=='\n' then assert(b:sub(1,1)=='\n','replacement loses instruction/label boundary')end
  local p=assert(s:find(a,1,true),'missing '..a)
  assert(not s:find(a,p+#a,true),'ambiguous '..a)
  s=s:sub(1,p-1)..b..s:sub(p+#a)
 end
 replace('</REGISTER_MAPPING>',[=[    222 : hp_mode
    223 : hp_max
    224 : hp_idx
    225 : hp_other_max
    226 : hp_other_idx
    227 : hp_tmp
    228,229 : hp_ptr<0-1>
</REGISTER_MAPPING>]=])
 -- Initialize before the row-shard wrapper's early exits. In non-head uses,
 -- inactive warps branch straight from BEGIN to END and must see mode0.
 replace('\nq5kk2560_BEGIN:','\n'..[=[q5kk2560_BEGIN:
--:-:6:-:2 LDS hp_mode, [RZ+0xbff8];
--:-:-:Y:d MOV32I hp_max, 0xff800000;
--:-:-:Y:d MOV32I hp_idx, 0xffffffff;
20:-:-:Y:d LOP32I.AND hp_mode, hp_mode, 0x1000000;]=])
 local store='--:2:-:-:1  @P2 STG.E [q5kk2560_yp], q5kk2560_accf;'
 replace(store,[=[--:-:-:Y:d ISETP.NE.U32.AND P4, PT, hp_mode, RZ, P2;
--:-:-:Y:d FSETP.GT.AND P5, PT, q5kk2560_accf, hp_max, P4;
--:-:-:Y:d @P5 MOV hp_max, q5kk2560_accf;
--:-:-:Y:d @P5 MOV hp_idx, q5kk2560_row;
]=]..store)
 local out={};local function e(x)out[#out+1]='--:-:-:Y:d '..x..';'end
 local function butterfly()
  for _,step in ipairs({16,8,4,2,1})do
   out[#out+1]=string.format('--:-:5:-:2 SHFL.BFLY PT, hp_other_max, hp_max, 0x%x, 0x1f;',step)
   out[#out+1]=string.format('--:-:5:-:2 SHFL.BFLY PT, hp_other_idx, hp_idx, 0x%x, 0x1f;',step)
   out[#out+1]='10:-:-:Y:d FSETP.GT.AND P4, PT, hp_other_max, hp_max, PT;'
   e('FSETP.EQ.AND P5, PT, hp_other_max, hp_max, PT')
   e('ISETP.LT.U32.AND P5, PT, hp_other_idx, hp_idx, P5')
   e('PSETP.OR.AND P4, PT, P4, P5, PT')
   e('@P4 MOV hp_max, hp_other_max');e('@P4 MOV hp_idx, hp_other_idx')
  end
 end
 out[#out+1]='3f:-:-:Y:d NOP;'
 e('ISETP.EQ.U32.AND P6, PT, hp_mode, RZ, PT');e('@P6 BRA HEAD_PRODUCER_DONE')
 butterfly()
 e('SHL hp_tmp, warp, 0x3');e('ISETP.EQ.U32.AND P4, PT, lane, RZ, PT')
 out[#out+1]='--:1:-:-:2 @P4 STS [hp_tmp], hp_idx;'
 out[#out+1]='--:1:-:-:2 @P4 STS [hp_tmp+0x4], hp_max;'
 out[#out+1]='01:-:-:Y:d BAR.SYNC 0x0;'
 e('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA HEAD_PRODUCER_DONE')
 e('MOV32I hp_max, 0xff800000');e('MOV32I hp_idx, 0xffffffff')
 e('SHL hp_tmp, lane, 0x3');e('ISETP.LT.U32.AND P4, PT, lane, 0x8, PT')
 out[#out+1]='--:-:1:-:2 @P4 LDS hp_idx, [hp_tmp];'
 out[#out+1]='--:-:1:-:2 @P4 LDS hp_max, [hp_tmp+0x4];'
 out[#out+1]='01:-:-:Y:d NOP;'
 butterfly()
 out[#out+1]='--:-:1:-:2 LDS.64 hp_ptr0, [RZ+0xbfc0];'
 e('SHL hp_tmp, cta, 0x3')
 out[#out+1]='01:-:-:Y:d IADD hp_ptr0.CC, hp_ptr0, hp_tmp;'
 e('IADD.X hp_ptr1, hp_ptr1, RZ');e('ISETP.EQ.U32.AND P4, PT, lane, RZ, PT')
 out[#out+1]='--:2:-:-:2 @P4 STG.E.WT [hp_ptr], hp_idx;'
 out[#out+1]='--:2:-:-:2 @P4 STG.E.WT [hp_ptr+0x4], hp_max;'
 out[#out+1]='02:-:-:Y:d NOP;'
 -- Release each CTA's pair before announcing completion to its local die.
 -- The last CTA's warp reduces the16 pairs and publishes one die winner.
 e('MEMBAR.GL')
 out[#out+1]='--:-:1:-:2 LDS.64 hp_ptr0, [RZ+0xbfd0];'
 e('MOV hp_tmp, 0x1')
 out[#out+1]='01:-:2:-:2 @P4 ATOM.E.ADD hp_other_idx, [hp_ptr], hp_tmp;'
 out[#out+1]='02:-:5:-:2 SHFL.IDX PT, hp_other_idx, hp_other_idx, RZ, 0x1f;'
 out[#out+1]='10:-:-:Y:d ISETP.NE.U32.AND P6, PT, hp_other_idx, 0xf, PT;'
 e('@P6 BRA HEAD_PRODUCER_DONE')
 e('MEMBAR.GL.IVALLD')
 out[#out+1]='--:-:1:-:2 LDS.64 hp_ptr0, [RZ+0xbfc0];'
 e('SHL hp_tmp, lane, 0x3')
 out[#out+1]='01:-:-:Y:d IADD hp_ptr0.CC, hp_ptr0, hp_tmp;'
 e('IADD.X hp_ptr1, hp_ptr1, RZ')
 e('MOV32I hp_max, 0xff800000');e('MOV32I hp_idx, 0xffffffff')
 e('ISETP.LT.U32.AND P4, PT, lane, 0x10, PT')
 out[#out+1]='--:-:1:-:2 @P4 LDG.E.CV hp_idx, [hp_ptr];'
 out[#out+1]='--:-:1:-:2 @P4 LDG.E.CV hp_max, [hp_ptr+0x4];'
 out[#out+1]='01:-:-:Y:d NOP;'
 butterfly()
 out[#out+1]='--:-:1:-:2 LDS.64 hp_ptr0, [RZ+0xbfc8];'
 e('ISETP.EQ.U32.AND P4, PT, lane, RZ, PT')
 out[#out+1]='01:2:-:-:2 @P4 STG.E.WT [hp_ptr], hp_idx;'
 out[#out+1]='--:2:-:-:2 @P4 STG.E.WT [hp_ptr+0x4], hp_max;'
 out[#out+1]='02:-:1:-:2 LDS.64 hp_ptr0, [RZ+0xbfd0];'
 out[#out+1]='01:2:-:-:2 @P4 STG.E.WT [hp_ptr], RZ;'
 out[#out+1]='02:-:-:Y:d NOP;'
 out[#out+1]='HEAD_PRODUCER_DONE:'
 replace('\nq5kk2560_END:','\nq5kk2560_END:\n'..table.concat(out,'\n'))
 assert(s:find('\nq5kk2560_BEGIN:\n',1,true))
 assert(s:find('\n--:-:6:-:2 LDS hp_mode, [RZ+0xbff8];\n',1,true))
 return s
end
