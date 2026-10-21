-- E4M3FN residual storage, one FP32 scale per 256 values. The conversion
-- operators are appended to either the full-target or native-prefill module.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2])
local maps,bodies={},{}
local function operator(tag,pack)
 local out={}
 local function emit(line)out[#out+1]=line:gsub('%%',tag)end
 maps[#maps+1]=([=[
    16 : %_value
    17 : %_maximum
    18 : %_temp
    19 : %_code
    20 : %_sign
    21 : %_absolute
    22 : %_index
    23 : %_offset
    24 : %_scale
    25 : %_inverse
    26 : %_lane
    27 : %_warp
    28 : %_n
    29 : %_sub
    30,31 : %_source<0-1>
    32,33 : %_dest<0-1>
    34,35 : %_scalep<0-1>
    36,37 : %_srcbase<0-1>
    38,39 : %_dstbase<0-1>
]=]):gsub('%%',tag)
 emit([=[%_SETUP:
--:-:-:-:6      MOV vid, cta;
--:-:-:-:6      ISETP.GE.U32.AND P6, PT, vid, units, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P6 BRA POSITION_END;
%_BEGIN:
--:-:1:-:1      LDS.64 %_srcbase0, [RZ+0xbf88];
--:-:1:-:1      LDS.64 %_dstbase0, [RZ+0xbf90];
--:-:1:-:2      LDS %_n, [RZ+0xbf98];
01:-:-:-:6      ISCADD %_index, vid, tid, 0x8;
--:-:-:-:6      SHL %_offset, %_index, 0x2;
]=])
 if pack then
  emit([=[--:-:-:-:6      IADD %_source0.CC, %_srcbase0, %_offset;
--:-:-:-:6      IADD.X %_source1, %_srcbase1, RZ;
--:-:-:-:6      IADD %_dest0.CC, %_dstbase0, %_index;
--:-:-:-:6      IADD.X %_dest1, %_dstbase1, RZ;
--:-:1:-:2      LDG.E.CG %_value, [%_source];
01:-:-:-:6      LOP32I.AND %_maximum, %_value, 0x7fffffff;
]=])
  for _,distance in ipairs{16,8,4,2,1}do emit(string.format('--:-:2:-:2      SHFL.BFLY PT, %%_temp, %%_maximum, 0x%x, 0x1f;\n02:-:-:-:6      FMNMX %%_maximum, %%_maximum, %%_temp, !PT;',distance))end
  emit([=[--:-:-:-:6      LOP32I.AND %_lane, tid, 0x1f;
--:-:-:-:6      SHR.U32 %_warp, tid, 0x5;
--:-:-:-:6      SHL %_offset, %_warp, 0x2;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, %_lane, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 STS [%_offset], %_maximum;
--:-:-:-:6      BAR.SYNC 0x0;
--:-:-:-:6      MOV32I %_maximum, 0x0d800000;
]=])
  for w=0,7 do emit(string.format('--:-:1:-:2      LDS %%_temp, [RZ+0x%x];\n01:-:-:-:6      FMNMX %%_maximum, %%_maximum, %%_temp, !PT;',4*w))end
  local div448=string.unpack('I4',string.pack('f',1/448))
  emit(string.format('--:-:-:-:6      MOV32I %%_temp, 0x%x;\n--:-:-:-:6      FMUL %%_scale, %%_maximum, %%_temp;',div448))
  emit([=[--:-:3:-:2      MUFU.RCP %_inverse, %_scale;
--:-:-:-:6      ISCADD %_offset, vid, %_n, 0x2;
--:-:-:-:6      IADD %_scalep0.CC, %_dstbase0, %_offset;
--:-:-:-:6      IADD.X %_scalep1, %_dstbase1, RZ;
--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, tid, RZ, PT;
04:-:-:-:6      FMUL %_value, %_value, %_inverse;
--:-:-:-:6      SHR.U32 %_sign, %_value, 0x18;
--:-:-:-:6      LOP32I.AND %_sign, %_sign, 0x80;
--:2:-:-:2  @P1 STG.E [%_scalep], %_scale;
--:-:-:-:6      LOP32I.AND %_absolute, %_value, 0x7fffffff;
--:-:-:-:6      SHR.U32 %_temp, %_absolute, 0x14;
--:-:-:-:6      LOP32I.AND %_temp, %_temp, 0x1;
--:-:-:-:6      IADD32I %_code, %_absolute, 0x7ffff;
--:-:-:-:6      IADD %_code, %_code, %_temp;
--:-:-:-:6      SHR.U32 %_code, %_code, 0x14;
--:-:-:-:6      IADD %_code, %_code, -0x3c0;
--:-:-:-:6      FMUL %_sub, %_absolute, 512;
--:-:3:-:2      F2I.S32.F32 %_sub, %_sub;
--:-:-:-:6      MOV32I %_temp, 0x3c800000;
--:-:-:-:6      ISETP.LT.U32.AND P2, PT, %_absolute, %_temp, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
04:-:-:-:6      SEL %_code, %_sub, %_code, P2;
--:-:-:-:6      IMNMX.U32 %_code, %_code, 0x7e, PT;
--:-:-:-:6      LOP.OR %_code, %_code, %_sign;
--:2:-:-:2      STG.E.U8 [%_dest], %_code;
]=])
 else
  emit([=[--:-:-:-:6      IADD %_source0.CC, %_srcbase0, %_index;
--:-:-:-:6      IADD.X %_source1, %_srcbase1, RZ;
--:-:-:-:6      IADD %_dest0.CC, %_dstbase0, %_offset;
--:-:-:-:6      IADD.X %_dest1, %_dstbase1, RZ;
--:-:-:-:6      ISCADD %_offset, vid, %_n, 0x2;
--:-:-:-:6      IADD %_scalep0.CC, %_srcbase0, %_offset;
--:-:-:-:6      IADD.X %_scalep1, %_srcbase1, RZ;
--:-:1:-:1      LDG.E.CG.U8 %_code, [%_source];
--:-:1:-:2      LDG.E.CG %_scale, [%_scalep];
01:-:-:-:6      LOP32I.AND %_sign, %_code, 0x80;
--:-:-:-:6      SHL %_sign, %_sign, 0x18;
--:-:-:-:6      LOP32I.AND %_code, %_code, 0x7f;
--:-:-:-:6      IADD %_value, %_code, 0x3c0;
--:-:-:-:6      SHL %_value, %_value, 0x14;
--:-:3:-:2      I2F.F32.S32 %_sub, %_code;
04:-:-:-:6      FMUL %_sub, %_sub, 0.001953125;
--:-:-:-:6      ISETP.LT.U32.AND P2, PT, %_code, 0x8, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      SEL %_value, %_sub, %_value, P2;
--:-:-:-:6      LOP.OR %_value, %_value, %_sign;
--:-:-:-:6      FMUL %_value, %_value, %_scale;
--:2:-:-:2      STG.E [%_dest], %_value;
]=])
 end
 emit([=[02:-:-:-:6      BAR.SYNC 0x0;
--:-:-:-:6      IADD vid, vid, 0x10;
--:-:-:-:6      ISETP.LT.U32.AND P6, PT, vid, units, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P6 BRA %_BEGIN;
--:-:-:-:6      BRA POSITION_END;
]=])
 bodies[#bodies+1]=table.concat(out,'\n')
end
operator('fp8pack',true);operator('fp8unpack',false)
local function once(a,b)local i,j=assert(s:find(a,1,true),'missing '..a);j=i+#a-1;assert(not s:find(a,j+1,true),'duplicate '..a);s=s:sub(1,i-1)..b..s:sub(j+1)end
local count=assert(tonumber(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;'),16))
assert(count==95 or count==119,'unexpected source opcode ABI')
once('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
once(string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count),string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count+2))
once('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6      BRA fp8pack_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA fp8unpack_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
once('POSITION_END:',table.concat(bodies,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'wb'));f:write(s);f:close()
