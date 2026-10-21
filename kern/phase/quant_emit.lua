-- Quantized producer publication emitter shared by GDN, GR and QSA.
return function(maps)
local function quant(tag,values,tableoffset,setup,finish)
 maps[#maps+1]=([=[
    128-131 : %_v<0-3>
    132-135 : %_q<0-3>
    136 : %_maximum
    137 : %_r
    138 : %_temp
    139 : %_mneg1
    140 : %_scale
    141 : %_s16
    142 : %_s32
    143 : %_index
    144 : %_qoff
    145 : %_off16
    146 : %_off32
    147 : %_lane
    148 : %_peer
    149 : %_cols
    150,151 : %_table<0-1>
    152,153 : %_qp<0-1>
    154,155 : %_sp32<0-1>
    156,157 : %_sp16<0-1>
    158,159 : %_ap<0-1>
    160,161 : %_packed<0-1>
]=]):gsub('%%',tag)
 local out={};local function emit(x)out[#out+1]=x:gsub('%%',tag)end
 emit('3f:-:-:-:6 NOP;\n'..setup)
 emit(string.format('--:-:1:-:2 LDS.64 %%_table0, [RZ+0x%x];',tableoffset))
 emit('--:-:-:-:6 MOV32I %_mneg1, 0xbf800000;\n--:-:-:-:6 FMNMX %_maximum, |%_v0|, 7.8886090522101180541e-31, !PT;')
 for i=1,values-1 do emit(string.format('--:-:-:-:6 FMNMX %%_maximum, %%_maximum, |%%_v%d|, !PT;',i))end
 local span=32//values
 local shift=span//2
 while shift>0 do
  emit(string.format('--:-:5:-:2 SHFL.BFLY PT, %%_temp, %%_maximum, 0x%x, 0x1f;\n10:-:-:-:6 FMNMX %%_maximum, %%_maximum, %%_temp, !PT;',shift));shift=shift//2
 end
 emit([=[--:-:3:-:2 MUFU.RCP %_r, %_maximum;
04:-:-:-:6 FFMA %_temp, %_maximum, %_r, %_mneg1;
--:-:-:-:6 FADD.FTZ %_temp, -%_temp, -RZ;
--:-:-:-:6 FFMA %_r, %_r, %_temp, %_r;
--:-:-:-:6 FMUL %_r, %_r, 127;
--:-:-:-:6 MOV32I %_temp, 0x3c010204;
--:-:-:-:6 FMUL %_scale, %_maximum, %_temp;
]=])
 for i=0,values-1 do emit(string.format('--:-:-:-:6 FMUL %%_q%d, %%_v%d, %%_r;\n--:-:3:-:2 F2I.S32.F32 %%_q%d, %%_q%d;\n04:-:-:-:6 NOP;',i,i,i,i))end
 emit('--:-:-:-:6 MOV %_s16, %_q0;')
 for i=1,values-1 do emit(string.format('--:-:-:-:6 IADD %%_s16, %%_s16, %%_q%d;',i))end
 shift=1
 while shift<span//2 do emit(string.format('--:-:5:-:2 SHFL.BFLY PT, %%_temp, %%_s16, 0x%x, 0x1f;\n10:-:-:-:6 IADD %%_s16, %%_s16, %%_temp;',shift));shift=shift*2 end
 emit(string.format('--:-:5:-:2 SHFL.BFLY PT, %%_temp, %%_s16, 0x%x, 0x1f;\n10:-:-:-:6 IADD %%_s32, %%_s16, %%_temp;',span//2))
 emit([=[--:-:-:-:6 SHL %_qoff, %_index, 0x1;
--:-:-:-:6 SHR.U32 %_off16, %_index, 0x4;
--:-:-:-:6 SHL %_off16, %_off16, 0x2;
--:-:-:-:6 SHR.U32 %_off32, %_index, 0x5;
--:-:-:-:6 SHL %_off32, %_off32, 0x2;
--:-:-:-:6 LOP32I.AND %_temp, %_index, 0x1f;
--:-:-:-:6 ISETP.EQ.U32.AND P0, PT, %_temp, RZ, PT;
--:-:-:-:6 LOP32I.AND %_temp, %_index, 0xf;
--:-:-:-:6 ISETP.EQ.U32.AND P1, PT, %_temp, RZ, PT;
]=])
 if values==4 then
  for i=0,1 do emit(string.format('--:-:-:-:6 LOP32I.AND %%_packed%d, %%_q%d, 0xffff;\n--:-:-:-:6 SHL %%_temp, %%_q%d, 0x10;\n--:-:-:-:6 LOP.OR %%_packed%d, %%_packed%d, %%_temp;',i,2*i,2*i+1,i,i))end
 end
 emit([=[01:-:-:-:6 MOV %_peer, RZ;
%_SEND:
02:-:1:-:1 LDG.E.64 %_qp0, [%_table+0x0];
--:-:1:-:1 LDG.E.64 %_sp320, [%_table+0x8];
--:-:1:-:1 LDG.E.64 %_sp160, [%_table+0x10];
--:-:1:-:2 LDG.E.64 %_ap0, [%_table+0x18];
01:-:-:-:6 IADD %_qp0.CC, %_qp0, %_qoff;
--:-:-:-:6 IADD.X %_qp1, %_qp1, RZ;
--:-:-:-:6 IADD %_sp320.CC, %_sp320, %_off32;
--:-:-:-:6 IADD.X %_sp321, %_sp321, RZ;
--:-:-:-:6 IADD %_sp160.CC, %_sp160, %_off16;
--:-:-:-:6 IADD.X %_sp161, %_sp161, RZ;
--:-:-:-:6 IADD %_ap0.CC, %_ap0, %_off32;
--:-:-:-:6 IADD.X %_ap1, %_ap1, RZ;
]=])
 emit(values==4 and '--:2:-:-:1 STG.E.WT.64 [%_qp], %_packed0;' or '--:2:-:-:1 STG.E.WT.U16 [%_qp], %_q0;')
 emit([=[--:2:-:-:1 @P0 STG.E.WT [%_sp32], %_s32;
--:2:-:-:1 @P1 STG.E.WT [%_sp16], %_s16;
--:2:-:-:2 @P0 STG.E.WT [%_ap], %_scale;
--:-:-:-:6 IADD %_table0.CC, %_table0, 0x20;
--:-:-:-:6 IADD.X %_table1, %_table1, RZ;
--:-:-:-:6 IADD %_peer, %_peer, 0x1;
--:-:-:-:6 ISETP.LT.U32.AND P3, PT, %_peer, 0x6, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P3 BRA %_SEND;
02:-:-:-:6 NOP;
]=])
 emit(finish);return table.concat(out,'\n')
end
return quant
end
