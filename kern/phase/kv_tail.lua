-- Never dequantize unwritten rows after the causal end of the final KV block.
-- A zero attention probability cannot mask NaN from an uninitialized FP16 scale.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local count=0
local pattern='([^\n]*LDG%.E%.64 kvgather_kd0, %[kvgather_kdp%+0x(%x+)%];)\n([^\n]*LDG%.E%.64 kvgather_vd0, %[kvgather_vdp%+0x%x+%];)\n([^\n]*LDG%.E%.U16 kvgather_ksc, %[kvgather_ksp%+0x%x+%];)\n([^\n]*LDG%.E%.U16 kvgather_vsc, %[kvgather_vsp%+0x%x+%];)'
s=s:gsub(pattern,function(a,offset,b,c,d)
 count=count+1;local row=tonumber(offset,16)//256;local i=row//2
 assert(row>=0 and row<8)
 local out={
 '--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, kvgather_unit, kvgather_ns, PT;',
 '--:-:-:-:6 NOP;','--:-:-:-:6 NOP;','--:-:-:-:6 NOP;',
 string.format('--:-:-:-:6 ISETP.LE.U32.AND P5, PT, kvgather_tail, 0x%x, P5;',i),
 '--:-:-:-:6 NOP;','--:-:-:-:6 NOP;','--:-:-:-:6 NOP;'}
 for _,reg in ipairs({'kd0','kd1','vd0','vd1','ksc','vsc'})do out[#out+1]='--:-:-:-:6 MOV kvgather_'..reg..', RZ;'end
 for _,line in ipairs({a,b,c,d})do out[#out+1]=(line:gsub('LDG%.E','@!P5 LDG.E',1))end
 return table.concat(out,'\n')
end)
assert(count==8,'expected eight (KV head, row) gather loads')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
