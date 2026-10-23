-- Install the B4 entry for the shared scalar/B4 compact-column body.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i,j=assert(s:find(a,1,true));assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
assert(s:find('PCOL4_SETUP:',1,true),'paired scalar body missing')
replace('ISETP.GE.U32.AND P6, PT, op, 0x7b, PT;','ISETP.GE.U32.AND P6, PT, op, 0x7c, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA PCOL4_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
