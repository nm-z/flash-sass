-- Native B4 GDN: prepare all q/k/value/gates before loading recurrent state.
-- Phase-A math aliases the state registers in the scalar operator. Running it
-- inside the recurrent loop would destroy the live state tile. Four shared
-- records separate those lifetimes, then four ordered updates share one load.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2]);local scalar=read(assert(arg[3]))
local function rep(a,b)
 local i,j=s:find(a,1,true);assert(i,'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local maps={};for line in assert(scalar:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find(': dnstep_',1,true) and not line:match('dnstep_param_[SO][01]%s*$')then
  maps[#maps+1]=line:gsub('dnstep_','dfdn_')
 end
end
maps[#maps+1]=[=[
    134,135 : dfdn_param_S<0-1>
    148,149 : dfdn_param_O<0-1>
    200 : dfdn_descriptor
    201 : dfdn_position
    202 : dfdn_shared
    203 : dfdn_offset
    204,205 : dfdn_fetch<0-1>
    206 : dfdn_word
    208,209 : dfdn_table<0-1>
    210,211 : dfdn_select<0-1>
    212 : dfdn_index
    213 : dfdn_heads
]=]
local begin=assert(scalar:find('dnstep_BEGIN:',1,true))
local state_begin=assert(scalar:find('--:-:1:-:1      LDG.E.CG.128 dnstep_s00,',begin,true))
local math_begin=assert(scalar:find('--:-:2:-:1      LDS.128 dnstep_kk00,',state_begin,true))
local stores_begin=assert(scalar:find('--:2:-:-:1      STG.E.128 [dnstep_sp+0x0],',math_begin,true))
local ending=assert(scalar:find('02:-:-:-:f      BRA dnstep_END;',stores_begin,true))
local prepare=scalar:sub(begin,state_begin-1):gsub('dnstep_','dfdn_')
local loads=scalar:sub(state_begin,math_begin-1):gsub('dnstep_','dfdn_')
local math=scalar:sub(math_begin,stores_begin-1):gsub('dnstep_','dfdn_')
local stores=scalar:sub(stores_begin,ending-1):gsub('dnstep_','dfdn_')
prepare=prepare:gsub('%[RZ%+0x(%x+)%]',function(h)
 local n=tonumber(h,16)
 return n>=0xbf80 and n<0xc000 and string.format('[dfdn_descriptor+0x%x]',n-0xbf80) or '[RZ+0x'..h..']'
end)
-- Each prepared record occupies 2048 bytes, including two scalar gates.
prepare=prepare:gsub('([^\n]*STS%.128 %[dfdn_smaddr%][^\n]*)',function(line)
 return '--:-:-:-:6 IADD dfdn_smaddr, dfdn_smaddr, dfdn_shared;\n'..line
end)
prepare=prepare:gsub('MOV32I dfdn_t, 0x600;', 'MOV32I dfdn_t, 0x600;\n--:-:-:-:6 IADD dfdn_t, dfdn_t, dfdn_shared;')
math=math:gsub('%[RZ%+0x600%]','[dfdn_shared+0x600]'):gsub('%[RZ%+0x604%]','[dfdn_shared+0x604]')
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
l('dfdn_SETUP');e('NOP','3f:-:-:Y:d')
-- Four warps cache four descriptors. This storage is separate from all four
-- prepared records and from the production dispatcher's parameter bank.
pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
e('IADD dfdn_offset, pidx, 0x1');e('IADD dfdn_offset, dfdn_offset, warp');e('SHL dfdn_offset, dfdn_offset, 0x7')
e('IADD dfdn_fetch0.CC, prog0, dfdn_offset');e('IADD.X dfdn_fetch1, prog1, RZ')
e('SHL dfdn_offset, lane, 0x2');e('IADD dfdn_fetch0.CC, dfdn_fetch0, dfdn_offset');e('IADD.X dfdn_fetch1, dfdn_fetch1, RZ')
e('@P5 LDG.E.CG dfdn_word, [dfdn_fetch]','--:-:1:-:2')
e('ISCADD dfdn_offset, warp, dfdn_offset, 0x7');e('IADD dfdn_offset, dfdn_offset, 0x8000')
e('@P5 STS [dfdn_offset], dfdn_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS dfdn_heads, [RZ+0x8004]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.GE.U32.AND P6, PT, cta, dfdn_heads, PT');e('@P6 BRA POSITION_END','--:-:-:Y:d')
e('MOV vid, cta');e('MOV vtid, tid');e('MOV dfdn_position, RZ')
l('DFDN_PREPARE')
e('SHL dfdn_shared, dfdn_position, 0xb');e('SHL dfdn_descriptor, dfdn_position, 0x7');e('IADD dfdn_descriptor, dfdn_descriptor, 0x8000')
o[#o+1]=prepare
e('IADD dfdn_position, dfdn_position, 0x1');pred('ISETP.LT.U32.AND P6, PT, dfdn_position, 0x4, PT')
e('@P6 BRA DFDN_PREPARE','--:-:-:Y:d')
-- Resolve the first position's input bank after all aliasing preparation.
e('LDS.64 dfdn_param_S0, [RZ+0x8008]','--:-:1:-:2')
e('LDS.64 dfdn_table0, [RZ+0x8068]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('LDG.E.CG.64 dfdn_select0, [dfdn_table]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('LDG.E.CV dfdn_index, [dfdn_select]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('ISCADD dfdn_offset, dfdn_index, 0x8, 0x3');e('IADD dfdn_table0.CC, dfdn_table0, dfdn_offset');e('IADD.X dfdn_table1, dfdn_table1, RZ')
e('LDG.E.CG.64 dfdn_param_S0, [dfdn_table]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('SHL dfdn_offset, dfdn_head, 0x10');e('SHL dfdn_index, dfdn_g, 0xb');e('IADD dfdn_offset, dfdn_offset, dfdn_index')
e('ISCADD dfdn_offset, dfdn_p, dfdn_offset, 0x6')
e('IADD dfdn_sp0.CC, dfdn_param_S0, dfdn_offset');e('IADD.X dfdn_sp1, dfdn_param_S1, RZ')
o[#o+1]=loads;e('NOP','01:-:-:-:6');e('MOV dfdn_position, RZ')
l('DFDN_UPDATE')
e('SHL dfdn_shared, dfdn_position, 0xb');e('SHL dfdn_descriptor, dfdn_position, 0x7');e('IADD dfdn_descriptor, dfdn_descriptor, 0x8000')
e('LDS.64 dfdn_param_O0, [dfdn_descriptor+0x40]','--:-:1:-:2')
e('LDS.64 dfdn_param_S0, [dfdn_descriptor+0x60]','--:-:1:-:2')
e('SHL dfdn_smq, dfdn_p, 0x6');e('IADD dfdn_smq, dfdn_smq, dfdn_shared')
e('IADD dfdn_smk, dfdn_smq, 0x200');e('SHL dfdn_t2, dfdn_g, 0x4');e('IADD dfdn_t2, dfdn_t2, 0x400');e('IADD dfdn_t2, dfdn_t2, dfdn_shared')
e('SHL dfdn_t, dfdn_head, 0x9');e('ISCADD dfdn_t, dfdn_g, dfdn_t, 0x4')
e('IADD dfdn_op0.CC, dfdn_param_O0, dfdn_t','01:-:-:-:6');e('IADD.X dfdn_op1, dfdn_param_O1, RZ')
e('IADD dfdn_sp0.CC, dfdn_param_S0, dfdn_offset');e('IADD.X dfdn_sp1, dfdn_param_S1, RZ')
pred('ISETP.EQ.U32.AND P3, PT, dfdn_p, RZ, PT')
o[#o+1]=math;o[#o+1]=stores;e('NOP','02:-:-:-:6')
e('IADD dfdn_position, dfdn_position, 0x1');pred('ISETP.LT.U32.AND P6, PT, dfdn_position, 0x4, PT')
e('@P6 BRA DFDN_UPDATE','--:-:-:Y:d');e('BRA POSITION_END','--:-:-:Y:d')
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x72, PT;','ISETP.GE.U32.AND P6, PT, op, 0x73, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA dfdn_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'w'));f:write(s);f:close()
