-- Native four-position router projection. Preserve its 32-lane dot order,
-- then use the production local-top10 and peer-packet tail for each position.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i,j=s:find(a,1,true);assert(i,a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local first=assert(s:find('urproj_END:',1,true))
local finish=assert(s:find('UR_LOCAL_DONE:',first,true))
finish=assert(s:find('BRA POSITION_END;',finish,true))+#'BRA POSITION_END;'-1
local tail=s:sub(first,finish)
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
local function wait(m)e('NOP',m..':-:-:-:6')end
l('router4_SETUP');wait('3f');pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
e('IADD dfgr_offset, pidx, 0x1');e('IADD dfgr_offset, dfgr_offset, warp');e('SHL dfgr_offset, dfgr_offset, 0x7')
e('IADD dfgr_pointer0.CC, prog0, dfgr_offset');e('IADD.X dfgr_pointer1, prog1, RZ')
e('SHL dfgr_offset, lane, 0x2');e('IADD dfgr_pointer0.CC, dfgr_pointer0, dfgr_offset');e('IADD.X dfgr_pointer1, dfgr_pointer1, RZ')
e('@P5 LDG.E.CG dfgr_word, [dfgr_pointer]','--:-:1:-:2')
e('ISCADD dfgr_descriptor, warp, dfgr_offset, 0x7');e('IADD dfgr_descriptor, dfgr_descriptor, 0x8000')
e('@P5 STS [dfgr_descriptor], dfgr_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS dfgr_rows, [RZ+0x8020]','--:-:1:-:2');e('LDS.64 dfgr_base0, [RZ+0x8008]','--:-:1:-:2');wait('01')
e('ISCADD dfgr_row, cta, warp, 0x3')
l('ROUTER4_ROW');pred('ISETP.GE.U32.AND P6, PT, dfgr_row, dfgr_rows, PT');e('@P6 BRA ROUTER4_ROWS_DONE')
e('XMAD dfgr_offset, dfgr_row, 0x1400, RZ');e('ISCADD dfgr_offset, lane, dfgr_offset, 0x4')
e('IADD dfgr_wp0.CC, dfgr_base0, dfgr_offset');e('IADD.X dfgr_wp1, dfgr_base1, RZ')
for p=0,3 do e(string.format('LDS.64 dfgr_xp%02d, [RZ+0x%x]',p*2,0x8010+p*128),'--:-:1:-:2')end
wait('01');e('SHL dfgr_offset, lane, 0x5')
for p=0,3 do
 e(string.format('IADD dfgr_xp%02d.CC, dfgr_xp%02d, dfgr_offset',p*2,p*2));e(string.format('IADD.X dfgr_xp%02d, dfgr_xp%02d, RZ',p*2+1,p*2+1));e('MOV dfgr_acc'..p..', RZ')
end
e('MOV dfgr_tile, RZ');l('ROUTER4_K')
e('LDG.E.CI.128 dfgr_packed0, [dfgr_wp]','--:2:1:-:2')
for p=0,3 do for h=0,1 do e(string.format('LDG.E.CG.128 dfgr_x%02d, [dfgr_xp%02d+0x%x]',p*8+h*4,p*2,h*16),'--:3:4:-:2')end end
wait('09')
for j=0,7 do
 if j%2==0 then e(string.format('SHL dfgr_weight%d, dfgr_packed%d, 0x10',j,j//2))
 else e(string.format('LOP32I.AND dfgr_weight%d, dfgr_packed%d, 0xffff0000',j,j//2))end
end
for j=0,7 do for p=0,3 do e(string.format('FFMA dfgr_acc%d, dfgr_weight%d.reuse, dfgr_x%02d, dfgr_acc%d',p,j,p*8+j,p),p==3 and '--:-:-:-:3' or '--:-:-:-:1')end end
wait('02');e('IADD dfgr_wp0.CC, dfgr_wp0, 0x200');e('IADD.X dfgr_wp1, dfgr_wp1, RZ');wait('04')
for p=0,3 do e(string.format('IADD dfgr_xp%02d.CC, dfgr_xp%02d, 0x400',p*2,p*2));e(string.format('IADD.X dfgr_xp%02d, dfgr_xp%02d, RZ',p*2+1,p*2+1))end
e('IADD dfgr_tile, dfgr_tile, 0x1');pred('ISETP.LT.U32.AND P6, PT, dfgr_tile, 0xa, PT');e('@P6 BRA ROUTER4_K')
for _,distance in ipairs({16,8,4,2,1})do
 for p=0,3 do e(string.format('SHFL.BFLY PT, dfgr_shuffle%d, dfgr_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 wait('10');for p=0,3 do e(string.format('FADD dfgr_acc%d, dfgr_acc%d, dfgr_shuffle%d',p,p,p))end
end
pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT');e('SHL dfgr_offset, dfgr_row, 0x2')
for p=0,3 do
 e(string.format('LDS.64 dfgr_pointer0, [RZ+0x%x]',0x8018+p*128),'--:-:1:-:2');wait('01')
 e('IADD dfgr_pointer0.CC, dfgr_pointer0, dfgr_offset');e('IADD.X dfgr_pointer1, dfgr_pointer1, RZ')
 e(string.format('@P0 STG.E [dfgr_pointer], dfgr_acc%d',p),'--:2:-:-:2');wait('02')
end
e('IADD dfgr_row, dfgr_row, 0x80');e('BRA ROUTER4_ROW')
l('ROUTER4_ROWS_DONE');e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
for p=0,3 do
 pred('ISETP.EQ.U32.AND P5, PT, warp, RZ, PT');e('SHL dfgr_offset, lane, 0x2')
 e(string.format('@P5 LDS dfgr_word, [dfgr_offset+0x%x]',0x8000+p*128),'--:-:1:-:2')
 e('@P5 STS [dfgr_offset+0xbf80], dfgr_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
 local body=tail;local labels={}
 for line in body:gmatch('[^\n]+')do local name=line:match('^([%a_][%w_]*):%s*$');if name then labels[#labels+1]=name end end
 for _,name in ipairs(labels)do body=body:gsub('%f[%w_]'..name..'%f[^%w_]',name..'_r4_'..p)end
 body=body:gsub('BRA POSITION_END;','BRA ROUTER4_DONE_'..p..';')
 o[#o+1]=body;l('ROUTER4_DONE_'..p);e('BAR.SYNC 0x0','3f:-:-:Y:d')
end
e('BRA POSITION_END')
replace('ISETP.GE.U32.AND P6, PT, op, 0x7c, PT;','ISETP.GE.U32.AND P6, PT, op, 0x7d, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA router4_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
