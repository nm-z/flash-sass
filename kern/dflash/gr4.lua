-- Four-position GR down/read. A single producer CTA runs the existing exact
-- residual-write/RMS prologues, then releases their four global XH vectors.
-- Matrix CTAs share each unpacked BF16 weight across four independent chains.
-- The scalar reduction order is retained: 80 terms/lane, a warp butterfly,
-- then the four residual branches summed in their original order.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2])
local function replace(a,b)
 local i,j=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local maps={};local prefixes={}
for _,tag in ipairs({'bf16hcrx','bf16hcrwx','grreadsum'})do
 local new='dfgr_'..tag
 for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(': '..tag..'_',1,true)then maps[#maps+1]=line:gsub(tag..'_',new..'_')end
 end
 local first=assert(s:find(tag..'_BEGIN:',1,true))
 local last=assert(s:find('MOV '..tag..'_yp0, '..tag..'_param_Y0;',first,true))
 last=(s:sub(1,last-1):match('.*()\n') or 0)+1
 local body=s:sub(first,last-1):gsub(tag..'_',new..'_')
 if tag=='grreadsum' then
  for line in assert(s:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
   if line:find(': grcache_',1,true)then maps[#maps+1]=line:gsub('grcache_','dfgrcache_')end
  end
  body=body:gsub('grcache_','dfgrcache_'):gsub('GRCACHE_','DFGRCACHE_')
 end
 prefixes[#prefixes+1]=body..'\n--:-:-:Y:d BRA DFGR_PREPARED_ONE;\n'
end
maps[#maps+1]=[=[
    16-19 : dfgr_acc<0-3>
    20-23 : dfgr_sum<0-3>
    24-27 : dfgr_shuffle<0-3>
    32-63 : dfgr_x<00-31>
    64-67 : dfgr_packed<0-3>
    68-75 : dfgr_weight<0-7>
    80-87 : dfgr_xp<00-07>
    88,89 : dfgr_wp<0-1>
    90,91 : dfgr_base<0-1>
    92 : dfgr_row
    93 : dfgr_rows
    94 : dfgr_slice
    95 : dfgr_tile
    96 : dfgr_offset
    97 : dfgr_tmp
    98 : dfgr_position
    99 : dfgr_descriptor
    100,101 : dfgr_pointer<0-1>
    102,103 : dfgr_ready<0-1>
    104 : dfgr_epoch
    105 : dfgr_word
    106 : dfgr_peer
    108,109 : dfgr_table<0-1>
    110,111 : dfgr_dest<0-1>
    112 : dfgr_timer
    113 : dfgr_start
]=]
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
l('dfgr_SETUP');e('NOP','3f:-:-:Y:d')
e('LDS.64 dfgr_ready0, [RZ+0xbf98]','--:-:1:-:2')
e('LDS dfgr_epoch, [RZ+0xbf74]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('IADD dfgr_epoch, dfgr_epoch, 0x1')
pred('ISETP.EQ.U32.AND P0, PT, tid, RZ, PT')
e('@P0 STS.64 [RZ+0xa800], dfgr_ready0','--:2:-:-:2')
e('@P0 STS [RZ+0xa808], dfgr_epoch','--:2:-:-:2')
e('@P0 STS [RZ+0xa80c], RZ','--:2:-:-:2')
-- Four descriptor records remain above the 40 KiB RMS scratch region.
pred('ISETP.LT.U32.AND P5, PT, warp, 0x4, PT')
e('IADD dfgr_offset, pidx, 0x1');e('IADD dfgr_offset, dfgr_offset, warp');e('SHL dfgr_offset, dfgr_offset, 0x7')
e('IADD dfgr_pointer0.CC, prog0, dfgr_offset');e('IADD.X dfgr_pointer1, prog1, RZ')
e('SHL dfgr_offset, lane, 0x2');e('IADD dfgr_pointer0.CC, dfgr_pointer0, dfgr_offset');e('IADD.X dfgr_pointer1, dfgr_pointer1, RZ')
e('@P5 LDG.E.CG dfgr_word, [dfgr_pointer]','--:-:1:-:2')
e('ISCADD dfgr_descriptor, warp, dfgr_offset, 0x7');e('IADD dfgr_descriptor, dfgr_descriptor, 0xa400')
e('@P5 STS [dfgr_descriptor], dfgr_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
pred('ISETP.NE.U32.AND P6, PT, cta, RZ, PT');e('@P6 BRA DFGR_WAIT','--:-:-:Y:d')
l('DFGR_PREPARE')
e('LDS dfgr_position, [RZ+0xa80c]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('SHL dfgr_descriptor, dfgr_position, 0x7');e('IADD dfgr_descriptor, dfgr_descriptor, 0xa400')
pred('ISETP.EQ.U32.AND P5, PT, warp, RZ, PT')
e('SHL dfgr_offset, lane, 0x2');e('IADD dfgr_pointer0, dfgr_descriptor, dfgr_offset')
e('@P5 LDS dfgr_word, [dfgr_pointer0]','--:-:1:-:2')
e('@P5 STS [dfgr_offset+0xbf80], dfgr_word','01:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
e('LDS dfgr_word, [RZ+0xbf80]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('MOV vid, RZ');e('MOV vtid, tid')
pred('ISETP.EQ.U32.AND P5, PT, dfgr_word, 0x24, PT');e('@P5 BRA dfgr_bf16hcrx_BEGIN','--:-:-:Y:d')
pred('ISETP.EQ.U32.AND P5, PT, dfgr_word, 0x25, PT');e('@P5 BRA dfgr_bf16hcrwx_BEGIN','--:-:-:Y:d')
e('BRA dfgr_grreadsum_BEGIN','--:-:-:Y:d')
l('DFGR_PREPARED_ONE');e('NOP','3f:-:-:Y:d')
e('LDS dfgr_position, [RZ+0xa80c]','--:-:1:-:2');e('NOP','01:-:-:-:6')
e('IADD dfgr_position, dfgr_position, 0x1');pred('ISETP.EQ.U32.AND P0, PT, tid, RZ, PT')
e('@P0 STS [RZ+0xa80c], dfgr_position','--:2:-:-:2');e('BAR.SYNC 0x0','02:-:-:Y:d')
pred('ISETP.LT.U32.AND P6, PT, dfgr_position, 0x4, PT');e('@P6 BRA DFGR_PREPARE','--:-:-:Y:d')
e('LDS.64 dfgr_ready0, [RZ+0xa800]','--:-:1:-:2');e('LDS dfgr_epoch, [RZ+0xa808]','--:-:1:-:2')
e('MEMBAR.GL','3f:-:-:Y:d');e('BAR.SYNC 0x0')
pred('ISETP.EQ.U32.AND P0, PT, tid, RZ, PT')
e('@P0 STG.E.WT [dfgr_ready], dfgr_epoch','--:2:-:-:2');e('NOP','02:-:-:-:6')
l('DFGR_WAIT')
e('LDS.64 dfgr_ready0, [RZ+0xa800]','--:-:1:-:2');e('LDS dfgr_epoch, [RZ+0xa808]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA DFGR_READY','--:-:-:Y:d')
l('DFGR_POLL');e('LDG.E.CV dfgr_word, [dfgr_ready]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.GE.U32.AND P5, PT, dfgr_word, dfgr_epoch, PT');pred('VOTE.ALL P5, P5')
e('@P5 BRA DFGR_READY','--:-:-:Y:d')
e('S2R dfgr_timer, SR_GLOBALTIMERHI','--:-:6:-:2');e('LDS dfgr_start, [RZ+0xbf5c]','--:-:1:-:2')
e('IADD dfgr_timer, dfgr_timer, -dfgr_start','21:-:-:-:6')
pred('ISETP.GE.U32.AND P5, PT, dfgr_timer, 0x46, PT');pred('VOTE.ANY P5, P5');e('@P5 BRA BEXP','--:-:-:Y:d')
e('BRA DFGR_POLL','--:-:-:Y:d')
l('DFGR_READY');e('BAR.SYNC 0x0','3f:-:-:Y:d');e('MEMBAR.GL.IVALLD')
e('LDS dfgr_rows, [RZ+0xa420]','--:-:1:-:2');e('LDS.64 dfgr_base0, [RZ+0xa408]','--:-:1:-:2');e('NOP','01:-:-:-:6')
pred('ISETP.GE.U32.AND P6, PT, warp, 0x4, PT');e('@P6 BRA POSITION_END','--:-:-:Y:d')
e('ISCADD dfgr_row, cta, warp, 0x2')
l('DFGR_ROW');pred('ISETP.GE.U32.AND P6, PT, dfgr_row, dfgr_rows, PT');e('@P6 BRA POSITION_END','--:-:-:Y:d')
e('XMAD dfgr_offset, dfgr_row, 0x5000, RZ');e('ISCADD dfgr_offset, lane, dfgr_offset, 0x4')
e('IADD dfgr_wp0.CC, dfgr_base0, dfgr_offset');e('IADD.X dfgr_wp1, dfgr_base1, RZ')
for p=0,3 do
 e(string.format('LDS.64 dfgr_xp%02d, [RZ+0x%x]',p*2,0xa440+128*p),'--:-:1:-:2')
end
e('SHL dfgr_offset, lane, 0x5');e('NOP','01:-:-:-:6')
for p=0,3 do
 e(string.format('IADD dfgr_xp%02d.CC, dfgr_xp%02d, dfgr_offset',p*2,p*2))
 e(string.format('IADD.X dfgr_xp%02d, dfgr_xp%02d, RZ',p*2+1,p*2+1))
end
e('MOV dfgr_slice, RZ')
l('DFGR_SLICE');e('MOV dfgr_tile, RZ');for p=0,3 do e('MOV dfgr_acc'..p..', RZ')end
l('DFGR_K')
e('LDG.E.CI.128 dfgr_packed0, [dfgr_wp]','--:2:1:-:2')
for p=0,3 do for h=0,1 do
 e(string.format('LDG.E.CG.128 dfgr_x%02d, [dfgr_xp%02d+0x%x]',8*p+4*h,p*2,16*h),'--:3:4:-:2')
end end
e('NOP','09:-:-:-:6')
for j=0,7 do
 if j%2==0 then e(string.format('SHL dfgr_weight%d, dfgr_packed%d, 0x10',j,j//2))
 else e(string.format('LOP32I.AND dfgr_weight%d, dfgr_packed%d, 0xffff0000',j,j//2))end
end
for j=0,7 do for p=0,3 do
 e(string.format('FFMA dfgr_acc%d, dfgr_weight%d.reuse, dfgr_x%02d, dfgr_acc%d',p,j,p*8+j,p),p==3 and '--:-:-:-:3' or '--:-:-:-:1')
end end
e('IADD dfgr_wp0.CC, dfgr_wp0, 0x200','02:-:-:-:6');e('IADD.X dfgr_wp1, dfgr_wp1, RZ')
e('NOP','04:-:-:-:6')
for p=0,3 do
 e(string.format('IADD dfgr_xp%02d.CC, dfgr_xp%02d, 0x400',p*2,p*2))
 e(string.format('IADD.X dfgr_xp%02d, dfgr_xp%02d, RZ',p*2+1,p*2+1))
end
e('IADD dfgr_tile, dfgr_tile, 0x1');pred('ISETP.LT.U32.AND P6, PT, dfgr_tile, 0xa, PT');e('@P6 BRA DFGR_K','--:-:-:Y:d')
for _,distance in ipairs({16,8,4,2,1})do
 for p=0,3 do e(string.format('SHFL.BFLY PT, dfgr_shuffle%d, dfgr_acc%d, 0x%x, 0x1f',p,p,distance),'--:-:5:-:2')end
 e('NOP','10:-:-:-:6')
 for p=0,3 do e(string.format('FADD dfgr_acc%d, dfgr_acc%d, dfgr_shuffle%d',p,p,p))end
end
pred('ISETP.EQ.U32.AND P0, PT, dfgr_slice, RZ, PT')
for p=0,3 do e(string.format('@P0 MOV dfgr_sum%d, dfgr_acc%d',p,p));e(string.format('@!P0 FADD dfgr_sum%d, dfgr_sum%d, dfgr_acc%d',p,p,p))end
e('IADD dfgr_slice, dfgr_slice, 0x1');pred('ISETP.LT.U32.AND P6, PT, dfgr_slice, 0x4, PT');e('@P6 BRA DFGR_SLICE','--:-:-:Y:d')
-- One row owner publishes all four positions; no second read of the matrix.
pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT')
for p=0,3 do
 e(string.format('LDS.64 dfgr_table0, [RZ+0x%x]',0xa418+p*128),'--:-:1:-:2')
 e(string.format('LDS dfgr_offset, [RZ+0x%x]',0xa430+p*128),'--:-:1:-:2');e('NOP','01:-:-:-:6')
 e('ISCADD dfgr_offset, dfgr_row, dfgr_offset, 0x2')
 for peer=0,5 do
  e(string.format('@P0 LDG.E.CG.64 dfgr_dest0, [dfgr_table+0x%x]',peer*8),'--:-:1:-:2');e('NOP','01:-:-:-:6')
  e('IADD dfgr_dest0.CC, dfgr_dest0, dfgr_offset');e('IADD.X dfgr_dest1, dfgr_dest1, RZ')
  e(string.format('@P0 STG.E.WT [dfgr_dest], dfgr_sum%d',p),'--:2:-:-:2');e('NOP','02:-:-:-:6')
 end
end
e('IADD dfgr_row, dfgr_row, 0x40');e('BRA DFGR_ROW','--:-:-:Y:d')
for _,body in ipairs(prefixes)do o[#o+1]=body end
replace('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
replace('ISETP.GE.U32.AND P6, PT, op, 0x73, PT;','ISETP.GE.U32.AND P6, PT, op, 0x74, PT;')
replace('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA dfgr_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
replace('\nPOSITION_END:','\n'..table.concat(o,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'w'));f:write(s);f:close()
