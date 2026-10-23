-- Ordinary allgather forwarding runs before the normal phase footer. Producer
-- CTA c publishes its own payload-ready epoch to three board leaders. Leader
-- CTA c forwards only the ranges assigned to c by the host ownership ledger.
-- Other CTAs can still be computing. The old final exchange follows this stage.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i,j=s:find(a,1,true);assert(i,'missing '..a);assert(not s:find(a,j+1,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local maps=[=[
    16,17 : bt_control<0-1>
    18,19 : bt_plan<0-1>
    20,21 : bt_flags<0-1>
    22,23 : bt_targets<0-1>
    24,25 : bt_buddy<0-1>
    26,27 : bt_ptr<0-1>
    28 : bt_temp
    29 : bt_epoch
    30 : bt_die
    31 : bt_parity
    32,33 : bt_list<0-1>
    32-35 : bt_header<0-3>
    34 : bt_count
    35 : bt_active
    36 : bt_record
    37 : bt_word
    38,39 : bt_source<0-1>
    40,41 : bt_dest<0-1>
    42 : bt_bytes
    44 : bt_byte
    45 : bt_value
    46 : bt_offset
    47 : bt_timer
    48 : bt_start
]=]
local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
local function l(x)o[#o+1]=x..':'end
local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
local function wait(m)e('NOP',m..':-:-:-:6')end
local function addr(dst,base,offset)
 e('IADD '..dst..'0.CC, '..base..'0, '..offset);e('IADD.X '..dst..'1, '..base..'1, RZ')
end
local function poll(label)
 l(label);e('LDG.E.CV bt_word, [bt_ptr]','--:-:1:-:2');wait('01')
 pred('ISETP.GE.U32.AND P4, PT, bt_word, bt_epoch, PT');pred('VOTE.ALL P4, P4');e('@P4 BRA '..label..'_DONE')
 e('S2R bt_timer, SR_GLOBALTIMERHI','--:-:6:-:2');e('LDS bt_start, [RZ+0xbf5c]','--:-:1:-:2')
 e('IADD bt_timer, bt_timer, -bt_start','21:-:-:-:6');pred('ISETP.GE.U32.AND P5, PT, bt_timer, 0x46, PT')
 pred('VOTE.ANY P5, P5');e('@P5 BRA BEXP');e('BRA '..label);l(label..'_DONE')
 e('MEMBAR.GL.IVALLD')
end
wait('3f');e('LDS bt_active, [RZ+0xbf58]','--:-:1:-:2');wait('01')
e('LOP32I.AND bt_active, bt_active, 0x1');pred('ISETP.EQ.U32.AND P6, PT, bt_active, RZ, PT');e('@P6 BRA BOARD_TRANSPORT_DONE')
e('MOV bt_control0, ctl0');e('MOV bt_control1, ctl1')
e('LDG.E.CG.64 bt_plan0, [bt_control+0xe0]','--:-:1:-:2');e('SHL bt_temp, pidx, 0x8');e('ISCADD bt_temp, cta, bt_temp, 0x4');wait('01')
addr('bt_plan','bt_plan','bt_temp');e('LDG.E.CG.128 bt_header0, [bt_plan]','--:-:1:-:2');wait('01')
pred('ISETP.EQ.U32.AND P6, PT, bt_active, RZ, PT');e('@P6 BRA BOARD_TRANSPORT_DONE')
e('LDG.E.CG.64 bt_flags0, [bt_control+0xe8]','--:-:1:-:2')
e('LDG.E.CG.64 bt_targets0, [bt_control+0xf0]','--:-:1:-:2')
e('LDG.E.CG.64 bt_buddy0, [bt_control+0xf8]','--:-:1:-:2')
e('LDG.E.CG bt_die, [bt_control+0xc]','--:-:3:-:2')
e('LDS bt_epoch, [RZ+0xbf74]','--:-:3:-:2');wait('05')
e('IADD bt_epoch, bt_epoch, 0x1');e('LOP32I.AND bt_parity, bt_die, 0x1')
-- Every writer fences before CTA thread zero publishes payload readiness.
e('MEMBAR.SYS');e('BAR.SYNC 0x0')
pred('ISETP.EQ.U32.AND P0, PT, tid, RZ, PT');e('SHL bt_offset, cta, 0x2')
for b=0,2 do
 e(string.format('@P0 LDG.E.CG.64 bt_ptr0, [bt_targets+0x%x]',b*8),'--:-:1:-:2');wait('01')
 addr('bt_ptr','bt_ptr','bt_offset');e('@P0 STG.E.WT [bt_ptr], bt_epoch','--:2:-:-:2');wait('02')
end
pred('ISETP.EQ.U32.AND P6, PT, bt_parity, RZ, PT');e('@P6 BRA BOARD_LEADER')
pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA BOARD_ODD_JOIN')
addr('bt_ptr','bt_buddy','bt_offset');poll('BOARD_ODD_WAIT')
l('BOARD_ODD_JOIN');e('BAR.SYNC 0x0');e('BRA BOARD_TRANSPORT_DONE')
l('BOARD_LEADER');pred('ISETP.NE.U32.AND P6, PT, warp, RZ, PT');e('@P6 BRA BOARD_LEADER_JOIN')
e('IMNMX.U32 bt_temp, lane, 0x5, PT');e('ISCADD bt_temp, bt_temp, bt_offset, 0x6');addr('bt_ptr','bt_flags','bt_temp')
poll('BOARD_SOURCE_WAIT')
l('BOARD_LEADER_JOIN');e('BAR.SYNC 0x0');e('MEMBAR.GL.IVALLD');e('MOV bt_record, warp')
l('BOARD_COPY_RECORD');pred('ISETP.GE.U32.AND P6, PT, bt_record, bt_count, PT');e('@P6 BRA BOARD_COPY_DONE')
e('XMAD bt_temp, bt_record, 0x18, RZ');addr('bt_ptr','bt_list','bt_temp')
e('LDG.E.CG.64 bt_source0, [bt_ptr]','--:-:1:-:2');e('LDG.E.CG.64 bt_dest0, [bt_ptr+0x8]','--:-:1:-:2')
e('LDG.E.CG bt_bytes, [bt_ptr+0x10]','--:-:3:-:2');wait('05');e('SHL bt_byte, lane, 0x2')
addr('bt_source','bt_source','bt_byte');addr('bt_dest','bt_dest','bt_byte');e('MOV bt_offset, RZ')
l('BOARD_COPY_WORDS');pred('ISETP.LT.U32.AND P0, PT, bt_byte, bt_bytes, PT')
e('@P0 LDG.E.CG bt_value, [bt_source]','--:2:1:-:2');e('@P0 STG.E.WT [bt_dest], bt_value','01:3:-:-:2');wait('06')
e('IADD bt_source0.CC, bt_source0, 0x80');e('IADD.X bt_source1, bt_source1, RZ')
e('IADD bt_dest0.CC, bt_dest0, 0x80');e('IADD.X bt_dest1, bt_dest1, RZ')
e('IADD bt_byte, bt_byte, 0x80');e('IADD bt_offset, bt_offset, 0x80')
pred('ISETP.LT.U32.AND P6, PT, bt_offset, bt_bytes, PT');e('@P6 BRA BOARD_COPY_WORDS')
e('IADD bt_record, bt_record, 0x8');e('BRA BOARD_COPY_RECORD')
l('BOARD_COPY_DONE');e('MEMBAR.SYS','3f:-:-:Y:d');e('BAR.SYNC 0x0')
pred('ISETP.EQ.U32.AND P0, PT, tid, RZ, PT');e('SHL bt_offset, cta, 0x2');addr('bt_ptr','bt_buddy','bt_offset')
e('@P0 STG.E.WT [bt_ptr], bt_epoch','--:2:-:-:2');wait('02');e('BAR.SYNC 0x0')
l('BOARD_TRANSPORT_DONE')
replace('</REGISTER_MAPPING>',maps..'\n</REGISTER_MAPPING>')
replace('\nPHASE_END:','\nPHASE_END:\n'..table.concat(o,'\n'))
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
