-- One warp per speculative position transports one complete 64-row MoE tile.
-- Odd -> even local gather; three-board reduce/scatter; three-board broadcast;
-- even -> odd local forward. Scalar and B4 use the same FP32 addition order.
return function(maps)
 maps[#maps+1]=[=[
    16,17 : pt_table<0-1>
    18,19 : pt_src<0-1>
    20,21 : pt_dst<0-1>
    22,23 : pt_flag<0-1>
    24 : pt_offset
    25 : pt_tile
    26 : pt_epoch
    27 : pt_die
    28 : pt_board
    29 : pt_owner
    30 : pt_tmp
    31 : pt_parity
    32,33 : pt_v<0-1>
    34,35 : pt_buddy<0-1>
    36-41 : pt_part<0-5>
    42 : pt_word
    43 : pt_timer
    44 : pt_start
    46 : pt_descriptor
    48,49 : pt_lookup<0-1>
    50 : pt_flag_offset
    51 : pt_data_offset
    52 : pt_job
    53 : pt_pos
    54 : pt_slot
    55 : pt_row0
    56 : pt_jobs
]=]
 local o={};local function e(x,c)o[#o+1]=(c or '--:-:-:-:6')..' '..x..';'end
 local function l(x)o[#o+1]=x..':'end
 local function pred(x)e(x,'--:-:-:Y:d');e('NOP')end
 local function wait(m)e('NOP',m..':-:-:-:6')end
 local function addr(dst,base,offset)
  e('IADD '..dst..'0.CC, '..base..'0, '..offset);e('IADD.X '..dst..'1, '..base..'1, RZ')
 end
 local function pointer(dst,index,owner)
  if owner then
   e('SHL pt_offset, '..owner..', 0x3');e(string.format('IADD pt_offset, pt_offset, 0x%x',index*8))
   addr('pt_lookup','pt_table','pt_offset');e('LDG.E.CG.64 '..dst..'0, [pt_lookup]','--:-:1:-:2')
  else e(string.format('LDG.E.CG.64 %s0, [pt_table+0x%x]',dst,index*8),'--:-:1:-:2')end
  wait('01')
 end
 local function payload(dst)
  e('STG.E.WT ['..dst..'], pt_v0','--:2:-:-:2');e('STG.E.WT ['..dst..'+0x80], pt_v1','--:2:-:-:2');wait('02')
 end
 local function signal(index,owner)
  pointer('pt_flag',index,owner);addr('pt_flag','pt_flag','pt_flag_offset')
  e('@P0 STG.E.WT [pt_flag], pt_epoch','--:2:-:-:2');wait('02')
 end
 local function poll(tag)
  l(tag);e('LDG.E.CV pt_word, [pt_flag]','--:-:1:-:2');wait('01')
  pred('ISETP.GE.U32.AND P5, PT, pt_word, pt_epoch, PT');pred('VOTE.ALL P5, P5');e('@P5 BRA '..tag..'_DONE')
  e('S2R pt_timer, SR_GLOBALTIMERHI','--:-:6:-:2');e('LDS pt_start, [RZ+0xbf5c]','--:-:1:-:2')
  e('IADD pt_timer, pt_timer, -pt_start','21:-:-:-:6');pred('ISETP.GE.U32.AND P5, PT, pt_timer, 0x46, PT')
  pred('VOTE.ANY P5, P5');e('@P5 BRA BEXP');e('BRA '..tag)
  l(tag..'_DONE');e('MEMBAR.GL.IVALLD')
 end
 -- Jobs are (tile iteration, position): CTAs below 8 own three tiles, the rest two.
 e('MOV pt_job, warp')
 pred('ISETP.LT.U32.AND P6, PT, cta, 0x8, PT');e('MOV pt_jobs, 0x2');e('@P6 MOV pt_jobs, 0x3')
 e('XMAD pt_jobs, pt_jobs, pc_count, RZ')
 l('PT_JOB')
 pred('ISETP.GE.U32.AND P6, PT, pt_job, pt_jobs, PT');e('@P6 BRA PT_JOBS_DONE')
 pred('ISETP.EQ.U32.AND P6, PT, pc_count, 0x1, PT')
 e('@P6 MOV pt_pos, RZ');e('@P6 MOV pt_slot, pt_job')
 e('@!P6 LOP32I.AND pt_pos, pt_job, 0x3');e('@!P6 SHR.U32 pt_slot, pt_job, 0x2')
 e('SHL pt_slot, pt_slot, 0xa');e('SHL pt_row0, cta, 0x6');e('IADD pt_row0, pt_row0, pt_slot')
 e('SHL pt_descriptor, pt_pos, 0x7');e('IADD pt_descriptor, pt_descriptor, 0x8000')
 e('LDS.64 pt_table0, [pt_descriptor+0x30]','--:-:1:-:2');e('LDS pt_die, [pt_descriptor+0x2c]','--:-:1:-:2')
 e('MOV pt_epoch, pc_epoch');e('SHR.U32 pt_tile, pt_row0, 0x6');e('SHL pt_flag_offset, pt_tile, 0x2')
 e('IADD pt_data_offset, pt_row0, lane');e('SHL pt_data_offset, pt_data_offset, 0x2')
 e('SHL pt_offset, lane, 0x2');e('ISCADD pt_offset, pt_pos, pt_offset, 0x8');e('IADD pt_offset, pt_offset, pt_slot')
 e('LDS pt_v0, [pt_offset+0x3000]','--:-:1:-:2');e('LDS pt_v1, [pt_offset+0x3080]','--:-:1:-:2');wait('01')
 e('SHR.U32 pt_board, pt_die, 0x1');e('LOP32I.AND pt_parity, pt_die, 0x1')
 e('MOV32I pt_tmp, 0xaaab');e('XMAD pt_owner, pt_tile, pt_tmp, RZ');e('SHR.U32 pt_owner, pt_owner, 0x11')
 e('XMAD pt_tmp, pt_owner, 0x3, RZ');e('IADD pt_owner, pt_tile, -pt_tmp')
 pred('ISETP.EQ.U32.AND P0, PT, lane, RZ, PT')
 pred('ISETP.EQ.U32.AND P6, PT, pt_parity, RZ, PT');e('@P6 BRA PT_LEADER')
 -- Odd dies send only within the physical board, then await the final tile.
 pointer('pt_dst',0);addr('pt_dst','pt_dst','pt_data_offset');payload('pt_dst');e('MEMBAR.SYS');signal(1)
 pointer('pt_flag',7);addr('pt_flag','pt_flag','pt_flag_offset');poll('PT_ODD_FINAL')
 e('BRA PT_NEXT_JOB')
 l('PT_LEADER')
 pointer('pt_flag',3);addr('pt_flag','pt_flag','pt_flag_offset');poll('PT_PAIR_READY')
 pointer('pt_src',2);addr('pt_src','pt_src','pt_data_offset')
 e('LDG.E.CG pt_buddy0, [pt_src]','--:-:1:-:2');e('LDG.E.CG pt_buddy1, [pt_src+0x80]','--:-:1:-:2');wait('01')
 e('FADD pt_v0, pt_v0, pt_buddy0');e('FADD pt_v1, pt_v1, pt_buddy1')
 pointer('pt_dst',10,'pt_owner');addr('pt_dst','pt_dst','pt_data_offset');payload('pt_dst');e('MEMBAR.SYS');signal(13,'pt_owner')
 pred('ISETP.NE.U32.AND P6, PT, pt_owner, pt_board, PT');e('@P6 BRA PT_WAIT_FINAL')
 -- The row-owner board observes all three board sums before reduction.
 pointer('pt_flag',5);e('IMNMX.U32 pt_tmp, lane, 0x2, PT')
 e('XMAD pt_offset, pt_tmp, 0xa0, pt_flag_offset');addr('pt_flag','pt_flag','pt_offset');poll('PT_REDUCE_READY')
 pointer('pt_src',4);addr('pt_src','pt_src','pt_data_offset')
 for b=0,2 do
  e(string.format('LDG.E.CG pt_part%d, [pt_src+0x%x]',b*2,b*10240),'--:-:1:-:2')
  e(string.format('LDG.E.CG pt_part%d, [pt_src+0x%x]',b*2+1,b*10240+128),'--:-:1:-:2')
 end
 wait('01');e('FADD pt_v0, pt_part0, pt_part2');e('FADD pt_v1, pt_part1, pt_part3')
 e('FADD pt_v0, pt_v0, pt_part4');e('FADD pt_v1, pt_v1, pt_part5')
 for b=0,2 do pointer('pt_dst',16+b);addr('pt_dst','pt_dst','pt_data_offset');payload('pt_dst')end
 e('MEMBAR.SYS');for b=0,2 do signal(19+b)end
 l('PT_WAIT_FINAL');pointer('pt_flag',7);addr('pt_flag','pt_flag','pt_flag_offset');poll('PT_BROADCAST_READY')
 pointer('pt_src',6);addr('pt_src','pt_src','pt_data_offset')
 e('LDG.E.CG pt_v0, [pt_src]','--:-:1:-:2');e('LDG.E.CG pt_v1, [pt_src+0x80]','--:-:1:-:2');wait('01')
 pointer('pt_dst',8);addr('pt_dst','pt_dst','pt_data_offset');payload('pt_dst');e('MEMBAR.SYS');signal(9)
 e('BRA PT_NEXT_JOB')
 l('PT_NEXT_JOB');e('IADD pt_job, pt_job, 0x8');e('BRA PT_JOB')
 l('PT_JOBS_DONE')
 return table.concat(o,'\n')..'\n'
end
