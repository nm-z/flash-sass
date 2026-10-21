-- Retain the existing parallel attention partitions. A merge head joins only
-- its four producing CTAs (each supplies two chunks), not the entire grid.
local function read(p)local f=assert(io.open(p));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local out=assert(arg[3])
local maps={}
local function map(src,a,b)
 for line in assert(src:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(a..'_',1,true)then maps[#maps+1]=line:gsub(a,b)end
 end
end
map(base,'attnpartn','partmerge');map(s,'attnquant','pmmerge');map(s,'attnq','pmq')
maps[#maps+1]=[=[    16,17 : pmjoin_ptr<0-1>
    18 : pmjoin_epoch
    19 : pmjoin_layer
    20 : pmjoin_tmp
    21 : pmjoin_value
    22 : pmjoin_group
    23 : pmjoin_half
]=]
local body=assert(base:match('(attnpartn_SETUP:.-)\nattnqkq_SETUP:')):gsub('attnpartn','partmerge')
local tail='--:-:-:-:6  @P6 BRA partmerge_BEGIN;\n--:-:-:-:6      BRA POSITION_END;'
local at=assert(body:find(tail,1,true))
body=body:sub(1,at-1)..[=[--:-:-:-:6 @P6 BRA partmerge_BEGIN;
3f:-:-:-:6 MEMBAR.GL;
--:-:-:-:6 BAR.SYNC 0x0;
--:-:1:-:1 LDS.64 pmjoin_ptr0, [RZ+0xbfa8];
--:-:1:-:1 LDS pmjoin_epoch, [RZ+0xbfb8];
--:-:1:-:2 LDS pmjoin_layer, [RZ+0xbfc0];
01:-:-:-:6 IADD pmjoin_epoch, pmjoin_epoch, -0x1;
--:-:-:-:6 XMAD pmjoin_epoch, pmjoin_epoch, 0xd, pmjoin_layer;
--:-:-:-:6 SHL pmjoin_tmp, cta, 0x2;
--:-:-:-:6 IADD pmjoin_tmp, pmjoin_tmp, 0x300;
--:-:-:-:6 IADD pmjoin_ptr0.CC, pmjoin_ptr0, pmjoin_tmp;
--:-:-:-:6 IADD.X pmjoin_ptr1, pmjoin_ptr1, RZ;
--:-:-:-:6 ISETP.EQ.U32.AND P5, PT, tid, RZ, PT;
--:2:-:-:6 STS [RZ+0x7a00], pmjoin_epoch;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:2:-:-:2 @P5 STG.E.WT [pmjoin_ptr], pmjoin_epoch;
--:-:-:-:6 ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;
02:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA PART_MERGE_PARAMS_READY;
--:-:-:-:6 IADD pmjoin_tmp, pidx, 0x1;
--:-:-:-:6 SHL pmjoin_tmp, pmjoin_tmp, 0x7;
--:-:-:-:6 IADD pmjoin_ptr0.CC, prog0, pmjoin_tmp;
--:-:-:-:6 IADD.X pmjoin_ptr1, prog1, RZ;
--:-:-:-:6 SHL pmjoin_tmp, lane, 0x2;
--:-:-:-:6 IADD pmjoin_ptr0.CC, pmjoin_ptr0, pmjoin_tmp;
--:-:-:-:6 IADD.X pmjoin_ptr1, pmjoin_ptr1, RZ;
--:-:1:-:2 LDG.E pmjoin_value, [pmjoin_ptr];
01:2:-:-:6 STS [pmjoin_tmp+0xbf80], pmjoin_value;
02:-:-:-:6 NOP;
PART_MERGE_PARAMS_READY:
--:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 MOV vid, cta;
--:-:-:-:6 MOV vtid, tid;
--:-:-:-:6 MOV32I units, 0x18;
--:-:-:-:6 BRA pmmerge_BEGIN;
]=]
local merge=assert(s:match('(attnquant_BEGIN:.-)\nselectone_SETUP:')):gsub('attnquant','pmmerge'):gsub('attnq_','pmq_')
local wait=[=[pmmerge_BEGIN:
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, warp, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA PM_HEAD_READY;
--:-:1:-:1 LDS.64 pmjoin_ptr0, [RZ+0xbf90];
--:-:1:-:2 LDS pmjoin_epoch, [RZ+0x7a00];
--:-:-:-:6 ISETP.GE.U32.AND P4, PT, vid, 0xc, PT;
--:-:-:-:6 MOV pmjoin_tmp, vid;
--:-:-:-:6 MOV pmjoin_group, RZ;
--:-:-:-:6 NOP;
--:-:-:-:6 @P4 IADD pmjoin_tmp, pmjoin_tmp, -0xc;
--:-:-:-:6 @P4 MOV32I pmjoin_group, 0x1;
--:-:-:-:6 ISETP.GE.U32.AND P5, PT, pmjoin_tmp, 0x6, PT;
--:-:-:-:6 LOP32I.AND pmjoin_tmp, lane, 0x3;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P5 IADD pmjoin_group, pmjoin_group, 0x2;
--:-:-:-:6 ISCADD pmjoin_tmp, pmjoin_tmp, pmjoin_group, 0x2;
--:-:-:-:6 SHL pmjoin_tmp, pmjoin_tmp, 0x2;
--:-:-:-:6 IADD pmjoin_tmp, pmjoin_tmp, 0x300;
01:-:-:-:6 IADD pmjoin_ptr0.CC, pmjoin_ptr0, pmjoin_tmp;
--:-:-:-:6 IADD.X pmjoin_ptr1, pmjoin_ptr1, RZ;
PM_HEAD_WAIT:
--:-:1:-:2 LDG.E.CV pmjoin_value, [pmjoin_ptr];
01:-:-:-:6 ISETP.EQ.U32.AND P5, PT, pmjoin_value, pmjoin_epoch, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 VOTE.ALL P5, P5;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P5 BRA PM_HEAD_WAIT;
PM_HEAD_READY:
--:-:-:-:6 MEMBAR.GL.IVALLD;
--:-:-:-:6 BAR.SYNC 0x0;
]=]
merge=wait..merge:sub(#'pmmerge_BEGIN:'+1)
local function rep(a,b)local i=assert(s:find(a,1,true),'missing '..a);s=s:sub(1,i-1)..b..s:sub(i+#a)end
rep('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
rep('ISETP.GE.U32.AND P6, PT, op, 0x68, PT;','ISETP.GE.U32.AND P6, PT, op, 0x69, PT;')
rep('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA partmerge_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
rep('POSITION_END:',body..'\n'..merge..'\nPOSITION_END:')
local f=assert(io.open(out,'w'));f:write(s);f:close()
