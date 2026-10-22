-- Diagnostic image only. All times are each physical GPU's GLOBALTIMER domain.
-- Per [position][descriptor], 2048 bytes: 16 CTA records of 8 u64,
-- followed by CTA0's first observations of the 96 producer flags.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local maps=[=[    64,65 : xt_ptr<0-1>
    66,67 : xt_window<0-1>
    66 : xt_first
    67 : xt_count
    68 : xt_token
    69 : xt_offset
    70,71 : xt_ctl<0-1>
    72,73 : xt_clock<0-1>
    74 : xt_hi_after
    75 : xt_tmp
    78,79 : xt_epoch<0-1>
    80,81 : xt_hash<0-1>
    82 : xt_value
    83 : xt_shift
    84 : xt_shift2
    85 : xt_iter
    200-202 : xt_seen<0-2>
]=]
local function insert(a,b)
 local i=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,i+#a,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(i+#a)
end
insert('</REGISTER_MAPPING>',maps..'</REGISTER_MAPPING>')
local function address(tag,arrival)
 return ([=[
3f:-:-:-:6 NOP;
--:-:-:-:6 ISETP.NE.U32.AND P1, PT, warp, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P1 BRA @_DONE;
%CTA%
--:-:-:-:6 MOV xt_ctl0, ctl0;
--:-:-:-:6 MOV xt_ctl1, ctl1;
--:-:1:-:1 LDG.E.64 xt_ptr0, [xt_ctl+0xe8];
--:-:2:-:1 LDG.E.64 xt_window0, [xt_ctl+0xf0];
--:-:3:-:2 LDS xt_token, [RZ+0xbf70];
06:-:-:-:6 IADD xt_token, xt_token, -xt_first;
--:-:-:-:6 ISETP.LT.U32.AND P1, PT, xt_token, xt_count, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @!P1 BRA @_DONE;
--:-:-:-:6 SHL xt_offset, xt_token, 0xc;
--:-:-:-:6 IADD xt_offset, xt_offset, pidx;
--:-:-:-:6 SHL xt_offset, xt_offset, 0xb;
01:-:-:-:6 IADD xt_ptr0.CC, xt_ptr0, xt_offset;
--:-:-:-:6 IADD.X xt_ptr1, xt_ptr1, RZ;
%OFFSET%
@_CLOCK:
--:-:1:-:1 S2R xt_clock1, SR_GLOBALTIMERHI;
--:-:2:-:1 S2R xt_clock0, SR_GLOBALTIMERLO;
--:-:3:-:2 S2R xt_hi_after, SR_GLOBALTIMERHI;
07:-:-:-:6 ISETP.NE.U32.AND P2, PT, xt_clock1, xt_hi_after, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P2 BRA @_CLOCK;
]=]):gsub('%%CTA%%',arrival and [=[--:-:-:-:6 ISETP.NE.U32.AND P1, PT, cta, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P1 BRA @_DONE;]=]or ''):gsub('%%OFFSET%%',arrival and [=[--:-:-:-:6 SHL xt_offset, lane, 0x3;
--:-:-:-:6 IADD xt_ptr0.CC, xt_ptr0, xt_offset;
--:-:-:-:6 IADD.X xt_ptr1, xt_ptr1, RZ;]=]or [=[--:-:-:-:6 SHL xt_offset, cta, 0x6;
--:-:-:-:6 IADD xt_ptr0.CC, xt_ptr0, xt_offset;
--:-:-:-:6 IADD.X xt_ptr1, xt_ptr1, RZ;]=]):gsub('@_',tag..'_')
end
local function stage(tag,n)
 local code=address(tag,false)..string.format([=[
--:-:-:-:6 ISETP.EQ.U32.AND P0, PT, tid, RZ, PT;
--:-:3:-:2 LDS xt_epoch0, [RZ+0xbf74];
--:-:-:-:6 MOV xt_epoch1, RZ;
04:-:-:-:6 IADD xt_epoch0, xt_epoch0, 0x%x;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:1:-:-:6 @P0 STG.E.64 [xt_ptr+0x%x], xt_clock0;
01:1:-:-:6 @P0 STG.E.64 [xt_ptr+0x%x], xt_epoch0;
01:-:-:-:6 NOP;
%s_DONE:
]=],n==0 and 1 or 0,n*8,n<2 and 0x20 or 0x28,tag)
 return code
end
local p=assert(s:find('\nPHASE_END:',1,true));local b,e=assert(s:find('[^\n]*BAR%.SYNC 0x1;\n',p))
s=s:sub(1,e)..stage('XT_READY',0)..s:sub(e+1)
insert('--:2:-:-:6  @P2 STG.E [bar_cp+0x20], op;',
       '--:2:-:-:6  @P2 STG.E [bar_cp+0x20], op;\n'..stage('XT_PUBLISH',1))
insert('XWSPIN:',[=[--:-:-:-:6 MOV xt_seen0, RZ;
--:-:-:-:6 MOV xt_seen1, RZ;
--:-:-:-:6 MOV xt_seen2, RZ;
]=]..stage('XT_WAIT_START',2)..'\nXWSPIN:')
insert('XWDONE:','XWDONE:\n'..stage('XT_WAIT_END',3))
local arrival=address('XT_ARRIVAL',true)..[=[
--:-:-:-:6 ISETP.GE.U32.AND P0, PT, bar_t, bar_g, PT;
--:-:-:-:6 ISETP.GE.U32.AND P1, PT, bar_one, bar_g, PT;
--:-:-:-:6 ISETP.GE.U32.AND P2, PT, bar_g2, bar_g, PT;
--:-:-:-:6 ISETP.NE.U32.AND P3, PT, xt_seen0, RZ, PT;
--:-:-:-:6 ISETP.NE.U32.AND P4, PT, xt_seen1, RZ, PT;
--:-:-:-:6 ISETP.NE.U32.AND P5, PT, xt_seen2, RZ, PT;
--:-:-:-:6 PSETP.AND.AND P0, PT, P0, !P3, PT;
--:-:-:-:6 PSETP.AND.AND P1, PT, P1, !P4, PT;
--:-:-:-:6 PSETP.AND.AND P2, PT, P2, !P5, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:1:-:-:2 @P0 STG.E.64 [xt_ptr+0x400], xt_clock0;
--:1:-:-:2 @P1 STG.E.64 [xt_ptr+0x500], xt_clock0;
--:1:-:-:2 @P2 STG.E.64 [xt_ptr+0x600], xt_clock0;
--:-:-:-:6 @P0 MOV32I xt_seen0, 0x1;
--:-:-:-:6 @P1 MOV32I xt_seen1, 0x1;
--:-:-:-:6 @P2 MOV32I xt_seen2, 0x1;
01:-:-:-:6 NOP;
XT_ARRIVAL_DONE:
]=]
insert('04:-:-:-:6      IMNMX.U32 bar_t, bar_t, bar_one, PT;',arrival..'\n04:-:-:-:6      IMNMX.U32 bar_t, bar_t, bar_one, PT;')
-- Fingerprint the immutable incoming residual before each GR body. Only
-- CTA0/warp0 participates, and only within the configured trace window.
-- This adds no phase and retains the production operation and dependencies.
if arg[3]=='gr_input_hash' then
for _,op in ipairs({'bf16hcrx','bf16hcrwx','grreadsum'})do
 local anchor=op..'_SETUP:'
 if s:find(anchor,1,true)then
  local tag='XT_GR_'..op
  local code=address(tag,true)..[=[
--:-:1:-:2 LDS.64 xt_ctl0, [RZ+0xbf90];
--:-:-:-:6 SHL xt_tmp, lane, 0x2;
01:-:-:-:6 IADD xt_ctl0.CC, xt_ctl0, xt_tmp;
--:-:-:-:6 IADD.X xt_ctl1, xt_ctl1, RZ;
--:-:-:-:6 MOV32I xt_hash0, 0x811c9dc5;
--:-:-:-:6 MOV32I xt_hash1, 0x9e3779b9;
--:-:-:-:6 MOV xt_iter, RZ;
@_HASH:
--:-:1:-:2 LDG.E.CV xt_value, [xt_ctl];
01:-:-:-:6 LOP.XOR xt_hash0, xt_hash0, xt_value;
--:-:-:-:6 SHL xt_shift, xt_hash0, 0x5;
--:-:-:-:6 SHR.U32 xt_shift2, xt_hash0, 0x1b;
--:-:-:-:6 LOP.OR xt_hash0, xt_shift, xt_shift2;
--:-:-:-:6 IADD xt_hash1, xt_hash1, xt_value;
--:-:-:-:6 SHL xt_shift, xt_hash1, 0x7;
--:-:-:-:6 SHR.U32 xt_shift2, xt_hash1, 0x19;
--:-:-:-:6 LOP.OR xt_hash1, xt_shift, xt_shift2;
--:-:-:-:6 IADD xt_ctl0.CC, xt_ctl0, 0x80;
--:-:-:-:6 IADD.X xt_ctl1, xt_ctl1, RZ;
--:-:-:-:6 IADD xt_iter, xt_iter, 0x1;
--:-:-:-:6 ISETP.LT.U32.AND P1, PT, xt_iter, 0x140, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P1 BRA @_HASH;
--:1:-:-:6 STG.E.64 [xt_ptr+0x700], xt_hash0;
01:-:-:-:6 NOP;
]=]
  code=code:gsub('@_',tag..'_')..tag..'_DONE:\n'
  insert(anchor,anchor..'\n'..code)
 end
end
end
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
