-- Preserve the footer's address lifetimes independently of descriptor prefetch.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local function replace(a,b)
 local i=assert(s:find(a,1,true),'missing '..a);assert(not s:find(a,i+#a,true),'ambiguous '..a)
 s=s:sub(1,i-1)..b..s:sub(i+#a)
end
replace('--:-:6:-:1      S2R tid, SR_TID.X;', '--:-:5:-:1      S2R tid, SR_TID.X;')
replace('20:-:-:-:6      LOP32I.AND lane, tid, 0x1f;', '30:-:-:-:6      LOP32I.AND lane, tid, 0x1f;')
-- Descriptor payload and token-offset mask are independent global loads.
-- Keep both completions live across the footer and wait for both consumers.
local masks,consumers=0,0
s=s:gsub('([^\n]*):4:%-:2([^\n]*LDG%.E vid, %[dsp%+0x7c%];)',function(a,b)masks=masks+1;return a..':5:-:2'..b end)
s=s:gsub('09:%-:%-:%-:6      SHR%.U32 vid, vid, lane;',function()consumers=consumers+1;return '19:-:-:-:6      SHR.U32 vid, vid, lane;'end)
assert(masks==consumers+1 and (masks==3 or masks==6),'unexpected descriptor completion geometry '..masks..'/'..consumers)
-- Do not reuse a pending descriptor-mask completion barrier for the mixed
-- S2R/LDS timeout pair. Different pipelines can complete out of issue order.
local clocks,clock_uses=0,0
s=s:gsub('%-%-:%-:6:%-:2([^\n]*LDS bar_old, %[RZ%+0xbf5c%];)',function(rest)clocks=clocks+1;return '--:-:1:-:2'..rest end)
s=s:gsub('20:%-:%-:%-:6([^\n]*IADD (bar_[%w]+),[^\n]*%-bar_old;)',function(rest,reg)
 if reg=='bar_t' or reg=='bar_g2' then clock_uses=clock_uses+1;return '21:-:-:-:6'..rest end
 return '20:-:-:-:6'..rest
end)
assert(clocks==clock_uses and clocks>=3,'unexpected timeout completion geometry')
replace('--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, cta, RZ, P4;',
        '--:-:-:Y:d      ISETP.EQ.U32.AND P2, PT, cta, RZ, P4;')
local old='20:3:-:-:6  @P4 STG.E [bar_cp+0x100], bar_g2;'
if not s:find(old,1,true)then old='20:-:-:-:6  @P4 STG.E [bar_cp+0x100], bar_g2;'end
local cursor,count,out=1,0,{}
while true do local i,j=s:find(old,cursor,true);if not i then break end
 out[#out+1]=s:sub(cursor,i-1)
 out[#out+1]='22:2:-:-:6  @P4 STG.E [bar_cp+0x100], bar_g2;\n02:-:-:-:6      NOP; # retire timestamp address/data independently of prefetch'
 cursor=j+1;count=count+1
end
assert(count==1 or count==2,'unexpected footer/chain timestamp count')
out[#out+1]=s:sub(cursor);s=table.concat(out)
local advance=s:find('05:-:-:-:6      IADD pidx, pidx, bar_t;',1,true)and '05' or '01'
replace(advance..':-:-:-:6      IADD pidx, pidx, bar_t;',
        '07:-:-:-:6      IADD pidx, pidx, bar_t;')
-- The host polls ctl[2] and then copies TOKS while this persistent launch is
-- active. Order token/ID/state stores at system scope before publishing that
-- progress; peer-only publication fences do not cover a final non-send phase.
replace('# ---- token done:', '3f:-:-:-:6 MEMBAR.SYS; # release completed model stores before host progress\n# ---- token done:')
-- Every publishing thread must order its own peer writes before the CTA's
-- representative publishes the flag. A fence executed only by thread zero
-- does not order writes issued by the other seven warps.
replace('PHASE_END:',[=[PHASE_END:
3f:-:1:-:2 LDS bar_old, [RZ+0xbf58];
01:-:-:-:6 LOP32I.AND bar_old, bar_old, 0x1;
--:-:-:-:6 ISETP.NE.U32.AND P0, PT, bar_old, RZ, PT;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P0 MEMBAR.SYS;
]=])
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
