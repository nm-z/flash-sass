-- Same-owner dependencies stay inside one operator. No grid barrier is hidden
-- inside these bodies: router -> selection is warp-local; GDN -> norm is CTA-local.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local output=assert(arg[3])
local maps,bodies={},{}
local function mapping(src,old,new)
 for line in assert(src:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
  if line:find(old..'_',1,true)then maps[#maps+1]=line:gsub(old,new)end
 end
end
local function replace(src,a,b)
 local i=assert(src:find(a,1,true),'missing '..a)
 return src:sub(1,i-1)..b..src:sub(i+#a)
end
mapping(base,'router','routesel');mapping(base,'moesel','rselect')
local router=assert(base:match('(router_SETUP:.-)\nattnqk_SETUP:')):gsub('router','routesel')
local selection=assert(base:match('(moesel_BEGIN:.-)\nmoesel_END:')):gsub('moesel','rselect')
-- Keep the public router outputs, but hand the ten indices/weights to the next
-- operation through CTA shared memory. Only warp zero participates in either.
for j=0,9 do
 local a=string.format('--:2:-:-:1  @P1 STG.E [routesel_op+0x%x], routesel_idx%d;',4*j,j)
 router=replace(router,a,a..string.format('\n--:2:-:-:1  @P1 STS [RZ+0x%x], routesel_idx%d;',0x2000+4*j,j))
 a=string.format('--:2:-:-:1  @P1 STG.E [routesel_op+0x%x], routesel_e%d;',4*j,j)
 router=replace(router,a,a..string.format('\n--:2:-:-:1  @P1 STS [RZ+0x%x], routesel_e%d;',0x2040+4*j,j))
end
-- Selection's five additional pointers occupy descriptor words 12..21.
local offsets={ [0xbf88]=0xbf98,[0xbf8c]=0xbf9c,[0xbf90]=0xbfa0,[0xbf94]=0xbfa4 }
for n=0,9 do offsets[0xbf98+n*4]=0xbfb0+n*4 end
selection=selection:gsub('%[RZ%+0x(%x+)%]',function(h)local x=tonumber(h,16);return string.format('[RZ+0x%x]',offsets[x] or x)end)
selection=replace(selection,'--:-:1:-:1  @P0 LDG.E rselect_id, [rselect_ap];',
 '--:-:-:-:6      SHL rselect_t, rselect_lane, 0x2;\n--:-:1:-:1  @P0 LDS rselect_id, [rselect_t+0x2000];')
selection=replace(selection,'--:-:1:-:2  @P0 LDG.E rselect_w, [rselect_bp];',
 '--:-:1:-:2  @P0 LDS rselect_w, [rselect_t+0x2040];')
-- Remove selection's original branch to its old per-unit epilogue.
selection=selection:gsub('BRA rselect_END;', 'BRA POSITION_END;')
local at=assert(router:find('routesel_END:',1,true))
router=router:sub(1,at-1)..'routesel_END:\n3f:-:-:-:6 NOP;\n'..selection..'\n'
bodies[#bodies+1]=router

mapping(base,'dnstep','dnlocal');mapping(s,'dnquant','dnlocalnorm');mapping(s,'dnq','dnlocalq')
local dn=assert(base:match('(dnstep_SETUP:.-)\ndnnorm_SETUP:')):gsub('dnstep','dnlocal')
local norm=assert(s:match('(dnquant_BEGIN:.-)\ndnquant_END:')):gsub('dnquant','dnlocalnorm'):gsub('dnq_','dnlocalq_')
-- The state-update CTA owns an entire head. Its part-zero threads write four
-- adjacent output elements apiece; warp zero reads the same 128 values for RMS.
local store='--:2:-:-:1  @P3 STG.E.128 [dnlocal_op], dnlocal_o0;'
dn=replace(dn,store,store..'\n--:-:-:-:6 SHL dnlocal_t, dnlocal_g, 0x4;\n--:2:-:-:1 @P3 STS.128 [dnlocal_t+0x2400], dnlocal_o0;')
local noff={ [0xbf88]=0xbfc0,[0xbf8c]=0xbfc4,[0xbf90]=0xbfd8,[0xbf94]=0xbfdc,
 [0xbf98]=0xbfe0,[0xbf9c]=0xbfe4,[0xbfa0]=0xbfe8,[0xbfa4]=0xbfec,[0xbfa8]=0xbfc8,[0xbfb0]=0xbff0 }
norm=norm:gsub('%[RZ%+0x(%x+)%]',function(h)local x=tonumber(h,16);return string.format('[RZ+0x%x]',noff[x] or x)end)
norm=replace(norm,'--:-:1:-:1      LDG.E.128 dnlocalnorm_x0, [dnlocalnorm_xp];',
 '--:-:-:-:6 SHL dnlocalnorm_t, lane, 0x4;\n--:-:1:-:1 LDS.128 dnlocalnorm_x0, [dnlocalnorm_t+0x2400];')
norm=norm:gsub('BRA dnlocalnorm_END;', 'BRA dnlocal_JOIN;')
local epilogue=assert(dn:match('(dnlocal_END:.*)'))
epilogue=epilogue:gsub('dnlocal_END:','dnlocal_JOIN:',1)
at=assert(dn:find('dnlocal_END:',1,true))
dn=dn:sub(1,at-1)..[=[dnlocal_END:
3f:-:-:-:6 BAR.SYNC 0x0;
--:-:-:-:6 ISETP.NE.U32.AND P6, PT, warp, RZ, PT;
--:-:-:-:6 MOV vtid, lane;
--:-:-:-:6 NOP;
--:-:-:-:6 NOP;
--:-:-:-:6 @P6 BRA dnlocal_JOIN;
]=]..norm..'\n'..epilogue
-- vtid must be restored for the next head when a CTA loops over multiple heads.
dn=replace(dn,'dnlocal_BEGIN:','dnlocal_BEGIN:\n--:-:-:-:6 MOV vtid, tid;')
bodies[#bodies+1]=dn
s=replace(s,'</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
s=replace(s,'ISETP.GE.U32.AND P6, PT, op, 0x61, PT;','ISETP.GE.U32.AND P6, PT, op, 0x63, PT;')
s=replace(s,'--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6 BRA routesel_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 BRA dnlocal_SETUP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
s=replace(s,'POSITION_END:',table.concat(bodies,'\n')..'\nPOSITION_END:')
local f=assert(io.open(output,'wb'));f:write(s);f:close()
