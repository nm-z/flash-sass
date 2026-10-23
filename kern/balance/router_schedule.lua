-- Schedule independent router comparisons and exponential chains together.
-- Keep the original float maxima, lowest-index tie rule, exponent polynomial,
-- normalization sum order, selected-rank order and publication path.
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local a=assert(s:find('routesel_SETUP:',1,true))
local b=assert(s:find('\n[%w_]+_SETUP:',a+1));local body=s:sub(a,b-1)
local function gen()
 local out={};local function e(x,c)out[#out+1]=(c or '--:-:-:-:1')..' '..x..';'end
 return out,e
end
local function block(first,last,replacement)
 local p=assert(body:find(first,1,true),'missing '..first)
 local start=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local _,finish=body:find(last,p,true);assert(finish,'missing '..last)
 body=body:sub(1,start-1)..replacement..body:sub(finish+1)
end
local ties,masks=0,0
for rank=0,9 do
 local first='LOP.XOR routesel_d, routesel_v00, routesel_mx;'
 local last='IMNMX.U32 routesel_ix, routesel_ix, routesel_t, PT;'
 local p=assert(body:find(first,1,true));local q=p
 -- The original tie pass contains exactly16 XOR/ICMP/min triplets.
 for j=0,15 do
  local x=string.format('ICMP.EQ routesel_t, routesel_kb%02d, routesel_ix, routesel_d;',j)
  q=assert(body:find(x,q,true));local _,stop=assert(body:find(last,q,true));q=stop+1
 end
 local start=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local old=body:sub(start,q-1)
 local _,n=old:gsub('LOP%.XOR','');assert(n==16,'tie comparison geometry')
 local out,e=gen()
 for j=0,15 do e(string.format('LOP.XOR rs_tmp%02d, routesel_v%02d, routesel_mx',j,j))end
 for j=0,15 do e(string.format('ICMP.EQ rs_tmp%02d, routesel_kb%02d, routesel_ix, rs_tmp%02d',j,j,j))end
 -- ICMP results take13 clocks. Six extra clocks plus the pair-tree issue
 -- order leave at least14 clocks before even the latest candidate is read.
 e('NOP','--:-:-:-:6')
 for _,stride in ipairs({1,2,4,8})do
  for j=0,15,2*stride do
   e(string.format('IMNMX.U32 rs_tmp%02d, rs_tmp%02d, rs_tmp%02d, PT',j,j,j+stride),
     j+2*stride>=16 and '--:-:-:-:6' or '--:-:-:-:1')
  end
 end
 e('MOV routesel_ix, rs_tmp00','--:-:-:-:6')
 body=body:sub(1,start-1)..table.concat(out,'\n')..body:sub(q);ties=ties+1
 -- All index differences and invalidations are independent. Keep every
 -- original selected value update, including the otherwise-unused last one.
 out,e=gen()
 for j=0,15 do e(string.format('IADD rs_tmp%02d, routesel_kb%02d, -routesel_ix',j,j))end
 for j=0,15 do e(string.format('ICMP.EQ routesel_v%02d, routesel_neginf, routesel_v%02d, rs_tmp%02d',j,j,j))end
 e('NOP','--:-:-:Y:d')
 block('IADD routesel_d, routesel_kb00, -routesel_ix;',
       'ICMP.EQ routesel_v15, routesel_neginf, routesel_v15, routesel_d;',table.concat(out,'\n'))
 masks=masks+1
end
-- Collect the ten identical exponent chains and check their exact structure
-- before interleaving. Register names are changed, arithmetic is not.
local chains,first,last={}
for rank=0,9 do
 local p=assert(body:find('FADD routesel_ft5, routesel_m'..rank..', -routesel_mtop;',1,true))
 local start=(body:sub(1,p-1):match('.*()\n') or 0)+1
 local _,stop=assert(body:find('FMUL routesel_e'..rank..', routesel_ft1, routesel_ft2;',p,true))
 if rank==0 then first=start else assert(body:sub(last+1,start-1):match('^%s*$'),'work between exponent chains')end
 last=stop;local ops={}
 for line in body:sub(start,stop):gmatch('[^\n]+')do
  local ins=assert(line:match('^%S+%s+(.+);$'))
  ins=ins:gsub('routesel_m'..rank..',','routesel_m@,'):gsub('routesel_e'..rank..',','routesel_e@,')
  ops[#ops+1]=ins
 end
 assert(#ops==18,'exponent polynomial shape')
 if rank>0 then assert(table.concat(ops,'\n')==table.concat(chains[0],'\n'),'exponent arithmetic differs')end
 chains[rank]=ops
end
local out,e=gen()
for step,template in ipairs(chains[0])do
 if step==16 then e('NOP','04:-:-:-:6')end -- all ten F2I results
 for rank=0,9 do
  local ins=template:gsub('@',tostring(rank))
  ins=ins:gsub('routesel_ft([1235])',function(n)return 'rs_t'..n..string.format('_%02d',rank)end)
  e(ins,step==15 and '--:-:3:-:2' or '--:-:-:-:1')
 end
end
body=body:sub(1,first-1)..table.concat(out,'\n')..body:sub(last+1)
s=s:sub(1,a-1)..body..s:sub(b)
local maps=[=[
    180-195 : rs_tmp<00-15>
    180-189 : rs_t1_<00-09>
    190-199 : rs_t2_<00-09>
    200-209 : rs_t3_<00-09>
    210-219 : rs_t5_<00-09>
]=]
local at=assert(s:find('</REGISTER_MAPPING>',1,true));s=s:sub(1,at-1)..maps..s:sub(at)
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
io.stderr:write(string.format('router_schedule tie_passes=%d invalidation_passes=%d independent_exponent_chains=10 extra_phases=0\n',ties,masks))
