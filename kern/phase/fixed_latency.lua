-- Correct identified fixed-pipeline producer/consumer gaps. MaxAs models
-- register-result ICMP/FCMP at13 cycles, distinct from six-cycle integer ALU.
-- The final QSA accumulator needs six cycles before its clamp (was five).
local f=assert(io.open(assert(arg[1])));local s=f:read('a');f:close()
local compares,scores,line_no=0,0,0
local rows={'line,old_control,new_control,instruction'}
s=s:gsub('[^\n]+',function(line)
 line_no=line_no+1
 local ctl,space,ins=line:match('^(%S+)(%s+)(.+);')
 if not ctl then return line end
 local wait,rd,wr,y,stall=ctl:match('^([^:]+):([^:]+):([^:]+):([^:]+):([%x])$')
 if not stall then return line end
 local op=ins:match('^@!?P[0-6]%s+(%S+)')or ins:match('^(%S+)')
 local changed=false
 if op:match('^[IF]CMP%.') and (y~='Y' or tonumber(stall,16)<13)then
  y='Y';stall='d';compares=compares+1;changed=true
 elseif ins:match('^FFMA idxscore_acc3, idxscore_qv15, idxscore_kf3, idxscore_acc3$') or
        ins:match('^FFMA scoreselect_acc3, scoreselect_qv15, scoreselect_kf3, scoreselect_acc3$')then
  assert(tonumber(stall,16)==2,'unexpected final accumulator schedule')
  stall='3';scores=scores+1;changed=true
 end
 if not changed then return line end
 local nextctl=table.concat({wait,rd,wr,y,stall},':')
 rows[#rows+1]=string.format('%d,%s,%s,"%s"',line_no,ctl,nextctl,ins)
 return nextctl..space..ins..';'
end)
-- Each scoring body contains32 unrolled copies of the accumulator sequence.
assert(compares>=400 and (scores==32 or scores==64),string.format('unexpected fixed scheduling geometry compares=%d scores=%d',compares,scores))
f=assert(io.open(assert(arg[2]),'w'));f:write(s);f:close()
f=assert(io.open(assert(arg[3]),'w'));f:write(table.concat(rows,'\n'),'\n');f:close()
io.stderr:write(string.format('fixed_latency register_compares=%d final_accumulators=%d\n',compares,scores))
