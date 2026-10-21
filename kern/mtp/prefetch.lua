-- Separate the scalar dispatcher's descriptor and token-mask load scoreboards.
return function(s)
 local loads=0
 s=s:gsub('[^\n]*LDG%.E vid, %[dsp%+0x7c%];',function(line)
  local code,rest=assert(line:match('^(%S+)(%s+.*)$'))
  local fields={};for x in code:gmatch('[^:]+')do fields[#fields+1]=x end
  assert(#fields==5 and fields[3]=='4' and (fields[2]=='-' or fields[2]=='3'))
  fields[2]='-';fields[3]='5';loads=loads+1
  return table.concat(fields,':')..rest
 end)
 local consumers
 s,consumers=s:gsub('09:%-:%-:%-:6(%s+SHR%.U32 vid, vid, lane;)','19:-:-:-:6%1')
 local address
 s,address=s:gsub('04:%-:%-:%-:6(%s+IADD xa0%.CC, bar_cp0, bar_t;)','14:-:-:-:6%1')
 assert(loads==3 and consumers==2 and address==1,'unexpected scalar dispatcher layout')
 return s
end
