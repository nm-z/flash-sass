-- Extend the production/batched dispatcher with a fused MoE residual write,
-- RMS-normalized GR down read, and publication. Six peer partials are summed
-- in the existing RMS-write prologue; no FP8 storage conversion is required.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local base=read(assert(arg[1]));local s=read(assert(arg[2]));local output=assert(arg[3])
local maps={}
for line in assert(base:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>')):gmatch('[^\n]+')do
 if line:find('bf16hcrwx_',1,true)then maps[#maps+1]=line:gsub('bf16hcrwx','grreadsum')end
end
maps[#maps+1]='    240-243 : grreadsum_tmp<0-3>'
local body=assert(base:match('(bf16hcrwx_SETUP:.-)\nq4kk2560_SETUP:')):gsub('bf16hcrwx','grreadsum')
local replacements=0
body=body:gsub('[^\n]+',function(line)
 local first,offset=line:match('LDG%.E%.CI%.128 grreadsum_act(%d+), %[grreadsum_yb%+0x(%x+)%];')
 if not first then return line end
 first=tonumber(first);offset=tonumber(offset,16);replacements=replacements+1
 local pred=line:find('@P4',1,true) and '@P4 ' or '    '
 local out={line,'01:-:-:-:6      NOP;'}
 for r=0,3 do out[#out+1]=string.format('--:-:-:-:6  %sFADD grreadsum_act%02d, RZ, grreadsum_act%02d;',pred,first+r,first+r)end
 for peer=1,5 do
  out[#out+1]=string.format('--:-:1:-:2  %sLDG.E.CI.128 grreadsum_tmp0, [grreadsum_yb+0x%x];',pred,offset+peer*0x2800)
  for r=0,3 do out[#out+1]=string.format('%s:-:-:-:6  %sFADD grreadsum_act%02d, grreadsum_act%02d, grreadsum_tmp%d;',r==0 and '01' or '--',pred,first+r,first+r,r)end
 end
 return table.concat(out,'\n')
end)
assert(replacements==30,'unexpected RMS write load geometry')
local function once(a,b)
 local i,j=assert(s:find(a,1,true),'missing '..a);j=i+#a-1;assert(not s:find(a,j+1,true),'duplicate '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
local count=assert(tonumber(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;'),16))
assert(count==94,'unexpected opcode ABI')
once('</REGISTER_MAPPING>',table.concat(maps,'\n')..'\n</REGISTER_MAPPING>')
once(string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count),string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count+1))
once('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6      BRA grreadsum_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
once('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(output,'wb'));f:write(s);f:close()
