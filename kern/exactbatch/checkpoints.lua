-- Add optional per-position state captures to the expanded B8 DeltaNet body.
-- Each scalar descriptor supplies an independent checkpoint pointer at words
-- 24/25. Zero skips the copy. Captures are consumed only after kernel completion.
local function read(p)local f=assert(io.open(p,'rb'));local s=f:read('a');f:close();return s end
local s=read(assert(arg[1]));local output=assert(arg[2])
local copy=read(assert(arg[3]))
local function once(a,b)
 local i,j=assert(s:find(a,1,true),'missing anchor '..a);j=i+#a-1
 assert(not s:find(a,j+1,true),'duplicate anchor '..a)
 s=s:sub(1,i-1)..b..s:sub(j+1)
end
once('    194 : dnstep8_param_npos\n','    194 : dnstep8_param_npos\n    196,197 : dnstep8_checkpoint<0-1>\n    198 : dnstep8_checkpoint_offset\n')
local begin=assert(s:find('--:2:-:-:1      STG.E.128 [dnstep8_sp+0x0], dnstep8_s00;\n',1,true))
local finish=assert(s:find('02:-:-:-:f      BRA dnstep8_END;',begin,true))
local stores=s:sub(begin,finish-1):gsub('%[dnstep8_sp','[dnstep8_checkpoint')
local count=0;for _ in stores:gmatch('STG%.E%.128')do count=count+1 end
assert(count==16,'unexpected DeltaNet state geometry')
local capture=[=[--:-:1:-:2      LDS.64 dnstep8_checkpoint0, [dnstep8_descriptor+0x60];
01:-:-:-:6      LOP.OR dnstep8_t, dnstep8_checkpoint0, dnstep8_checkpoint1;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, dnstep8_t, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA dnstep8_CHECKPOINT_DONE;
--:-:-:-:6      SHL dnstep8_checkpoint_offset, dnstep8_head, 0x10;
--:-:-:-:6      SHL dnstep8_t, dnstep8_g, 0xb;
--:-:-:-:6      IADD dnstep8_checkpoint_offset, dnstep8_checkpoint_offset, dnstep8_t;
--:-:-:-:6      ISCADD dnstep8_checkpoint_offset, dnstep8_p, dnstep8_checkpoint_offset, 0x6;
--:-:-:-:6      IADD dnstep8_checkpoint0.CC, dnstep8_checkpoint0, dnstep8_checkpoint_offset;
--:-:-:-:6      IADD.X dnstep8_checkpoint1, dnstep8_checkpoint1, RZ;
]=]..stores..[=[02:-:-:-:6      NOP;
dnstep8_CHECKPOINT_DONE:
]=]
local anchor='02:-:-:-:6      BAR.SYNC 0x0;\n--:-:-:-:6      IADD dnstep8_position, dnstep8_position, 0x1;'
once(anchor,'02:-:-:-:6      BAR.SYNC 0x0;\n'..capture..'--:-:-:-:6      IADD dnstep8_position, dnstep8_position, 0x1;')
local count=assert(tonumber(s:match('ISETP%.GE%.U32%.AND P6, PT, op, 0x(%x+), PT;'),16))
local map=assert(copy:match('<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>'))
local body=copy:gsub('<REGISTER_MAPPING>.-</REGISTER_MAPPING>','')
once('</REGISTER_MAPPING>',map..'\n</REGISTER_MAPPING>')
once(string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count),string.format('ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;',count+1))
once('--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:',
 '--:-:-:-:6      BRA verifycopy_SETUP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      NOP;\n--:-:-:-:6      BRA POSITION_END;\n\nembrow_SETUP:')
once('POSITION_END:',body..'\nPOSITION_END:')
local f=assert(io.open(output,'wb'));assert(f:write(s));assert(f:close())
