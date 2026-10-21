-- Import the expanded, scalar-compatible B8 operators and their framework.
-- This is a one-time source import, not a build dependency on another checkout.
local function read(p) local f=assert(io.open(p,"rb"));local s=f:read("a");f:close();return s end
local function write(p,s) local f=assert(io.open(p,"wb"));assert(f:write(s));assert(f:close()) end
local ref=read(assert(arg[1]));local dir=assert(arg[2])
local tags={"bf16bias8","bf16k25608","bf16k3208","bf16hc8","bf16hcx8",
 "q2k25608","q2k6408","q2acc6408","q2moegn8","q2moedn8","bf16hcup8",
 "q3kk25608","q3kk61448","dnstep8","actq2568","rms25608"}
local map=assert(ref:match("<REGISTER_MAPPING>(.-)</REGISTER_MAPPING>"))
local global={}
for line in map:gmatch("[^\n]+") do
 if line:match(":%s*embrow_") then break end
 global[#global+1]=line.."\n"
end
local _,re=assert(ref:find("</REGISTER_MAPPING>",1,true))
local ds=assert(ref:find("\nOP_DISPATCH:\n",re,true))
local fs=assert(ref:find("\nPOSITION_END:\n",ds,true))
local pro=ref:sub(re+1,ds-1)
assert(not pro:find("CCTL.E.PF2",1,true),"use a framework without optional weight prefetch")
write(dir.."/frame.sass","# Imported expanded B8 framework from die_slab21.sass.\n"
 ..assert(ref:match("(<CONSTANT_MAPPING>.-</CONSTANT_MAPPING>)"))
 .."\n<REGISTER_MAPPING>\n"..table.concat(global).."# EXACT_OPERATOR_MAP\n</REGISTER_MAPPING>\n"
 ..pro.."\n# EXACT_DISPATCH\n# EXACT_OPERATOR_BODY\n"..ref:sub(fs+1))
local table_begin=assert(ref:find("\nDISPATCH_TABLE:\n",ds,true))
local body_begin=assert(ref:find("\nembrow_SETUP:\n",table_begin,true))
local all={};for tag in ref:sub(table_begin,body_begin-1):gmatch("BRA%s+([%w_]+)_SETUP;") do all[#all+1]=tag end
local bounds={}
for i,tag in ipairs(all) do
 local a=assert(ref:find("\n"..tag.."_SETUP:\n",body_begin,true))
 local b=i<#all and assert(ref:find("\n"..all[i+1].."_SETUP:\n",a+1,true)) or fs
 bounds[tag]={a+1,b}
end
local maps,bodies={},{}
for _,tag in ipairs(tags) do
 local found=0
 for line in map:gmatch("[^\n]+") do
  if line:match(":%s*"..tag.."_") then maps[#maps+1]=line.."\n";found=found+1 end
 end
 assert(found>0,"missing register map "..tag)
 local b=assert(bounds[tag],"missing operator "..tag)
 bodies[#bodies+1]=ref:sub(b[1],b[2])
end
write(dir.."/operators.sass","# Imported expanded B8 operators; scalar arithmetic order is retained.\n<REGISTER_MAPPING>\n"
 ..table.concat(maps).."</REGISTER_MAPPING>\n"..table.concat(bodies,"\n"))
