-- Run through Archy's gpuq. Replace only this task's six identified idle
-- holders, then execute the existing full-model benchmark with DFlash2 B4.
local work=arg[1] or '/home/nate/work/flash-decode120-v128'
local cache=arg[2] or work..'/paired-cache'
local manifests='/tmp/flash-decode120-v128'
local original='/home/nate/work/flash-balanced-native-20261022-v1/original-cache'
local old_loader='/home/nate/work/flash-per-phase-balance-v44/native_weights'
local old_manifests='/tmp/flash-unrestricted-v120'
local pids={2704895,2706503,2707816,2709053,2710637,2711908}
local old_gpus={0,1,6,3,4,5}
local function quote(s)return "'"..s:gsub("'","'\\''").."'"end
local function read(path)local f=io.open(path,'rb');if not f then return nil end;local s=f:read('a');f:close();return s end
local function run(args)
 local words={};for _,a in ipairs(args)do words[#words+1]=quote(tostring(a))end
 local command=table.concat(words,' ');io.stderr:write(command..'\n')
 local ok,_,code=os.execute(command);if not ok then os.exit(code or 1)end
end
assert(os.getenv('GPUQ_JOB_ID'),'execute this deployment inside gpuq')
local ready=0
for d=0,5 do
 local folder=cache..'/die'..d
 local m=assert(io.open(folder..'/metadata.bin','rb'),'paired cache is not complete')
 assert(m:read(8)=='NWCAT'..string.char(0,1,0),'cache catalog ABI')
 local header=assert(m:read('l'));m:close()
 local bytes,logical=header:match('native_weights 1 bytes (%d+) physical %d+ logical (%d+)')
 assert(tonumber(logical)==d,'cache rank')
 local arena=assert(io.open(folder..'/arena.bin','rb'));assert(arena:seek('end')==tonumber(bytes),'unfinished cache arena');arena:close()
 print(string.format('deployment_plan logical=%d physical=%d packed_bytes=%s',d,d,bytes))
 if read(manifests..'/h'..d..'.ipc.bin.ready')then ready=ready+1 end
end
assert(ready==0 or ready==6,'partial holder transition needs inspection before proceeding')
if ready==0 then
 for d=0,5 do
  local expected=table.concat({old_loader,'launch',original..'/die'..d,tostring(old_gpus[d+1]),old_manifests..'/h'..d..'.ipc.bin'},'\0')..'\0'
  assert(read('/proc/'..pids[d+1]..'/cmdline')==expected,'task holder identity changed')
  assert(not read(manifests..'/h'..d..'.ipc.bin'),'new manifest already exists')
 end
 run({'mkdir','-p',manifests})
 for _,pid in ipairs(pids)do run({'kill','-TERM',pid})end
 for _,pid in ipairs(pids)do
  local stopped=false
  for attempt=1,200 do
   local stat=read('/proc/'..pid..'/stat')
   if not stat or stat:match('^%d+ %b() Z ')then stopped=true;break end
   os.execute('sleep 0.05')
  end
  assert(stopped,'old task holder has not exited')
 end
 for d=0,5 do run({work..'/native_weights','launch',cache..'/die'..d,d,manifests..'/h'..d..'.ipc.bin'})end
end
local env={
 'BENCH_ROW_BALANCE=1','BENCH_DFLASH2=1','BENCH_NO_MTP=1','BENCH_KERNEL_ONLY=1',
 'BENCH_CANDIDATE_ONLY='..(os.getenv('DECODE120_COMPARE')=='1' and '0' or '1'),
 'BENCH_GENERATE=128','BENCH_RUNS=1','BENCH_CONTEXT=512','BENCH_PLE_READERS=16',
 'BENCH_HOLDER_DIR='..manifests,'BENCH_VISIBLE_DEVICES=0,1,2,3,4,5','BENCH_ANCHOR_DIE=2',
 'BENCH_BALANCED_BATCH_CUBIN='..work..'/die_dflash2_b4.cubin',
 'FLASH_DFLASH_RANK=2','FLASH_DFLASH_DEVICE_CUBIN='..work..'/dflash/device.cubin',
 'FLASH_DFLASH_GGUF=/home/nate/models-cache/flash-sass/dflash2-q4km/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf',
 'FLASH_BALANCED_HCUP5=0','FLASH_BALANCED_HCUP5_LOCALQ=0','FLASH_BALANCED_GR_PACKET=0',
 'FLASH_PLE_DEVICE_CACHE=1','FLASH_WATCHDOG_S=30'
}
local command={'env'};for _,v in ipairs(env)do command[#command+1]=v end
for _,v in ipairs({work..'/resident_decode_bench',work..'/flashcore',work..'/libengine_dflash2.so',
                  work..'/die_dflash2_scalar.cubin',work..'/die_dflash2_b4.cubin'})do command[#command+1]=v end
run(command)
