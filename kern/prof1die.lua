#!/usr/bin/env lua5.4
-- prof1die.lua CUBIN [ENV ...]: block 0 (both halves) of a 256-position chunk on one die (GPU 6) with per-phase times (chunk_test
-- CHUNK_PROF, the f32 exchanges): a holder of the whole block (split 0/1), the test, the holder stopped. Run inside gpuq, from kern/.
local cubin = assert(arg[1], "usage: prof1die.lua CUBIN [K=V ...]")
local extra = table.concat(arg, " ", 2)
local pre = "unset SPLIT_SHARES ATTN_MAP ATTN_DIE; export CUDA_DEVICE_ORDER=PCI_BUS_ID HOLD_VISIBLE=1; "
local gguf = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
os.remove("/tmp/ck-s0.manifest.ready")
local h = io.popen(pre .. "CUDA_VISIBLE_DEVICES=6 ./hold " .. gguf .. " 6 0 1 /tmp/ck-s0.manifest split=0/1 > /tmp/ck-s0.log 2>&1 & echo $!")
local pid = h:read("l"); h:close()
while not io.open("/tmp/ck-s0.manifest.ready") do
  if not os.execute("kill -0 " .. pid .. " 2>/dev/null") then io.write("holder died\n"); os.exit(1) end
  os.execute("sleep 2")
end
local _, _, code = os.execute(pre .. "CUDA_VISIBLE_DEVICES=6 DIE_CUBIN=" .. cubin .. " CHUNK_STAGE=0 CHUNK_PROF=1 CHUNK_WAIT=90 CHUNK_XH=0 " .. extra ..
  " ./chunk_test /tmp/ck-cache-b0 0 256 256 /tmp/ck-s0.manifest")
os.execute("kill " .. pid)
os.exit(code or 1)
