#!/usr/bin/env ruby
# Restart a dead flashcore and reload the serving engine exactly as it was configured.
# Run inside gpuq. The weight holders must still be alive (the core attaches to their arenas).
require 'socket'
require 'timeout'
require 'json'
require 'open3'

raise 'run inside gpuq' if ENV.fetch('GPUQ_JOB_ID', '').empty?
ENGINE = '/home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so'
CUBIN = '/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
SETTINGS = 'FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 FLASH_HCR_BATCH=0 ' \
           'FLASH_DNSTEP_BATCH=0 FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0 ' \
           'FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 ' \
           'FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
EXPECTED = 'config rev=attention-q-release kv_host=0 kv_trace=0 ple_disk=1 ple_readers=1 ' \
           'ple_file=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf ' \
           'ple_cache=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf maxt=230000 layers=48/48 ' \
           'nblk=512 dies=6 attn_dies=1,4,5 split=112,97,81,116,97,97 attn_map=1,1,1,1,4,4,4,4,5,5,5,5 ' \
           'img_rows=4096 rope_owner=5 trace_tokens=0 two_position=1 prefill_chunk=64 attn_q_split=1 ' \
           "attn_q_mask=37 cubin=#{CUBIN} cubin_fnv=9abf8aafb9134d59"

# attn_q_mask (the dies holding attention-Q rows) follows the free memory at engine start; it is not part of the identity
def same_serving?(a, b) = a.sub(/ attn_q_mask=\h+/, "") == b.sub(/ attn_q_mask=\h+/, "")

def put(label, value)
  _, err, status = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(value))
  raise err unless status.success?
end

def command(text)
  Timeout.timeout(180) { UNIXSocket.open('/tmp/flash.sock') { |s| s.puts(text); s.gets || raise(IOError, 'core EOF') } }
end

ok = system(RbConfig.ruby, '/home/nate/codex/flash-astra-perf-20261018/recover-resume.rb')
put('restore-core-started', { recover_resume_ok: ok })
raise 'core recovery failed' unless ok
reply = command("reload #{ENGINE} FLASH_CUBIN=#{CUBIN} #{SETTINGS}")
put('restore-reload', reply)
raise reply unless reply.start_with?('ok')
config = command('config').chomp
put('restore-config', { config: config, expected: EXPECTED, match: same_serving?(config, EXPECTED) })
puts config
raise "restored config differs from the pre-crash serving config:\n#{config}" unless same_serving?(config, EXPECTED)
