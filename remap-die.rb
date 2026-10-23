#!/usr/bin/env ruby
# remap-die.rb: run logical die 2 of the six-die split on another physical M60 die (GPU 6 by default) or back on
# GPU 2, restarting the core around the move. Run inside gpuq.
#   ruby remap-die.rb [PHYS]     PHYS = 6: die 2's holder on GPU 6, the core's anchor context on GPU 7
#                                PHYS = 2: back to the original layout (holder on GPU 2, anchor on GPU 6)
# Only the die-2 holder is restarted (it reads its own slice of the shard); the other five keep their arenas.
# The core restarts on the original engine and image with the serving settings, and the resulting config must equal
# the serving one. On any failure after the old holder is stopped, the original layout is restored.
require 'socket'
require 'timeout'
require 'json'
require 'open3'
require 'fileutils'

OUTSIDE = ENV["REMAP_OUTSIDE"] == "1"   # recovery only: run without gpuq (nothing recorded)
raise "run inside gpuq" if ENV.fetch("GPUQ_JOB_ID", "").empty? && !OUTSIDE
phys = Integer(ARGV.fetch(0, '6'))
raise 'PHYS must be 2 or 6' unless [2, 6].include?(phys)
GGUF = '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
LOCK = { 2 => '/home/nate/codex/asm-port/GPU-c767d250-4454-d181-d648-8706b5f2fb64.lock',
         6 => '/home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock' }
MANIFEST = '/tmp/hold-h2.manifest'
HOLD_DIR = '/home/nate/claude/flash-sass-next/kern'
ROOT = '/home/nate/codex/flash-goal-140-1200/remap'
FileUtils.mkdir_p(ROOT)
SOCKET = '/tmp/flash.sock'
HOLD_ENV = { 'HOLD_VISIBLE' => '1', 'CUDA_DEVICE_ORDER' => 'PCI_BUS_ID', 'ATTN_MAP' => '1,1,1,1,4,4,4,4,5,5,5,5',
             'ATTN_DIE' => '1', 'SPLIT_SHARES' => '112,97,81,116,97,97', 'FLASH_PLE_DISK' => '1',
             'PATH' => '/usr/local/sbin:/usr/local/bin:/usr/bin', 'HOME' => '/home/nate' }
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
  return if OUTSIDE
  _, err, status = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(value))
  raise err unless status.success?
end

def log(msg)
  line = "#{Time.now.strftime('%-I:%M:%S %p')} #{msg}"
  puts line; STDOUT.flush
  File.open(File.join(ROOT, 'remap.log'), 'a') { |f| f.puts(line) }
end

def command(text, wait = 180)
  Timeout.timeout(wait) { UNIXSocket.open(SOCKET) { |s| s.puts(text); s.gets || raise(IOError, 'core EOF') } }
end

def core_pid
  Dir.glob('/proc/[0-9]*').each do |dir|
    cmd = File.binread("#{dir}/cmdline") rescue next
    next unless File.basename(cmd.split("\0").first.to_s) == 'flashcore'
    env = File.binread("#{dir}/environ") rescue next
    return dir[6..].to_i if env.include?("FLASH_SOCK=#{SOCKET}")
  end
  nil
end

def wait_gone(pid, seconds)
  Timeout.timeout(seconds) do
    loop do
      break unless File.exist?("/proc/#{pid}/stat")
      break if File.read("/proc/#{pid}/stat").split(') ', 2)[1].to_s.start_with?('Z ')
      sleep 0.2
    end
  end
end

def stop_core
  pid = core_pid
  return log('core: not running') unless pid
  log("core: stopping #{pid}")
  Process.kill('TERM', pid)
  wait_gone(pid, 30)
end

# the die-2 holder: the ./hold process whose manifest argument is /tmp/hold-h2.manifest
def holder2_pids
  Dir.glob('/proc/[0-9]*').filter_map do |dir|
    args = (File.binread("#{dir}/cmdline").split("\0") rescue next)
    dir[6..].to_i if File.basename(args.first.to_s) == 'hold' && args.include?(MANIFEST)
  end
end

def stop_holder2
  holder2_pids.each do |pid|
    log("holder 2: stopping #{pid}")
    Process.kill('TERM', pid)
    wait_gone(pid, 60)
  end
  sleep 2
end

def start_holder2(gpu)
  File.unlink("#{MANIFEST}.ready") if File.exist?("#{MANIFEST}.ready")
  env = HOLD_ENV.merge('CUDA_VISIBLE_DEVICES' => gpu.to_s)
  args = ['flock', '-s', '-n', LOCK.fetch(gpu), './hold', GGUF, gpu.to_s, '0', '48', MANIFEST, 'output', 'split=2/6']
  log("holder 2: starting on GPU #{gpu}: #{args.join(' ')}")
  pid = Process.spawn(env, *args, chdir: HOLD_DIR, in: File::NULL, unsetenv_others: true,
                      out: [File.join(ROOT, "hold2-gpu#{gpu}.out"), 'a'], err: [File.join(ROOT, "hold2-gpu#{gpu}.err"), 'a'], pgroup: true, close_others: true)
  Process.detach(pid)
  Timeout.timeout(1800) do
    loop do
      break if File.exist?("#{MANIFEST}.ready")
      raise "holder 2 exited before it was ready (see #{ROOT}/hold2-gpu#{gpu}.err)" unless File.exist?("/proc/#{pid}/stat")
      sleep 2
    end
  end
  log("holder 2: ready on GPU #{gpu}")
end

def start_core(anchor)
  env = { 'CUDA_DEVICE_ORDER' => 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES' => '0,1,2,3,4,5,6,7',
          'FLASH_ANCHOR_DIE' => anchor.to_s, 'FLASH_GRAPH_CUBIN' => '', 'FLASH_ROPE_OWNER' => '5', 'FLASH_SNAPSHOT_HOST' => '0',
          'FLASH_SOCK' => SOCKET, 'FLASH_KV_HOST' => '0', 'FLASH_MAXT' => '262144', 'FLASH_ENGINE_MAXT' => '230000',
          'FLASH_MAXIMG' => '4096', 'FLASH_PLE_DISK' => '1', 'FLASH_PLE_OFF' => '192', 'FLASH_PLE_READERS' => '1',
          'FLASH_TWO_POSITION' => '1', 'FLASH_PREFILL_CHUNK' => '64', 'FLASH_KV_TRACE' => '0', 'FLASH_TRACE' => '0',
          'FLASH_ATTN_Q_SPLIT' => '1', 'FLASH_PLE_FILE' => '/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf',
          'FLASH_CUBIN' => CUBIN, 'SPLIT_SHARES' => '112,97,81,116,97,97', 'ATTN_MAP' => '1,1,1,1,4,4,4,4,5,5,5,5', 'ATTN_DIE' => '1',
          'PATH' => '/usr/local/sbin:/usr/local/bin:/usr/bin', 'HOME' => '/home/nate' }
  args = ['/home/nate/claude/flash-sass-live/kern/flashcore',
          '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf',
          ENGINE, *(0..5).map { |d| "/tmp/hold-h#{d}.manifest" }]
  log("core: starting (anchor GPU #{anchor})")
  File.write(File.join(ROOT, 'core-launch.json'), JSON.pretty_generate(environment: env, argv: args) + "\n")
  pid = Process.spawn(env, *args, chdir: '/home/nate/claude/flash-sass-live/kern', in: File::NULL, unsetenv_others: true,
                      out: [File.join(ROOT, 'core.stdout'), 'a'], err: [File.join(ROOT, 'core.stderr'), 'a'], close_others: true, pgroup: true)
  Process.detach(pid)
  Timeout.timeout(120) do
    loop do
      raise 'core exited during startup' unless File.exist?("/proc/#{pid}/stat")
      begin
        reply = command('config', 8)
        break if reply.start_with?('config ')
      rescue Errno::ENOENT, Errno::ECONNREFUSED, IOError
      end
      sleep 0.5
    end
  end
  reply = command("reload #{ENGINE} FLASH_CUBIN=#{CUBIN} #{SETTINGS}")
  raise reply unless reply.start_with?('ok')
  config = command('config').chomp
  put("remap-config-gpu#{anchor}", { config: config, match: same_serving?(config, EXPECTED) })
  raise "config differs from the serving config:\n#{config}" unless same_serving?(config, EXPECTED)
  log('core: serving config verified')
end

log("remap: logical die 2 -> GPU #{phys}")
put('remap-before', { holders: holder2_pids, core: core_pid, gpus: `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader` })
stop_core
stop_holder2
begin
  start_holder2(phys)
  start_core(phys == 6 ? 7 : 6)
rescue StandardError => e
  log("remap failed: #{e.message}; restoring die 2 on GPU 2")
  put('remap-failed', { error: e.message })
  stop_core rescue nil
  stop_holder2 rescue nil
  start_holder2(2)
  start_core(6)
  raise
end
put('remap-after', { holders: holder2_pids, core: core_pid, gpus: `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader` })
log("remap: done, logical die 2 on GPU #{phys}")
