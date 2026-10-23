#!/usr/bin/env ruby
# holders-units.rb: swap the six v1 holders for the one-pass unit-layout holder (kern/hold2.c), run the core on an engine
# and image that understand the unit layout, measure, then put the v1 holders and the serving engine back. Run inside gpuq.
#   ruby holders-units.rb ENGINE CUBIN [MEASURE...]
# MEASURE: decode:N and/or prefill:N (fresh chat requests through measure.rb, outputs compared but allowed to differ).
# Logical die 2 stays on the physical die the current die-2 manifest names (GPU 6 after remap-die.rb 6).
# The v1 holders and the serving engine are restored in every case (ensure), the config checked against the serving one.
require 'socket'
require 'timeout'
require 'json'
require 'open3'
require 'fileutils'

raise 'run inside gpuq' if ENV.fetch('GPUQ_JOB_ID', '').empty?
engine, cubin = ARGV[0], ARGV[1]
raise "usage: holders-units.rb ENGINE CUBIN [MEASURE...]" unless ENV["UNITS_RESTORE_ONLY"] || (engine && cubin)
measures = ARGV.drop(2).map { |m| k, n = m.split(':'); [k, Integer(n)] }
GGUF = '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
SHARD2 = '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf'
LOCKS = { 0 => 'GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f', 1 => 'GPU-ccb8b3a3-f45d-8962-215b-c2b140e4bb28',
          2 => 'GPU-c767d250-4454-d181-d648-8706b5f2fb64', 3 => 'GPU-5f06269e-5fe2-ff52-1a0d-b0d0c0c763a6',
          4 => 'GPU-906e6ede-6cde-d0dd-5110-d1bb8bf2458d', 5 => 'GPU-1d9dc086-8739-66c2-fa98-f5bf3ef2b2d4',
          6 => 'GPU-431371bd-b482-96b2-4b70-04f83c6d075b' }.transform_values { |u| "/home/nate/codex/asm-port/#{u}.lock" }
MANIFESTS = (0..5).map { |d| "/tmp/hold-h#{d}.manifest" }
V1_DIR = '/home/nate/claude/flash-sass-next/kern'
UNITS_DIR = '/home/nate/codex/flash-goal-140-1200/chain/kern'
ROOT = '/home/nate/codex/flash-goal-140-1200/holders'
FileUtils.mkdir_p(ROOT)
SOCKET = '/tmp/flash.sock'
SPLIT_ENV = { 'CUDA_DEVICE_ORDER' => 'PCI_BUS_ID', 'ATTN_MAP' => '1,1,1,1,4,4,4,4,5,5,5,5', 'ATTN_DIE' => '1',
              'SPLIT_SHARES' => '112,97,81,116,97,97', 'FLASH_PLE_DISK' => '1',
              'PATH' => '/usr/local/sbin:/usr/local/bin:/usr/bin', 'HOME' => '/home/nate' }
SERVE_ENGINE = '/home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so'
SERVE_CUBIN = '/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
SETTINGS = 'FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 FLASH_HCR_BATCH=0 ' \
           'FLASH_DNSTEP_BATCH=0 FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0 ' \
           'FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 ' \
           'FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
EXPECTED = 'config rev=attention-q-release kv_host=0 kv_trace=0 ple_disk=1 ple_readers=1 ' \
           'ple_file=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf ' \
           'ple_cache=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf maxt=230000 layers=48/48 ' \
           'nblk=512 dies=6 attn_dies=1,4,5 split=112,97,81,116,97,97 attn_map=1,1,1,1,4,4,4,4,5,5,5,5 ' \
           'img_rows=4096 rope_owner=5 trace_tokens=0 two_position=1 prefill_chunk=64 attn_q_split=1 ' \
           "attn_q_mask=37 cubin=#{SERVE_CUBIN} cubin_fnv=9abf8aafb9134d59"

# attn_q_mask (the dies holding attention-Q rows) follows the free memory at engine start; it is not part of the identity
def same_serving?(a, b) = a.sub(/ attn_q_mask=\h+/, "") == b.sub(/ attn_q_mask=\h+/, "")

def put(label, value)
  _, err, status = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(value))
  raise err unless status.success?
end

def log(msg)
  line = "#{Time.now.strftime('%-I:%M:%S %p')} #{msg}"
  puts line; STDOUT.flush
  File.open(File.join(ROOT, 'holders.log'), 'a') { |f| f.puts(line) }
end

def command(text, wait = 180)
  Timeout.timeout(wait) { UNIXSocket.open(SOCKET) { |s| s.puts(text); s.gets || raise(IOError, 'core EOF') } }
end

def procs
  Dir.glob('/proc/[0-9]*').filter_map do |dir|
    args = (File.binread("#{dir}/cmdline").split("\0") rescue next)
    env = (File.binread("#{dir}/environ") rescue '')
    [dir[6..].to_i, args, env]
  end
end

def core_pid = procs.find { |_, a, e| File.basename(a.first.to_s) == 'flashcore' && e.include?("FLASH_SOCK=#{SOCKET}") }&.first
def v1_holders = procs.select { |_, a, _| File.basename(a.first.to_s) == 'hold' && (a & MANIFESTS).any? }
def unit_holders = procs.select { |_, a, _| File.basename(a.first.to_s) == 'hold2' }

def wait_gone(pid, seconds)
  Timeout.timeout(seconds) do
    loop do
      break unless File.exist?("/proc/#{pid}/stat")
      break if File.read("/proc/#{pid}/stat").split(') ', 2)[1].to_s.start_with?('Z ')
      sleep 0.2
    end
  end
end

def stop(pid, what, seconds = 60)
  log("#{what}: stopping #{pid}")
  Process.kill('TERM', pid)
  wait_gone(pid, seconds)
end

def manifest_die(path) = File.read(path)[/\Aarena \d+ die (\d+)/, 1].to_i

def wait_ready(pid, seconds = 2400)
  Timeout.timeout(seconds) do
    loop do
      break if MANIFESTS.all? { |m| File.exist?("#{m}.ready") }
      raise 'holder exited before it was ready' if pid && !File.exist?("/proc/#{pid}/stat")
      sleep 3
    end
  end
end

def start_v1(phys)
  pids = []
  running = v1_holders.map { |_, a, _| a.find { |s| MANIFESTS.include?(s) } }
  wait_free((0..5).map { |d| phys[d] }.reject { |g| running.include?(MANIFESTS[phys.index(g)]) })
  (0..5).each do |d|
    if running.include?(MANIFESTS[d]) && File.exist?("#{MANIFESTS[d]}.ready")
      log("v1 holder #{d}: already running and ready")
      next
    end
    File.unlink("#{MANIFESTS[d]}.ready") if File.exist?("#{MANIFESTS[d]}.ready")
    env = SPLIT_ENV.merge('HOLD_VISIBLE' => '1', 'CUDA_VISIBLE_DEVICES' => phys[d].to_s)
    args = ['flock', '-s', '-n', LOCKS.fetch(phys[d]), './hold', GGUF, phys[d].to_s, '0', '48', MANIFESTS[d], *(d.zero? ? ['token_embd'] : []), 'output', "split=#{d}/6"]
    log("v1 holder #{d}: GPU #{phys[d]}")
    pid = Process.spawn(env, *args, chdir: V1_DIR, in: File::NULL, unsetenv_others: true,
                        out: [File.join(ROOT, "v1-#{d}.out"), 'a'], err: [File.join(ROOT, "v1-#{d}.err"), 'a'], pgroup: true, close_others: true)
    Process.detach(pid); pids << pid
  end
  wait_ready(nil)
  log('v1 holders: ready')
  pids
end

# the driver releases a stopped holder's arena some time after the process is gone: wait until the dies read as empty
def wait_free(gpus, limit_mib = 600, seconds = 120)
  Timeout.timeout(seconds) do
    loop do
      used = `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`.lines.to_h { |l| l.split(',').map(&:to_i) }
      # a fence's blocks (kern/fence.c, the bad VRAM of a die) stay allocated by design: they do not count
      bus = `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`.lines.to_h { |l| i, b = l.split(',').map(&:strip); [b, i.to_i] }
      `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits`.lines.each do |l|
        pid, b, mib = l.split(',').map(&:strip)
        used[bus[b]] -= mib.to_i if bus[b] && File.basename((File.binread("/proc/#{pid}/cmdline").split("\0").first rescue '')) == 'fence'
      end
      break if gpus.all? { |g| used.fetch(g) < limit_mib }
      sleep 1
    end
  end
  log("dies #{gpus.join(',')}: memory released")
end

def start_units(phys, attempts = 2)
  wait_free(phys)
  MANIFESTS.each { |m| File.unlink("#{m}.ready") if File.exist?("#{m}.ready") }
  env = SPLIT_ENV.merge('CUDA_VISIBLE_DEVICES' => '0,1,2,3,4,5,6,7', 'HOLD2_DEVICES' => phys.join(','))
  args = phys.flat_map { |p| ['flock', '-s', '-n', LOCKS.fetch(p)] } + ['./hold2', GGUF, *MANIFESTS, 'token_embd', 'output']
  log("unit holder: GPUs #{phys.join(',')}")
  pid = Process.spawn(env, *args, chdir: UNITS_DIR, in: File::NULL, unsetenv_others: true,
                      out: [File.join(ROOT, 'units.out'), 'a'], err: [File.join(ROOT, 'units.err'), 'a'], pgroup: true, close_others: true)
  Process.detach(pid)
  begin
    wait_ready(pid)
  rescue RuntimeError => e
    raise if (attempts -= 1) <= 0
    log("unit holder: #{e.message}; retrying in 15 s")
    sleep 15
    return start_units(phys, attempts)
  end
  log('unit holder: ready')
  pid
end

def start_core(eng, cub, extra, anchor)
  env = { 'CUDA_DEVICE_ORDER' => 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES' => '0,1,2,3,4,5,6,7',
          'FLASH_ANCHOR_DIE' => anchor.to_s, 'FLASH_GRAPH_CUBIN' => '', 'FLASH_ROPE_OWNER' => '5', 'FLASH_SNAPSHOT_HOST' => '0',
          'FLASH_SOCK' => SOCKET, 'FLASH_KV_HOST' => '0', 'FLASH_MAXT' => '262144', 'FLASH_ENGINE_MAXT' => '230000',
          'FLASH_MAXIMG' => '4096', 'FLASH_PLE_DISK' => '1', 'FLASH_PLE_OFF' => '192', 'FLASH_PLE_READERS' => '1',
          'FLASH_TWO_POSITION' => '1', 'FLASH_PREFILL_CHUNK' => '64', 'FLASH_KV_TRACE' => '0', 'FLASH_TRACE' => '0',
          'FLASH_ATTN_Q_SPLIT' => '1', 'FLASH_PLE_FILE' => '/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf',
          'FLASH_CUBIN' => cub, 'SPLIT_SHARES' => '112,97,81,116,97,97', 'ATTN_MAP' => '1,1,1,1,4,4,4,4,5,5,5,5', 'ATTN_DIE' => '1',
          'PATH' => '/usr/local/sbin:/usr/local/bin:/usr/bin', 'HOME' => '/home/nate' }
  extra.split.each { |kv| k, v = kv.split('=', 2); env[k] = v }
  args = ['/home/nate/claude/flash-sass-live/kern/flashcore', SHARD2, eng, *MANIFESTS]
  log("core: starting #{File.basename(eng)} #{File.basename(cub)} #{extra}")
  pid = Process.spawn(env, *args, chdir: '/home/nate/claude/flash-sass-live/kern', in: File::NULL, unsetenv_others: true,
                      out: [File.join(ROOT, 'core.stdout'), 'a'], err: [File.join(ROOT, 'core.stderr'), 'a'], close_others: true, pgroup: true)
  Process.detach(pid)
  Timeout.timeout(180) do
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
  config = command('config').chomp
  log("core: #{config[0, 160]}")
  config
end

# the v1 holders and the serving engine, the config checked against the serving one
def restore(phys, anchor, results)
  log("restore: v1 holders and the serving engine")
  pid = core_pid
  stop(pid, "core", 30) if pid
  unit_holders.each { |p, _, _| stop(p, "unit holder", 120) }
  sleep 3
  start_v1(phys)
  start_core(SERVE_ENGINE, SERVE_CUBIN, "", anchor)
  reply = command("reload #{SERVE_ENGINE} FLASH_CUBIN=#{SERVE_CUBIN} #{SETTINGS}")
  raise reply unless reply.start_with?("ok")
  final = command("config").chomp
  put("units-restored", { config: final, match: same_serving?(final, EXPECTED), results: results })
  log(same_serving?(final, EXPECTED) ? "restore: serving config verified" : "restore: CONFIG DIFFERS #{final}")
end

if ENV["UNITS_RESTORE_ONLY"]   # PHYS (logical die -> GPU) as given, e.g. 0,1,6,3,4,5
  phys = ENV["UNITS_RESTORE_ONLY"].split(",").map { |v| Integer(v) }
  restore(phys, ([6, 7] - phys).first, {})
  exit
end

phys = MANIFESTS.map { |m| manifest_die(m) }
anchor = ([6, 7] - phys).first
log("units run: logical dies on GPUs #{phys.join(',')}, anchor #{anchor}")
put('units-before', { phys: phys, gpus: `nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader` })
pid = core_pid
stop(pid, 'core', 30) if pid
v1_holders.each { |p, _, _| stop(p, 'v1 holder') }
sleep 3
results = {}
begin
  start_units(phys)
  config = start_core(engine, cubin, ENV.fetch('UNITS_CORE_ENV', ''), anchor)
  put('units-config', config)
  # UNITS_CONFIGS="K=V ...;K=V ...": reload the engine with each setting list in turn and measure each (the core
  # rebuilds its contexts after a device fault, so a failing setting does not end the run)
  ENV.fetch('UNITS_CONFIGS', '').split(';').map(&:strip).reject(&:empty?).each_with_index do |cfg, i|
    reply = (command("reload #{engine} FLASH_CUBIN=#{cubin} #{cfg}") rescue "error #{$!.message}")
    log("config #{i}: #{cfg}: reload #{reply.chomp[0, 80]}")
    measures.each do |kind, n|
      env = { 'BASELINE_JOB' => '63668679-0f08-49f2-b283-22969e66b4bf', 'MEASURE_KIND' => kind, 'MEASURE_PAIRS' => n.to_s,
              'MEASURE_DEFLECT' => '1', 'ALLOW_OUTPUT_DIFFERENCE' => '1' }
      ok = system(env, RbConfig.ruby, '/home/nate/codex/flash-goal-140-1200/measure.rb', '753710')
      results["#{i}:#{kind}"] = ok
      log("config #{i}: #{kind} #{ok ? 'done' : 'FAILED'}")
    end
  end
  measures = [] unless ENV.fetch('UNITS_CONFIGS', '').strip.empty?
  measures.each do |kind, n|
    env = { 'BASELINE_JOB' => '63668679-0f08-49f2-b283-22969e66b4bf', 'MEASURE_KIND' => kind, 'MEASURE_PAIRS' => n.to_s,
            'MEASURE_DEFLECT' => '1', 'ALLOW_OUTPUT_DIFFERENCE' => '1' }
    log("measure: #{kind} x #{n}")
    ok = system(env, RbConfig.ruby, '/home/nate/codex/flash-goal-140-1200/measure.rb', '753710')
    results[kind] = ok
    log("measure: #{kind} #{ok ? 'done' : 'FAILED'}")
  end
ensure
  restore(phys, anchor, results)
end
