#!/usr/bin/env ruby
# phase-bisect.rb ENGINE CUBIN BASE N...: on the serving core (the v1 holders), for each stop phase N reload the engine with
#   FLASH_STOP_PHASE=N FLASH_DUMP_S=BASE.pN FLASH_DUMP_SEGS=1 FLASH_LAYERS=1  (+ PHASE_BISECT_ENV)
# and run one tiny decode request (measure.rb), then restore the serving engine and image. Run inside gpuq.
# The unit-layout counterpart is holders-units.rb with UNITS_CONFIGS (see phase-bisect-units.sh).
require 'socket'
require 'timeout'
require 'json'
require 'open3'

raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
engine, cubin, base = ARGV[0], ARGV[1], ARGV[2]
stops = ARGV.drop(3)   # "p16": FLASH_STOP_PHASE=16 (that many phases of every position); "t3": FLASH_STOP_TOKEN=3 (the first three positions, all phases); "t2p8": position 0 complete, then 8 phases of position 1; a bare number is "p<n>"
raise 'usage: phase-bisect.rb ENGINE CUBIN BASE N...' unless engine && cubin && base && !stops.empty?
SERVE_ENGINE = '/home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so'
SERVE_CUBIN = '/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
extra = ENV.fetch('PHASE_BISECT_ENV', '').split
BASE_ENV = 'FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 FLASH_HCR_BATCH=0 FLASH_DNSTEP_BATCH=0 ' \
           'FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0 FLASH_CHAIN=0 FLASH_TWO_POSITION=0 ' \
           'FLASH_PREFILL_CHUNK=0 FLASH_ROPE_RING=0 FLASH_QSA_SHORT=0'

def command(text, wait = 180)
  Timeout.timeout(wait) { UNIXSocket.open('/tmp/flash.sock') { |s| s.puts(text); s.gets || raise(IOError, 'core EOF') } }
end

before = command('config').chomp
ok = true
begin
  stops.each do |n|
    n = "p#{n}" if n =~ /\A\d+\z/
    m = n.match(/\A(?:t(\d+))?(?:p(\d+))?\z/) or raise "bad stop #{n}"
    stop = "FLASH_STOP_TOKEN=#{m[1]} FLASH_STOP_PHASE=#{m[2]}"   # an empty value unsets the variable
    reply = command("reload #{engine} FLASH_CUBIN=#{cubin} #{BASE_ENV} FLASH_LAYERS=1 FLASH_DUMP_SEGS=1 #{stop} FLASH_DUMP_S=#{base}.#{n} #{extra.join(' ')}")
    raise reply if reply.start_with?('error')
    env = { 'BASELINE_JOB' => '63668679-0f08-49f2-b283-22969e66b4bf', 'MEASURE_KIND' => 'decode', 'MEASURE_PAIRS' => '1',
            'MEASURE_DEFLECT' => '1', 'ALLOW_OUTPUT_DIFFERENCE' => '1' }
    system(env, RbConfig.ruby, File.join(__dir__, 'measure.rb'), '753710', out: File::NULL, err: File::NULL)
    puts "stop #{n}: #{Dir.glob("#{base}.#{n}.s0.meta").size == 1 ? 'dumped' : 'NO DUMP'}"
    $stdout.flush
  end
ensure
  restore = "reload #{SERVE_ENGINE} FLASH_CUBIN=#{SERVE_CUBIN} FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 " \
            'FLASH_HCR_BATCH=0 FLASH_DNSTEP_BATCH=0 FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0 FLASH_CHAIN=0 ' \
            'FLASH_LAYERS= FLASH_DUMP_S= FLASH_STOP_PHASE= FLASH_STOP_TOKEN= FLASH_DUMP_SEGS=FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 ' \
            'FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
  reply = (command(restore) rescue "error #{$!.message}")
  after = (command('config').chomp rescue '')
  puts "restore: #{reply.chomp[0, 80]}; config #{after.sub(/ attn_q_mask=\h+/, '') == before.sub(/ attn_q_mask=\h+/, '') ? 'matches' : 'DIFFERS'}"
end
