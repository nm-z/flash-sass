#!/usr/bin/env ruby
# Finite peer fixture in borrowed serving contexts, recorded through gpuq.
require_relative 'state-compare'
raise 'run inside gpuq' if ENV.fetch('GPUQ_JOB_ID','').empty?
library,cubin=ARGV
raise 'expected probe library and cubin' unless ARGV.size==2
[library,cubin].each { |p| raise 'unsafe protocol path' unless p.match?(/\A[\w.\/-]+\z/) }
widths=ENV.fetch('HCUP_PROBE_WIDTHS','8').split(',').map { |w| Integer(w) }
raise 'invalid widths' unless !widths.empty? && widths.all? { |w| [1,2,8,16,32].include?(w) }
core=StateCompare::Core.new('/tmp/flash.sock')
before=core.single('config')
raise 'original image not selected' unless StateCompare.fields(before)['cubin']==StateCompare::REFERENCE_CUBIN
settings='FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 FLASH_HCR_BATCH=0 FLASH_DNSTEP_BATCH=0 FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_HCUP_OFFLOAD_PERIOD=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0'
reload=->(lib,image) do
  reply=core.single("reload #{lib} FLASH_CUBIN=#{image} #{settings}")
  raise reply unless reply.start_with?('ok:')
  core.close;core.connect
  core.single('config')
end
StateCompare.put('hcup-probe-artifacts',{before:before,library:library,cubin:cubin,
  hashes:[library,cubin,__FILE__].to_h { |p| [p,Digest::SHA256.file(p).hexdigest] },widths:widths})
begin
  selected=reload.call(library,cubin)
  StateCompare.put('hcup-probe-selected',selected)
  widths.each do |width|
    lines=core.exchange("hcup_offload_probe #{width}",timeout:360) { |line| line.start_with?('hcup_offload_probe done ') }
    StateCompare.put("hcup-probe-width-#{width}",{config:selected,lines:lines})
    raise 'peer fixture failed' unless lines.last.include?('status=PASS')
    puts lines.last
  end
rescue StandardError => error
  StateCompare.put('hcup-probe-error',{error:error.message,wire:core.history})
  raise
ensure
  core.close
  begin
    core=StateCompare::Core.new('/tmp/flash.sock')
    after=reload.call(StateCompare::REFERENCE_ENGINE,StateCompare::REFERENCE_CUBIN)
  rescue IOError,Errno::ECONNRESET,Errno::ECONNREFUSED,Errno::ENOENT => error
    StateCompare.put('hcup-probe-recovery-needed',error.message)
    raise 'core recovery failed' unless system(RbConfig.ruby,'/home/nate/codex/flash-astra-perf-20261018/recover-resume.rb')
    core=StateCompare::Core.new('/tmp/flash.sock')
    settings+=' FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
    after=reload.call(StateCompare::REFERENCE_ENGINE,StateCompare::REFERENCE_CUBIN)
  end
  StateCompare.put('hcup-probe-restored',{before:before,after:after,wire:core.history})
  raise 'original configuration not restored' unless before==after
  core.close
end
