require 'socket'
require 'timeout'
require 'json'
require 'open3'
require 'digest'

raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
engine='/home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so'
original='/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
candidate=ARGV.fetch(0)
candidate_engine=ARGV.fetch(1,engine)
batch=ENV.fetch('CANDIDATE_BATCH','0')
raise 'invalid batch' unless %w[0 8 16 32].include?(batch)
q2_batch=ENV.fetch('CANDIDATE_Q2','0')
raise 'invalid Q2 batch selection' unless %w[0 1].include?(q2_batch)
hcup_batch=ENV.fetch('CANDIDATE_HCUP','0')
raise 'invalid HCUP batch selection' unless %w[0 1].include?(hcup_batch)
hcr_batch=ENV.fetch('CANDIDATE_HCR','0')
raise 'invalid HCR batch selection' unless %w[0 1].include?(hcr_batch)
q3_batch=ENV.fetch('CANDIDATE_Q3','0')
raise 'invalid Q3 batch selection' unless %w[0 1].include?(q3_batch)
dnstep_batch=ENV.fetch('CANDIDATE_DNSTEP','0')
raise 'invalid DNSTEP batch selection' unless %w[0 1].include?(dnstep_batch)
actq_batch=ENV.fetch('CANDIDATE_ACTQ','0')
raise 'invalid ACTQ batch selection' unless %w[0 1].include?(actq_batch)
rms_batch=ENV.fetch('CANDIDATE_RMS','0')
raise 'invalid RMS batch selection' unless %w[0 1].include?(rms_batch)
chain=ENV.fetch('CANDIDATE_CHAIN','0')
raise 'invalid chain width' unless chain=='0' || (chain.match?(/\A\d+\z/) && (2..255).cover?(chain.to_i))
rope=ENV["CANDIDATE_ROPE_OWNER"]
extra=ENV.fetch("CANDIDATE_ENV","").split
raise "invalid CANDIDATE_ENV" unless extra.all? { |kv| kv.match?(/\A[A-Z_][A-Z0-9_]*=[\w.,\/-]*\z/) }
pairs=ENV["CANDIDATE_TWO_POSITION"]
raise "invalid two-position setting" if pairs && !%w[0 1].include?(pairs)
chunk=ENV["CANDIDATE_PREFILL_CHUNK"]
raise "invalid prefill chunk" if chunk && !chunk.match?(/\A\d+\z/)
raise "invalid rope owner" if rope && !rope.match?(/\A-?\d+\z/)
prefill_profile=ENV.fetch('CAPTURE_PREFILL_PROFILE','0')
raise 'invalid prefill profile selection' unless %w[0 1].include?(prefill_profile)
trace=Integer(ENV.fetch('TRACE_COUNT','0'))
raise 'invalid trace count' unless (0..8).cover?(trace)
expected='954b27bf92fc968004aafa69a6d5896153b6fa03340f768b63483ffc0137fe3c'
raise 'engine library identity changed' unless Digest::SHA256.file(engine).hexdigest == expected
def command(text)
  Timeout.timeout(120) { UNIXSocket.open('/tmp/flash.sock') { |s| s.puts(text); s.gets || raise(IOError,'core EOF') } }
end
def save(label,value)
  _,err,status=Open3.capture3('gpuq','put',label,stdin_data:JSON.generate(value))
  raise err unless status.success?
end
before=command('config')
raise 'unexpected current image' unless before.include?("cubin=#{original} ")
before_rope=before[/\brope_owner=(-?\d+)/,1]
raise "serving rope owner unknown" if rope && !before_rope
before_pairs=before[/\btwo_position=(\d)/,1]
before_chunk=before[/\bprefill_chunk=(\d+)/,1]
raise "serving prefill settings unknown" if (pairs && !before_pairs) || (chunk && !before_chunk)
save('candidate-artifacts',{engine_sha256:expected,original:original,candidate:candidate,candidate_sha256:Digest::SHA256.file(candidate).hexdigest,candidate_engine:candidate_engine,candidate_engine_sha256:Digest::SHA256.file(candidate_engine).hexdigest,config:before})
source=File.join(File.dirname(candidate),'gemv_bf16_pair.sass')
save('candidate-source',{path:source,content:File.read(source)}) if File.file?(source)
begin
  reply=command("reload #{candidate_engine} FLASH_CUBIN=#{candidate} FLASH_PREFILL_BATCH=#{batch} FLASH_Q2_BATCH=#{q2_batch} FLASH_Q3_BATCH=#{q3_batch} FLASH_HCUP_BATCH=#{hcup_batch} FLASH_HCR_BATCH=#{hcr_batch} FLASH_DNSTEP_BATCH=#{dnstep_batch} FLASH_ACTQ_BATCH=#{actq_batch} FLASH_RMS_BATCH=#{rms_batch} FLASH_PROFILE_PREFILL=#{prefill_profile} FLASH_TRACE=#{trace} FLASH_CHAIN=#{chain}#{rope ? " FLASH_ROPE_OWNER=#{rope}" : ""}#{pairs ? " FLASH_TWO_POSITION=#{pairs}" : ""}#{chunk ? " FLASH_PREFILL_CHUNK=#{chunk}" : ""}#{extra.map { |kv| " #{kv}" }.join}")
  save('candidate-reload',reply)
  raise reply if reply.start_with?('error')
  config=command('config')
  raise 'candidate not loaded' unless config.include?("cubin=#{candidate} ")
  save('candidate-allocated',{config:config,gpu_memory:IO.popen(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader'],&:read)})
  ok=system({'BASELINE_JOB'=>'63668679-0f08-49f2-b283-22969e66b4bf'},RbConfig.ruby,File.join(__dir__,'measure.rb'),'753710')
  raise 'candidate measurement failed' unless ok
ensure
  restore="reload #{engine} FLASH_CUBIN=#{original} FLASH_PREFILL_BATCH=0 FLASH_Q2_BATCH=0 FLASH_Q3_BATCH=0 FLASH_HCUP_BATCH=0 FLASH_HCR_BATCH=0 FLASH_DNSTEP_BATCH=0 FLASH_ACTQ_BATCH=0 FLASH_RMS_BATCH=0 FLASH_PROFILE_PREFILL=0 FLASH_TRACE=0 FLASH_CHAIN=0"
  restore+=" FLASH_ROPE_OWNER=#{before_rope}" if rope
  restore+=" FLASH_TWO_POSITION=#{before_pairs}" if pairs
  restore+=" FLASH_PREFILL_CHUNK=#{before_chunk}" if chunk
  extra.each { |kv| restore+=" #{kv.split("=",2)[0]}=" }
  begin
    reply=command(restore)
    raise IOError,reply if reply.start_with?('error')
  rescue Errno::ECONNRESET,Errno::ECONNREFUSED,Errno::ENOENT,IOError => error
    save('candidate-recovery-needed',error.message)
    raise 'core recovery failed' unless system(RbConfig.ruby,'/home/nate/codex/flash-astra-perf-20261018/recover-resume.rb')
    reply=command(restore+' FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf')
  end
  after=command('config')
  save('candidate-restored',{reply:reply,config:after})
  raise 'original configuration not restored' unless after==before
end
