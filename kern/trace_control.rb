#!/usr/bin/env ruby
# Explicitly configured idle-core reload/restore helper. The caller owns the
# cooperative validation lock; this helper never exits or kills the core.
require 'json'
require 'digest'
require_relative 'trace_lifecycle_check'
require_relative 'trace_validation'

begin
  raise 'usage: trace_control.rb CONTROL.json MODE' unless ARGV.length == 2
  cfg_path,mode = ARGV
  cfg = JSON.parse(File.read(cfg_path)); base = File.dirname(File.realpath(cfg_path))
  raise 'caller must hold the validation lock' unless ENV['FLASH_TRACE_VALIDATION_LOCK_HELD'] == '1'
  pid = Integer(cfg.fetch('core_pid'))
  raise 'resident core process identity changed' unless Digest::SHA256.file("/proc/#{pid}/cmdline").hexdigest == cfg.fetch('core_cmdline_sha256')
  selected = cfg.fetch('modes').fetch(mode)
  %w[library cubin_file].each do |key|
    file = File.expand_path(selected.fetch(key),base)
    raise "#{key} identity changed" unless Digest::SHA256.file(file).hexdigest == selected.fetch(key+'_sha256')
  end
  guard_dir = File.expand_path(cfg.fetch('guard_directory'),base)
  TraceValidation::Runner.new(File.expand_path(cfg.fetch('guard_config'),base)).idle!(
    File.join(guard_dir,Time.now.utc.strftime('%Y%m%dT%H%M%S')+"-#{Process.pid}-#{mode}"),'before-reload')
  query = lambda do |command|
    c = Core.new(cfg.fetch('socket'),120)
    begin
      c.send(command); c.line
    ensure
      c.close
    end
  end
  before_line = query.call('config')
  before = TraceValidation.fields(before_line)
  state_path = File.expand_path(cfg.fetch('state_file'),base)
  unless File.exist?(state_path) || mode == 'restore'
    cfg.fetch('baseline_config').each do |key,value|
      raise "baseline #{key} changed before borrowing core" unless before[key] == value.to_s
    end
    File.write(state_path,JSON.pretty_generate({original_config: before,original_line: before_line})+"\n",
               mode: File::WRONLY|File::CREAT|File::EXCL)
  end
  allowed = cfg.fetch('allowed_revisions')
  raise "unexpected current engine revision #{before['rev']}" unless allowed.include?(before['rev'])
  target = selected.fetch('expected_config').transform_values(&:to_s)
  target_count = Integer(target.fetch('trace_tokens','0'))
  if target_count.positive?
    # Reload frees the old engine's allocations before constructing the new
    # one. Only credit its trace buffer when every other runtime field matches;
    # first prepare an ordinary image with that configuration otherwise.
    same = before.reject { |key,_| TraceValidation::IMAGE_FIELDS.include?(key) }
    wanted = target.reject { |key,_| TraceValidation::IMAGE_FIELDS.include?(key) }
    raise 'prepare the matching ordinary configuration before enabling tracing' unless same == wanted
    raise 'unknown trace allocation schema' unless target.fetch('trace_schema') == '2'
    ctas = Integer(target.fetch('trace_ctas'))
    raise 'invalid bounded trace dimensions' unless [1,16].include?(ctas) && target_count <= 256/ctas
    current_count = Integer(before.fetch('trace_tokens'))
    current_bytes = current_count * 4096 * ctas * 128
    target_bytes = target_count * 4096 * ctas * 128
    extra_bytes = [0,target_bytes-current_bytes].max
    reserve_bytes = Integer(cfg.fetch('trace_memory_reserve_mib',8)) * (1 << 20)
    raise 'invalid trace memory reserve' unless reserve_bytes.positive?
    gpu_indices = cfg.fetch('gpu_indices').map { |index| Integer(index) }
    raise 'GPU inventory does not match die count' unless gpu_indices.uniq.length == Integer(target.fetch('dies'))
    out,err,status = Open3.capture3('nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits')
    raise "memory query failed: #{err}" unless status.success?
    rows = out.lines.map { |line| line.strip.split(',').map(&:strip) }.to_h { |index,uuid,free| [Integer(index),[uuid,Integer(free)]] }
    checks = gpu_indices.map do |index|
      uuid,free_mib = rows.fetch(index); free_bytes = free_mib * (1 << 20)
      {gpu: index,uuid: uuid,free_bytes: free_bytes,current_trace_bytes: current_bytes,
       target_trace_bytes: target_bytes,additional_bytes: extra_bytes,reserve_bytes: reserve_bytes,
       pass: free_bytes >= extra_bytes + reserve_bytes}
    end
    puts JSON.generate(trace_memory_preflight: checks)
    raise 'insufficient memory headroom; reload was not sent' unless checks.all? { |row| row[:pass] }
  end
  env = selected.fetch('environment')
  env.each do |key,value|
    raise 'unsafe reload setting' unless key.match?(/\AFLASH_[A-Z0-9_]+\z/) && value.to_s.match?(/\A[A-Za-z0-9_.,\/-]+\z/)
  end
  library = File.expand_path(selected.fetch('library'),base)
  raise 'reload path cannot contain whitespace' if library.match?(/\s/)
  command = 'reload ' + library + ' ' + env.map { |k,v| "#{k}=#{v}" }.join(' ')
  puts "> #{command}"
  reply = query.call(command); puts reply
  raise 'engine reload failed' unless reply == 'ok: engine reloaded'
  after_line = query.call('config'); puts after_line
  after = TraceValidation.fields(after_line)
  selected.fetch('expected_config').each do |key,value|
    raise "reloaded #{key}=#{after[key].inspect}, expected #{value.inspect}" unless after[key] == value.to_s
  end
  if mode == 'restore'
    cfg.fetch('baseline_config').each do |key,value|
      raise "restoration differs in #{key}" unless after[key] == value.to_s
    end
    puts 'PASS original resident configuration restored'
  end
rescue StandardError => e
  warn "trace_control: #{e.class}: #{e.message}"
  exit 1
end
