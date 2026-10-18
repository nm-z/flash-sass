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
