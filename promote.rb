#!/usr/bin/env ruby
require 'socket'
require 'timeout'
require 'json'
require 'fileutils'
require 'digest'
root = ARGV.fetch(0)
raise 'output exists' if File.exist?(root)
FileUtils.mkdir_p(root)
socket = '/tmp/flash.sock'
log = File.open(File.join(root, 'control.jsonl'), 'w'); log.sync = true
query = lambda do |cmd|
  Timeout.timeout(180) do
    UNIXSocket.open(socket) do |s|
      s.puts(cmd)
      reply = s.gets or raise 'control EOF'
      log.puts(JSON.generate(command: cmd, response: reply))
      raise reply if reply.start_with?('error')
      reply
    end
  end
end
parse = ->(line) { line.split.drop(1).map { |x| x.split('=', 2) }.to_h }
original = parse.call(query.call('config'))
raise 'unexpected model' unless original.values_at('layers','dies','nblk') == %w[48/48 6 512]
pid = UNIXSocket.open(socket) { |s| s.getsockopt(Socket::SOL_SOCKET, Socket::SO_PEERCRED).unpack('iii').first }
mapped = File.readlines("/proc/#{pid}/maps").filter_map { |s| s.split.last if s =~ /\/tmp\/flashcore-#{pid}-gen\d+\.so\s*$/ }.uniq
raise 'ambiguous engine mapping' unless mapped.size == 1
backup_lib = File.join(root, 'original-libengine.so')
backup_cubin = File.join(root, 'original-die.cubin')
FileUtils.cp(mapped.first, backup_lib)
image = File.expand_path(original.fetch('cubin'), File.readlink("/proc/#{pid}/cwd"))
FileUtils.cp(image, backup_cubin)
fnv = File.binread(backup_cubin).bytes.reduce(0xcbf29ce484222325) { |h,b| ((h ^ b)*0x100000001b3)&0xffffffffffffffff }
raise 'loaded image changed' unless '%016x' % fnv == original.fetch('cubin_fnv')
settings = {
  'FLASH_ENGINE_MAXT'=>original.fetch('maxt'), 'FLASH_MAXIMG'=>original.fetch('img_rows'),
  'FLASH_KV_HOST'=>original.fetch('kv_host'), 'FLASH_TWO_POSITION'=>original.fetch('two_position'),
  'FLASH_PREFILL_CHUNK'=>original.fetch('prefill_chunk'), 'FLASH_ROPE_OWNER'=>original.fetch('rope_owner','-1'),
  'FLASH_ATTN_Q_SPLIT'=>original.fetch('attn_q_split','0'),
  'FLASH_PLE_READERS'=>original.fetch('ple_readers','1'), 'FLASH_KV_TRACE'=>original.fetch('kv_trace','0'),
  'FLASH_TRACE'=>original.fetch('trace_tokens','0')
}
settings['FLASH_PLE_FILE']=original.fetch('ple_cache',original.fetch('ple_file','core'))=='core' ? '' : original.fetch('ple_cache',original.fetch('ple_file'))
File.write(File.join(root,'original.json'), JSON.pretty_generate(config:original,pid:pid,settings:settings))
reload = lambda do |lib,cubin,env|
  query.call("reload #{lib} FLASH_CUBIN=#{cubin} #{env.map { |k,v| "#{k}=#{v}" }.join(' ')}")
  parse.call(query.call('config'))
end
measure = lambda do |label,case_name|
  out = File.join(root,"#{label}-#{case_name}")
  raise 'benchmark failed' unless system({'BENCH_CASE'=>case_name,'BENCH_REPEATS'=>'2'}, RbConfig.ruby,
    File.join(__dir__,'bench.rb'),out,socket,label)
  records = JSON.parse(File.read(File.join(out,'results.json'))).fetch('runs')
  raise 'unstable tokens' unless records.all? { |r| r.fetch('output_ids') == records.first.fetch('output_ids') }
  records.last
end
changed = false; promoted = false
begin
  baseline = %w[chat64 repeat512].to_h { |c| [c,measure.call('resident',c)] }
  candidate = ENV.fetch('CANDIDATE_DIR','/home/nate/codex/flash-astra-build-20261018/fast/kern')
  raise 'missing build' unless File.file?("#{candidate}/die.cubin") && File.file?("#{candidate}/libengine.so")
  env = settings.merge('FLASH_ROPE_OWNER'=>'5','FLASH_TWO_POSITION'=>'1','FLASH_PREFILL_CHUNK'=>'64','FLASH_TRACE'=>'0')
  changed = true
  config = reload.call("#{candidate}/libengine.so","#{candidate}/die.cubin",env)
  raise 'capacity changed' unless config.values_at('maxt','img_rows') == original.values_at('maxt','img_rows')
  faster = %w[chat64 repeat512].to_h { |c| [c,measure.call('candidate',c)] }
  %w[chat64 repeat512].each do |c|
    raise "token mismatch: #{c}" unless baseline[c].fetch('output_ids') == faster[c].fetch('output_ids')
  end
  decode_gain = faster['chat64'].fetch('engine_decode_tps',faster['chat64'].fetch('decode_tps')) / baseline['chat64'].fetch('engine_decode_tps',baseline['chat64'].fetch('decode_tps'))
  prefill_gain = baseline['repeat512'].fetch('engine_prefill_ms') / faster['repeat512'].fetch('engine_prefill_ms')
  summary = {decode_gain:decode_gain,prefill_gain:prefill_gain,baseline:baseline.transform_values { |v| v.slice('decode_tps','engine_prefill_ms') },
    candidate:faster.transform_values { |v| v.slice('decode_tps','engine_prefill_ms') }, config:config}
  File.write(File.join(root,'summary.json'),JSON.pretty_generate(summary))
  raise "no measured improvement: #{summary}" unless decode_gain > 1.02 && prefill_gain > 1.10
  File.write(File.join(root,'promoted-config.txt'),query.call('config'))
  promoted = true
  puts JSON.generate(summary.merge(promoted:true))
ensure
  if changed && !promoted
    restored = reload.call(backup_lib,backup_cubin,settings)
    %w[rev maxt img_rows kv_host two_position prefill_chunk cubin_fnv].each do |k|
      raise "restore mismatch: #{k}" unless restored[k] == original[k]
    end
    File.write(File.join(root,'restored-config.json'),JSON.pretty_generate(restored))
  end
end
