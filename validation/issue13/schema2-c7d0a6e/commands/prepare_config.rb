#!/usr/bin/env ruby
# Prepare explicit identities/configs for the current idle resident core.
# Does not reload the engine or launch a GPU kernel.
require 'json'
require 'digest'
require 'fileutils'
require 'open3'
require 'socket'

root,pid_text,socket,revision,baseline_library,baseline_cubin = ARGV
abort 'usage: prepare_config.rb RUN_ROOT CORE_PID SOCKET REV BASELINE_LIBRARY BASELINE_CUBIN' unless baseline_cubin
root = File.realpath(root); pid = Integer(pid_text)
src = File.join(root,'kern'); evidence = File.join(root,'evidence','schema2')
FileUtils.mkdir_p(evidence)
require File.join(src,'trace_lifecycle_check.rb')
def query(socket,command)
  c = Core.new(socket,10)
  begin
    c.send(command); c.line
  ensure
    c.close
  end
end
def sha(p)
  Digest::SHA256.file(p).hexdigest
end
def fnv(p)
  h = 0xcbf29ce484222325
  File.binread(p).each_byte { |b| h = ((h ^ b)*0x100000001b3) & 0xffffffffffffffff }
  format('%016x',h)
end
def save(p,value)
  File.write(p,JSON.pretty_generate(value)+"\n")
end
line = query(socket,'config'); baseline = TraceChecks.fields(line)
raise 'unexpected baseline configuration' unless baseline['rev'] == 'edc66d9' && baseline['dies'] == '6' &&
  baseline['layers'] == '48/48' && baseline['maxt'] == '230000'
raise 'baseline cubin hash differs from loaded image' unless baseline['cubin_fnv'] == fnv(baseline_cubin)
core_cmd_hash = sha("/proc/#{pid}/cmdline")
holders = []
out,err,status = Open3.capture3('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits')
raise err unless status.success?
out.lines.map(&:strip).reject(&:empty?).map { |s| Integer(s) }.uniq.sort.each do |p|
  comm = File.read("/proc/#{p}/comm").strip
  next unless %w[hold fence].include?(comm) || (comm == 'flashcore' && p == pid)
  holders << {pid: p,command: comm,cmdline_sha256: sha("/proc/#{p}/cmdline")}
end
raise 'resident core was not found in GPU process inventory' unless holders.any? { |h| h[:pid] == pid }
save(File.join(evidence,'baseline.json'),{config: baseline,raw: line,core_pid: pid,
  core_cmdline_sha256: core_cmd_hash,library: baseline_library,library_sha256: sha(baseline_library),
  cubin: baseline_cubin,cubin_sha256: sha(baseline_cubin)})
FileUtils.cp(baseline_library,File.join(evidence,'baseline-library.so'))
FileUtils.cp(baseline_cubin,File.join(evidence,'baseline-die.cubin'))
File.write(File.join(evidence,'weights-manifest.txt'),
  (0...6).map { |d| f="/tmp/hold-h#{d}.manifest"; "#{sha(f)}  #{f}\n" }.join)
common_env = {
  'FLASH_ENGINE_MAXT'=>'512','FLASH_MAXIMG'=>'1','FLASH_NBLK'=>'512','FLASH_PLE_DISK'=>'1',
  'FLASH_KV_HOST'=>'0','FLASH_TWO_POSITION'=>'0','FLASH_PREFILL_CHUNK'=>'0','FLASH_WATCHDOG_S'=>'30'
}
restore_env = {
  'FLASH_ENGINE_MAXT'=>baseline.fetch('maxt'),'FLASH_MAXIMG'=>baseline.fetch('img_rows'),
  'FLASH_NBLK'=>baseline.fetch('nblk'),'FLASH_PLE_DISK'=>baseline.fetch('ple_disk'),
  'FLASH_KV_HOST'=>baseline.fetch('kv_host'),'FLASH_TWO_POSITION'=>baseline.fetch('two_position'),
  'FLASH_PREFILL_CHUNK'=>baseline.fetch('prefill_chunk'),'FLASH_WATCHDOG_S'=>'600',
  'FLASH_TRACE'=>'0','FLASH_TRACE_CTAS'=>'1','FLASH_CUBIN'=>baseline.fetch('cubin')
}
lib = File.join(src,'libengine.so')
control = {
  socket: socket,core_pid: pid,core_cmdline_sha256: core_cmd_hash,baseline_config: baseline,
  gpu_indices: (0...Integer(baseline.fetch('dies'))).to_a,trace_memory_reserve_mib: 8,
  allowed_revisions: [baseline.fetch('rev'),revision],state_file: File.join(evidence,'borrow-state.json'),
  guard_directory: File.join(evidence,'reload-preflights'),guard_config: File.join(evidence,'config-cta1.json'),
  modes: {'restore'=>{library: baseline_library,library_sha256: sha(baseline_library),
    cubin_file: baseline_cubin,cubin_file_sha256: sha(baseline_cubin),
    environment: restore_env,expected_config: baseline}}
}
[1,16].each do |ctas|
  ordinary = File.join(src,'die.cubin')
  traced = File.join(src,ctas == 1 ? 'die_trace.cubin' : 'die_trace16.cubin')
  common = baseline.merge('rev'=>revision,'maxt'=>'512','img_rows'=>'1','trace_ctas'=>ctas.to_s,
    'trace_schema'=>'2','two_position'=>'0','prefill_chunk'=>'0').reject { |k,_| %w[cubin cubin_fnv trace_tokens].include?(k) }
  modes = {}
  {'ordinary'=>ordinary,'traced'=>traced}.each do |name,image|
    count = name == 'ordinary' ? 0 : 4; label = "#{name}#{ctas}"
    env = common_env.merge('FLASH_TRACE_CTAS'=>ctas.to_s,'FLASH_TRACE'=>count.to_s,'FLASH_CUBIN'=>image)
    expected = common.merge('cubin'=>image,'cubin_fnv'=>fnv(image),'trace_tokens'=>count.to_s)
    control[:modes][label] = {library: lib,library_sha256: sha(lib),cubin_file: image,cubin_file_sha256: sha(image),
      environment: env,expected_config: expected}
    modes[name] = {cubin: image,cubin_fnv: fnv(image),trace_tokens: count,
      prepare_argv: ['ruby',File.join(src,'trace_control.rb'),File.join(evidence,'control.json'),label]}
  end
  config = {
    socket: socket,lock_path: '/tmp/flash-sass-engine-validation.lock',prompt_ids: [10,11],new: 3,
    repeats: 3,warmups: 2,timeout_seconds: 120,busy_wait_seconds: 600,allowed_holders: holders,
    expected_common: common,modes: modes,
    restore_argv: ['ruby',File.join(src,'trace_control.rb'),File.join(evidence,'control.json'),'restore'],
    identity_paths: {library: lib,source_manifest: File.join(evidence,'source-manifest.txt'),
      weights_manifest: File.join(evidence,'weights-manifest.txt'),ordinary_cubin: ordinary,traced_cubin: traced}
  }
  save(File.join(evidence,"config-cta#{ctas}.json"),config)
end
life_image = File.join(src,'die_trace.cubin')
life_common = control[:modes]['traced1'][:expected_config].merge('trace_tokens'=>'16','two_position'=>'1')
control[:modes]['lifecycle'] = {library: lib,library_sha256: sha(lib),
  cubin_file: life_image,cubin_file_sha256: sha(life_image),
  environment: common_env.merge('FLASH_TRACE_CTAS'=>'1','FLASH_TRACE'=>'16','FLASH_CUBIN'=>life_image,'FLASH_TWO_POSITION'=>'1'),
  expected_config: life_common}
control[:modes]['lifecycle-ordinary'] = Marshal.load(Marshal.dump(control[:modes]['lifecycle']))
life_ordinary = control[:modes]['lifecycle-ordinary']
life_ordinary[:cubin_file] = File.join(src,'die.cubin')
life_ordinary[:cubin_file_sha256] = sha(life_ordinary[:cubin_file])
life_ordinary[:environment] = life_ordinary[:environment].merge('FLASH_TRACE'=>'0','FLASH_CUBIN'=>life_ordinary[:cubin_file])
life_ordinary[:expected_config] = life_ordinary[:expected_config].merge('trace_tokens'=>'0','cubin'=>life_ordinary[:cubin_file],'cubin_fnv'=>fnv(life_ordinary[:cubin_file]))
save(File.join(evidence,'control.json'),control)
puts "Prepared baseline and three explicit test modes for source #{revision}; no reload performed"
