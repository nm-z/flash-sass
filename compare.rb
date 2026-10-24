#!/usr/bin/env ruby
require 'socket'
require 'json'
require 'digest'
require 'fileutils'
require 'timeout'
require 'time'
root, mode = ARGV
abort 'usage: compare.rb NEW_OUTPUT_DIRECTORY decode|prefill' unless root && %w[decode prefill].include?(mode)
raise 'output already exists' if File.exist?(root)
FileUtils.mkdir_p(root)
socket_path='/tmp/flash.sock'
build='/home/nate/codex/flash-astra-build-20261018'
log=File.open(File.join(root,'control.jsonl'),'w');log.sync=true
query=lambda do |command|
  Timeout.timeout(180) do
    UNIXSocket.open(socket_path) do |s|
      s.write(command+"\n")
      response=s.gets or raise 'EOF on control command'
      log.puts JSON.generate(at:Time.now.utc.iso8601,command:command,response:response)
      raise response if response.start_with?('error')
      response
    end
  end
end
# Check all queued component outcomes before changing the resident library.
%w[hc tree hc4 jump loop reg single].each do |candidate|
  hc = File.read(File.join(__dir__, "component", "#{candidate}-hc.log"))
  qsa = File.readlines(File.join(__dir__, "component", "#{candidate}-qsa.log")).grep(/^visible=[0-9]+ complete=/)
  raise "component checks failed or incomplete: #{candidate}" unless hc.include?("cases, 0 failed;") && qsa.length == 4 && qsa.all? { |line| line.rstrip.end_with?(" EXACT") }
end
original=query.call('config')
config=original.split.drop(1).map { |s| s.split('=',2) }.to_h
raise 'unexpected runtime' unless config['layers']=='48/48' && config['dies']=='6' && config['nblk']=='512'
pid=UNIXSocket.open(socket_path) { |s| s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii')[0] }
libraries=File.readlines("/proc/#{pid}/maps").filter_map { |line| line.split.last if line =~ /\/tmp\/flashcore-#{pid}-gen\d+\.so\s*$/ }.uniq
raise "ambiguous loaded library: #{libraries}" unless libraries.size==1
backup_lib=File.join(root,'original-libengine.so')
backup_cubin=File.join(root,'original-die.cubin')
FileUtils.cp(libraries.first,backup_lib)
original_cubin=config.fetch('cubin')
original_cubin=File.expand_path(original_cubin,File.readlink("/proc/#{pid}/cwd")) unless original_cubin.start_with?('/')
FileUtils.cp(original_cubin,backup_cubin)
fnv=File.binread(backup_cubin).bytes.reduce(0xcbf29ce484222325) { |h,b| ((h ^ b)*0x100000001b3)&0xffffffffffffffff }
raise 'loaded cubin path was replaced since loading' unless '%016x'%fnv==config.fetch('cubin_fnv')
settings={'FLASH_ENGINE_MAXT'=>config.fetch('maxt'),'FLASH_KV_HOST'=>config.fetch('kv_host'),
          'FLASH_MAXIMG'=>config.fetch('img_rows'),'FLASH_TWO_POSITION'=>config.fetch('two_position'),
          'FLASH_PREFILL_CHUNK'=>config.fetch('prefill_chunk'),'FLASH_PLE_READERS'=>config.fetch('ple_readers','1'),
          'FLASH_KV_TRACE'=>config.fetch('kv_trace','0'),
          'FLASH_PREFILL_HC_MODE'=>config.fetch('prefill_hc_mode','0'),
          'FLASH_ROPE_OWNER'=>config.fetch('rope_owner','-1'),
          'FLASH_ATTN_Q_SPLIT'=>config.fetch('attn_q_split','0'),
          'FLASH_SNAPSHOT_HOST'=>config.fetch('snapshot_host','0'),
          'FLASH_PACKED_PREFILL'=>config.fetch('packed_prefill','0'),
          'FLASH_GRAPH_ANCHOR'=>config.fetch('graph_anchor',''),
          'FLASH_GRAPH_LOCAL'=>config.fetch('graph_local','0'),
          'FLASH_GRAPH_TOKEN_TIMEOUT'=>config.fetch('graph_token_timeout','300'),
          'FLASH_PLE_FILE'=>config.fetch('ple_cache',config.fetch('ple_file','core')) == 'core' ? '' : config.fetch('ple_cache',config.fetch('ple_file')),
          'FLASH_GRAPH_CUBIN'=>config.fetch('graph_cubin','off') == 'off' ? '' : config.fetch('graph_cubin')}
File.write(File.join(root,'original.json'),JSON.pretty_generate(config:config,pid:pid,library:libraries.first,
    library_sha256:Digest::SHA256.file(backup_lib).hexdigest,cubin_sha256:Digest::SHA256.file(backup_cubin).hexdigest,settings:settings)+"\n")
changed=false
begin
  arms= mode=='decode' ? ENV.fetch('BENCH_ARMS','base,jump,base,loop,base').split(',') : ENV.fetch('PREFILL_ARMS','scalar,headless,paired,scalar').split(',')
  arms.each_with_index do |arm,index|
    env=settings.merge('FLASH_TWO_POSITION'=>'0','FLASH_PREFILL_CHUNK'=>'0','FLASH_KV_TRACE'=>'0','FLASH_PLE_READERS'=>'1')
    image=mode=='decode' ? arm : ENV.fetch('PREFILL_IMAGE','loop')
    if ENV['QSPLIT_ARM_NAME']
      env['FLASH_ATTN_Q_SPLIT']=arm==ENV['QSPLIT_ARM_NAME'] ? ENV.fetch('QSPLIT_MODE','1') : '0'
      image=ENV.fetch('QSPLIT_ARM_IMAGE','iq4-bias') if arm==ENV['QSPLIT_ARM_NAME']
    end
    env['FLASH_ATTN_Q_SPLIT']=ENV['TEST_ATTN_Q_SPLIT'] if ENV['TEST_ATTN_Q_SPLIT']
    if mode=='prefill' && ENV['PREFILL_IMAGES']
      images=ENV.fetch('PREFILL_IMAGES').split(',')
      raise 'PREFILL_IMAGES must match arm count' unless images.length==arms.length
      image=images.fetch(index)
    end
    if ENV['GRAPH_ARM_NAME']
      env['FLASH_GRAPH_CUBIN']=arm==ENV['GRAPH_ARM_NAME'] ? ENV.fetch('TEST_GRAPH_CUBIN') : ''
      image=ENV.fetch('GRAPH_ARM_IMAGE','base') if arm==ENV['GRAPH_ARM_NAME']
    end
    env['FLASH_ROPE_OWNER']=ENV['TEST_ROPE_OWNER'] if ENV['TEST_ROPE_OWNER']
    env['FLASH_TWO_POSITION']=ENV['TEST_TWO_POSITION'] if ENV['TEST_TWO_POSITION']
    env['FLASH_PREFILL_CHUNK']=ENV['TEST_PREFILL_CHUNK'] if ENV['TEST_PREFILL_CHUNK']
    env['FLASH_GRAPH_CUBIN']=ENV['TEST_GRAPH_CUBIN'] if ENV['TEST_GRAPH_CUBIN'] && !ENV['GRAPH_ARM_NAME']
    env['FLASH_GRAPH_LOCAL']=ENV.fetch('TEST_GRAPH_LOCAL','0') unless env['FLASH_GRAPH_CUBIN'].to_s.empty?
    env['FLASH_GRAPH_TOKEN_TIMEOUT']=ENV.fetch('TEST_GRAPH_TOKEN_TIMEOUT','300')
    env['FLASH_SNAPSHOT_HOST']=ENV['TEST_SNAPSHOT_HOST'] if ENV['TEST_SNAPSHOT_HOST']
    env['FLASH_PACKED_PREFILL']=ENV['TEST_PACKED_PREFILL'] if ENV['TEST_PACKED_PREFILL']
    env['FLASH_GRAPH_ANCHOR']=ENV['TEST_GRAPH_ANCHOR'] if ENV['TEST_GRAPH_ANCHOR']
    env['FLASH_ENGINE_MAXT']=ENV['TEST_MAXT'] if ENV['TEST_MAXT']
    env['FLASH_MAXIMG']=ENV['TEST_MAXIMG'] if ENV['TEST_MAXIMG']
    env['FLASH_PLE_FILE']=ENV['TEST_PLE_FILE'] if ENV['TEST_PLE_FILE']
    if mode=='prefill'
      env['FLASH_MAXIMG']=ENV.fetch('PREFILL_MAXIMG','0')
      env['FLASH_PREFILL_CHUNK']='64' unless arm=='scalar'
      env['FLASH_TWO_POSITION']='1' if %w[paired hcdown hcfull].include?(arm)
      env['FLASH_PREFILL_HC_MODE']= arm=='hcdown' ? '1' : arm=='hcfull' ? '2' : '0'
    end
    cubin="#{build}/#{image}/kern/die.cubin"
    library="#{build}/#{image=='base' ? 'base' : 'hc'}/kern/libengine.so"
    library=ENV.fetch('PREFILL_LIBRARY',library) if mode=='prefill'
    library=ENV.fetch('BENCH_LIBRARY',library) if mode=='decode'
    raise 'missing candidate' unless File.file?(cubin) && File.file?(library)
    if image!='base' && !ENV['BENCH_INTEGRATION']
      check=File.join(__dir__,'component',"#{image}-hc.log")
      raise "component check missing for #{image}" unless File.file?(check) && File.read(check).include?('cases, 0 failed;')
    end
    changed=true
    query.call("reload #{library} FLASH_CUBIN=#{cubin} #{env.map { |k,v| "#{k}=#{v}" }.join(' ')}")
    actual=query.call('config')
    actual_fields=actual.split.drop(1).map { |s| s.split('=',2) }.to_h
    if !env['FLASH_GRAPH_CUBIN'].to_s.empty?
      raise 'graph image not active' unless actual_fields['graph_cubin']==env['FLASH_GRAPH_CUBIN']
    end
    File.write(File.join(root,"#{index}-#{arm}-config.txt"),actual)
    if ENV['BENCH_MEMORY']
      File.write(File.join(root,"#{index}-#{arm}-memory.csv"),IO.popen(['nvidia-smi','--query-gpu=index,memory.free','--format=csv'],&:read))
    end
    run=File.join(root,"#{index}-#{arm}")
    telemetry=File.open(File.join(root,"#{index}-#{arm}-telemetry.csv"),'w')
    monitor=Process.spawn('nvidia-smi','--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu','--format=csv','-lms','500',out:telemetry,err:telemetry)
    begin
      case_name=mode=='decode' ? 'chat64' : 'repeat512'
      case_name=ENV.fetch('BENCH_CASE_OVERRIDE',case_name)
      ok=system({'BENCH_CASE'=>case_name},RbConfig.ruby,File.join(__dir__,'bench.rb'),run,socket_path,arm)
      raise "benchmark failed: #{arm}" unless ok
    ensure
      Process.kill('TERM',monitor) rescue Errno::ESRCH
      Process.wait(monitor) rescue Errno::ECHILD
      telemetry.close
    end
    records=JSON.parse(File.read(File.join(run,'results.json'))).fetch('runs')
    reference=@reference ||= records.first.fetch('output_ids')
    raise "token mismatch: #{arm}" unless records.all? { |r| r.fetch('output_ids')==reference }
    puts JSON.generate(arm:arm,index:index,runs:records.map { |r| r.slice('warmup','decode_tps','engine_prefill_ms') });STDOUT.flush
  end
ensure
  if changed
    restore_command="reload #{backup_lib} FLASH_CUBIN=#{backup_cubin} #{settings.map { |k,v| "#{k}=#{v}" }.join(' ')}"
    begin
      query.call(restore_command)
    rescue Errno::ECONNREFUSED, Errno::ENOENT, Errno::ECONNRESET
      raise 'core recovery failed' unless system(RbConfig.ruby,File.join(__dir__,'recover-resume.rb'))
      query.call(restore_command)
    end
    restored=query.call('config')
    restored_fields=restored.split.drop(1).map { |s| s.split('=',2) }.to_h
    %w[rev kv_host ple_disk maxt layers nblk dies split attn_map img_rows two_position prefill_chunk cubin_fnv].each do |key|
      raise "restoration mismatch: #{key}" unless restored_fields[key]==config[key]
    end
    File.write(File.join(root,'restored-config.txt'),restored)
  end
end
