require 'socket'
require 'timeout'
require 'json'
require 'fileutils'
root=ARGV.fetch(0)
raise 'existing deployment record' if File.exist?(root)
FileUtils.mkdir_p(root)
local=ENV.fetch('PLE_SOURCE','/home/nate/codex/flash-astra-perf-20261018/ple-cache.gguf')
source='/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf'
raise 'copy incomplete' unless File.file?(local) && File.size(local)==File.size(source)
log=File.open(File.join(root,'control.jsonl'),'w');log.sync=true
query=lambda do |cmd|
  Timeout.timeout(180) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      s.puts(cmd);line=s.gets or raise 'core EOF'
      log.puts(JSON.generate(command:cmd,response:line))
      raise line if line.start_with?('error')
      line
    end
  end
end
original=query.call('config').split.drop(1).map{|x|x.split('=',2)}.to_h
pid=UNIXSocket.open('/tmp/flash.sock'){|s|s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii').first}
libs=File.readlines("/proc/#{pid}/maps").filter_map{|s|s.split.last if s =~ /\/tmp\/flashcore-#{pid}-gen\d+\.so\s*$/}.uniq
raise 'ambiguous original library' unless libs.size==1
backup=File.join(root,'original-libengine.so');FileUtils.cp(libs.first,backup)
File.write(File.join(root,'original.json'),JSON.pretty_generate(config:original,pid:pid))
settings="FLASH_CUBIN=#{original.fetch('cubin')} FLASH_ENGINE_MAXT=#{original.fetch('maxt')} FLASH_MAXIMG=#{original.fetch('img_rows')} FLASH_KV_HOST=#{original.fetch('kv_host')} FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=0 FLASH_PLE_READERS=1 FLASH_TRACE=0 FLASH_KV_TRACE=0 FLASH_GRAPH_CUBIN= FLASH_GRAPH_ANCHOR= FLASH_SNAPSHOT_HOST=0 FLASH_ROPE_OWNER=-1"
begin
  query.call("reload /home/nate/codex/flash-astra-build-20261018/ple-local/kern/libengine.so #{settings} FLASH_PLE_FILE=#{local}")
  config=query.call('config')
  parsed=config.split.drop(1).map{|x|x.split('=',2)}.to_h
  raise 'wrong source' unless parsed['ple_file']==local
  %w[maxt img_rows cubin_fnv layers split attn_map].each{|key|raise "changed #{key}" unless parsed[key]==original[key]}
  File.write(File.join(root,'deployed-config.txt'),config)
  puts config
rescue
  query.call("reload #{backup} #{settings} FLASH_PLE_FILE=")
  File.write(File.join(root,'restored-config.txt'),query.call('config'))
  raise
end
