require 'socket'
require 'timeout'
root='/home/nate/codex/flash-astra-perf-20261018'
raise 'incomplete direct copy' unless File.size(root+'/ple-direct.gguf')==28800138432
query=lambda do |command|
  Timeout.timeout(180) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      s.puts(command);response=s.gets or raise 'core EOF';raise response if response.start_with?('error');puts response;response
    end
  end
end
query.call("reload /home/nate/codex/flash-astra-build-20261018/prefill-start/kern/libengine.so FLASH_CUBIN=/home/nate/codex/flash-astra-build-20261018/release/kern/die.cubin FLASH_PLE_FILE=#{root}/ple-direct.gguf FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_GRAPH_CUBIN= FLASH_GRAPH_LOCAL=0 FLASH_SNAPSHOT_HOST=0 FLASH_ENGINE_MAXT=230000 FLASH_MAXIMG=4096")
config=query.call('config')
raise 'wrong cache source' unless config.include?("ple_file=#{root}/ple-direct.gguf ")
File.write(root+'/direct-deployed-config.txt',config)
STDOUT.flush
raise 'HTTP requests failed' unless system({'FRESH_SET'=>'6'},RbConfig.ruby,root+'/http-real.rb',root+'/http-direct-fresh')
