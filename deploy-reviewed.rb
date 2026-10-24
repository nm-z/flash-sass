require 'socket'
require 'timeout'
require 'json'
root=ARGV.fetch(0)
summary=JSON.parse(File.read(File.join(root,'summary.json')))
raise 'no gain' unless summary.fetch('decode_gain')>1 && summary.fetch('prefill_gain')>1.1
original=JSON.parse(File.read(File.join(root,'original.json'))).fetch('config')
query=lambda do |cmd|
  Timeout.timeout(180) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      s.puts(cmd);line=s.gets or raise 'EOF';raise line if line.start_with?('error');line
    end
  end
end
current=query.call('config').split.drop(1).map{|x|x.split('=',2)}.to_h
raise 'resident changed since comparison' unless %w[maxt img_rows cubin_fnv rev].all?{|k|current[k]==original[k]}
candidate=summary.fetch('config')
lib='/home/nate/codex/flash-astra-build-20261018/guard/kern/libengine.so'
query.call("reload #{lib} FLASH_CUBIN=#{candidate.fetch('cubin')} FLASH_ENGINE_MAXT=#{current.fetch('maxt')} FLASH_MAXIMG=#{current.fetch('img_rows')} FLASH_KV_HOST=#{current.fetch('kv_host')} FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_TRACE=0 FLASH_KV_TRACE=0 FLASH_PLE_READERS=1")
actual=query.call('config')
File.write(File.join(root,'deployed-config.txt'),actual)
puts actual
