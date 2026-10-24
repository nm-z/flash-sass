require 'socket'
require 'timeout'
require 'json'
require 'fileutils'
require 'digest'
base='/home/nate/codex/flash-astra-perf-20261018'
root=base+'/http-qsplit-matched'
raise 'output exists' if File.exist?(root)
FileUtils.mkdir_p(root)
query=lambda do |cmd|
 Timeout.timeout(180){UNIXSocket.open('/tmp/flash.sock'){|s|s.puts(cmd);r=s.gets or raise 'EOF';raise r if r.start_with?('error');r}}
end
reload=lambda do |mode|
 query.call("reload /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so FLASH_CUBIN=/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin FLASH_ATTN_Q_SPLIT=#{mode}")
 query.call('config')
end
runs=[];keep=false
begin
 [0,1].each do |mode|
  File.write(root+"/config-#{mode}.txt",reload.call(mode))
  path=root+"/arm-#{mode}"
  raise 'HTTP run failed' unless system({'FRESH_SET'=>'2','HTTP_SEED'=>'7','HTTP_LIMIT'=>'1'},RbConfig.ruby,base+'/http-real.rb',path)
  data=JSON.parse(File.read(path+'/request-0.json'))
  pieces=Hash.new{|h,k|h[k]=''};timing=nil
  data['events'].each do |event|
   x=event['data'];next unless x.is_a?(Hash)
   timing=x['timings'] if x['timings']
   (x['choices']||[]).each{|c|(c['delta']||{}).each{|k,v|pieces[k]<<v if v.is_a?(String)&&k!='role'}}
  end
  runs << {mode:mode,timings:timing,text_hash:Digest::SHA256.hexdigest(JSON.generate(pieces.sort))}
 end
 raise 'sampled output text differs' unless runs[0][:text_hash]==runs[1][:text_hash]
 keep=runs[1][:timings]['predicted_per_second']>runs[0][:timings]['predicted_per_second']*1.01
 File.write(root+'/summary.json',JSON.pretty_generate(runs:runs,query_split_retained:keep))
 puts JSON.generate(runs:runs,query_split_retained:keep)
ensure
 reload.call(0) unless keep
end
