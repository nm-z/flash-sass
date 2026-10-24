require 'socket'
require 'timeout'
require 'fileutils'
require 'json'
root='/home/nate/codex/flash-astra-perf-20261018'
record=root+'/qsplit-live'
raise 'deployment record exists' if File.exist?(record)
FileUtils.mkdir_p(record)
query=lambda do |cmd|
  Timeout.timeout(180) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      s.puts(cmd);line=s.gets or raise 'control EOF'
      raise line if line.start_with?('error')
      puts line;line
    end
  end
end
parse=->(line){line.split.drop(1).map{|v|v.split('=',2)}.to_h}
before=parse.call(query.call('config'))
raise 'unexpected runtime' unless before.values_at('layers','dies','maxt','img_rows')==%w[48/48 6 230000 4096]
pid=UNIXSocket.open('/tmp/flash.sock'){|s|s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii').first}
mapped=File.readlines("/proc/#{pid}/maps").filter_map{|s|s.split.last if s =~ /\/tmp\/flashcore-#{pid}-gen\d+\.so\s*$/}.uniq
raise 'ambiguous mapped library' unless mapped.size==1
backup=record+'/original-libengine.so';FileUtils.cp(mapped.first,backup)
File.write(record+'/original.json',JSON.pretty_generate(before))
complete=false
begin
  query.call('reload /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so FLASH_CUBIN=/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin FLASH_ATTN_Q_SPLIT=1')
  actual=query.call('config');fields=parse.call(actual)
  before.each{|k,v|raise "changed setting #{k}" unless %w[rev cubin attn_q_split attn_q_mask].include?(k)||fields[k]==v}
  raise 'query split inactive' unless fields['attn_q_split']=='1'
  File.write(record+'/deployed-config.txt',actual);STDOUT.flush
  raise 'HTTP failed' unless system({'FRESH_SET'=>'6'},RbConfig.ruby,root+'/http-real.rb',root+'/http-qsplit-live')
  complete=true
ensure
  unless complete
    query.call("reload #{backup} FLASH_CUBIN=#{before.fetch('cubin')} FLASH_ATTN_Q_SPLIT=#{before.fetch('attn_q_split','0')}")
  end
end
