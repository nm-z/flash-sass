require 'json'
source='/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf'
target='/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
exit 0 if File.file?(target)
Process.kill('TERM',2462134) rescue Errno::ESRCH
[2462150,2462151].each do |pid|
  next unless File.file?("/proc/#{pid}/cmdline")
  args=File.binread("/proc/#{pid}/cmdline").split("\0")
  raise 'unexpected old copier' unless File.basename(args.first)=='dd' && args.any?{|a|a=="if=#{source}" || a=="of=#{target}.tmp"}
  Process.kill('TERM',pid)
end
[2462150,2462151].each do |pid|
  while File.file?("/proc/#{pid}/stat") && !File.read("/proc/#{pid}/stat").split(') ',2).last.start_with?('Z ')
    sleep 0.05
  end
end
block=1<<20; bytes=File.size(source); total=bytes/block
resume=[(File.size(target+'.tmp')/(8*block)-1)*8,0].max
reader,writer=IO.pipe
begin; writer.fcntl(1031,1<<20); rescue Errno::EPERM; end
puts JSON.generate(resume_bytes:resume*block,total_bytes:bytes,pipe_bytes:writer.fcntl(1032));STDOUT.flush
read_pid=Process.spawn('ionice','-c','2','-n','7','nice','-n','10','dd',"if=#{source}",'bs=1M',"skip=#{resume}","count=#{total-resume}",'iflag=direct,fullblock','status=none',out:writer)
write_pid=Process.spawn('ionice','-c','2','-n','7','nice','-n','10','dd',"of=#{target}.tmp",'bs=1M',"seek=#{resume}",'iflag=fullblock','oflag=direct','conv=notrunc,fsync','status=progress',in:reader)
reader.close;writer.close
File.write('/home/nate/codex/flash-astra-perf-20261018/ple-copy-pids.json',JSON.generate(reader:read_pid,writer:write_pid,parent:Process.pid))
statuses=[read_pid,write_pid].map{|pid|Process.waitpid2(pid).last}
raise 'direct copy failed' unless statuses.all?(&:success?)
tail=bytes-total*block
File.open(source,'rb') do |input|
  input.seek(total*block);last=input.read(tail);raise 'short tail' unless last.bytesize==tail
  File.open(target+'.tmp','r+b'){|out|out.seek(total*block);out.write(last);out.flush;out.fsync}
end
raise 'wrong copy size' unless File.size(target+'.tmp')==bytes
File.rename(target+'.tmp',target)
puts JSON.generate(complete_bytes:bytes)
