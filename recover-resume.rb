require 'socket'
require 'timeout'
require 'json'
require 'fileutils'
root='/home/nate/codex/flash-astra-perf-20261018/recovery-resume'
FileUtils.mkdir_p(root)
socket='/tmp/flash.sock'
query=lambda do
  Timeout.timeout(8) do
    UNIXSocket.open(socket) { |s| s.write("config\n"); [s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii')[0],s.gets] }
  end
end
public_pid=nil
begin
  public_pid,response=query.call
  if response&.start_with?('config ')
    File.write(File.join(root,'already-healthy.txt'),response)
    puts response
    exit 0
  end
  raise "unexpected response: #{response.inspect}" unless response&.include?('no engine loaded')
rescue Errno::ECONNREFUSED,Errno::ENOENT
  public_pid=nil
end
# Only recover an absent service or a service explicitly reporting no engine.
# An unresponsive active request is left to its owner; Timeout::Error propagates.
if public_pid
  args=File.binread("/proc/#{public_pid}/cmdline").split("\0")
  env=File.binread("/proc/#{public_pid}/environ").split("\0")
  raise 'not the public core' unless File.basename(args[0])=='flashcore' && env.include?("FLASH_SOCK=#{socket}")
  File.write(File.join(root,'failed-core.json'),JSON.pretty_generate(pid:public_pid,args:args)+"\n")
  Process.kill('TERM',public_pid)
  Timeout.timeout(20) do
    loop do
      break unless File.exist?("/proc/#{public_pid}/stat")
      break if File.read("/proc/#{public_pid}/stat").split(') ',2)[1].start_with?('Z ')
      sleep 0.2
    end
  end
end
compute=IO.popen(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],&:read)
File.write(File.join(root,'compute-before.txt'),compute)
raise 'another core is still active; retry through gpuq after its job finishes' if compute.lines.any? { |line| File.basename(line.split(',').last.strip)=='flashcore' }
env={'CUDA_DEVICE_ORDER'=>'PCI_BUS_ID','CUDA_VISIBLE_DEVICES'=>'0,1,2,3,4,5,6',
     'FLASH_ANCHOR_DIE'=>'6','FLASH_GRAPH_CUBIN'=>'','FLASH_ROPE_OWNER'=>'-1','FLASH_SNAPSHOT_HOST'=>'0',
     'FLASH_SOCK'=>socket,'FLASH_KV_HOST'=>'0','FLASH_MAXT'=>'262144','FLASH_ENGINE_MAXT'=>'230000',
     'FLASH_MAXIMG'=>'4096','FLASH_PLE_DISK'=>'1','FLASH_PLE_OFF'=>'192','FLASH_PLE_READERS'=>'1',
     'FLASH_TWO_POSITION'=>'0','FLASH_PREFILL_CHUNK'=>'0','FLASH_KV_TRACE'=>'0','FLASH_TRACE'=>'0',
     'FLASH_CUBIN'=>'/home/nate/codex/flash-astra-perf-20261018/promote-fast-rope/original-die.cubin',
     'SPLIT_SHARES'=>'112,97,81,116,97,97','ATTN_MAP'=>'1,1,1,1,4,4,4,4,5,5,5,5','ATTN_DIE'=>'1'}
args=['/home/nate/claude/flash-sass-live/kern/flashcore',
      '/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf',
      '/home/nate/codex/flash-astra-perf-20261018/promote-fast-rope/original-libengine.so',*(0..5).map { |d| "/tmp/hold-h#{d}.manifest" }]
File.write(File.join(root,'launch.json'),JSON.pretty_generate(environment:env,argv:args)+"\n")
pid=Process.spawn(env,*args,chdir:'/home/nate/claude/flash-sass-live/kern',in:File::NULL,
                  out:[File.join(root,'core.stdout'),'a'],err:[File.join(root,'core.stderr'),'a'],close_others:true,pgroup:true)
File.write(File.join(root,'core.pid'),"#{pid}\n")
Timeout.timeout(30) do
  loop do
    raise 'core exited during startup' if Process.waitpid(pid,Process::WNOHANG)
    begin
      peer,response=query.call
      if peer==pid && response&.start_with?('config ')
        File.write(File.join(root,'restored-config.txt'),response)
        puts response
        Process.detach(pid)
        break
      end
      raise response if peer==pid && response&.start_with?('error')
    rescue Errno::ENOENT,Errno::ECONNREFUSED
    end
    sleep 0.2
  end
end
