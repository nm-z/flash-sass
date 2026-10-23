require 'socket'
require 'timeout'
require 'open3'
require 'json'
raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
pid=UNIXSocket.open('/tmp/flash.sock') { |s| s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii').first }
raise 'unexpected target' unless File.basename(File.readlink("/proc/#{pid}/exe"))=='flashcore'
debugger='/home/nate/.cache/flash-cuda-gdb-11.4/bin/cuda-gdb'
commands=['set pagination off','set confirm off','set print thread-events off',
          'set cuda api_failures stop',"attach #{pid}",'info inferiors',
          'echo GPU_DEBUG_READY\n','continue','echo GPU_DEBUG_STOPPED\n',
          'info cuda kernels','info cuda threads','info registers','p/x $errorpc',
          'x/24i $pc-48','bt','detach','quit']
argv=['sudo','-n',debugger,'-nx','-nh','--batch',*commands.flat_map { |c| ['-ex',c] }]
ready=Queue.new
attached=false
stopped=false
client_ok=false
Open3.popen2e(*argv) do |input,output,wait|
  input.close
  reader=Thread.new do
    output.each_line do |line|
      STDOUT.write(line); STDOUT.flush
      attached=true if line.include?("process #{pid}")
      ready << :ready if line.include?('GPU_DEBUG_READY')
      stopped=true if line.include?('GPU_DEBUG_STOPPED')
    end
    ready << :ended
  end
  begin
    state=Timeout.timeout(60) { ready.pop }
    raise 'debugger did not attach' unless state==:ready && attached
    client_ok=system({'BASELINE_JOB'=>'63668679-0f08-49f2-b283-22969e66b4bf','ALLOW_OUTPUT_DIFFERENCE'=>'1'},RbConfig.ruby,File.join(__dir__,'measure.rb'),'753710')
  ensure
    if wait.alive? && !stopped
      begin
        Process.kill('INT',wait.pid)
      rescue Errno::EPERM
        system('sudo','-n','kill','-INT',wait.pid.to_s)
      rescue Errno::ESRCH
      end
    end
    unless wait.join(20)
      system('sudo','-n','kill','-TERM',wait.pid.to_s)
      raise 'debugger did not exit' unless wait.join(20)
    end
    reader.join
  end
end
puts JSON.generate(debugged_core:pid,client_completed:client_ok,debugger_stopped:stopped)
# The established recovery routine verifies an absent/no-engine service before
# starting a core. Keep recovery inside this GPU queue allocation.
raise 'core recovery failed' unless system(RbConfig.ruby,'/home/nate/codex/flash-astra-perf-20261018/recover-resume.rb')
Timeout.timeout(120) do
  UNIXSocket.open('/tmp/flash.sock') do |s|
    s.puts('reload /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so FLASH_CUBIN=/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_PREFILL_BATCH=0 FLASH_TRACE=0 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf')
    reply=s.gets or raise 'restore EOF'
    raise reply if reply.start_with?('error')
    puts reply
  end
end
raise 'diagnostic request failed; inspect debugger record' unless client_ok
