require 'socket'
require 'timeout'
require 'open3'
require 'json'
require 'fiddle'
raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
RECOVER='/home/nate/codex/flash-astra-perf-20261018/recover-resume.rb'
RELOAD='reload /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so FLASH_CUBIN=/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin FLASH_ATTN_Q_SPLIT=1 FLASH_ROPE_OWNER=5 FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_PREFILL_BATCH=0 FLASH_TRACE=0 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf'
def query(command)
  Timeout.timeout(120) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      pid=s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii').first
      s.puts(command); line=s.gets or raise 'core EOF'
      raise line if line.start_with?('error')
      [pid,line]
    end
  end
end
def stop_core(pid)
  return unless File.exist?("/proc/#{pid}/exe")
  raise 'unexpected process' unless File.basename(File.readlink("/proc/#{pid}/exe"))=='flashcore'
  Process.kill('TERM',pid)
  Timeout.timeout(20) do
    loop do
      break unless File.exist?("/proc/#{pid}/stat")
      break if File.read("/proc/#{pid}/stat").split(') ',2).last.start_with?('Z ')
      sleep 0.1
    end
  end
rescue Errno::ESRCH,Errno::ENOENT
end
def save_bytes(label,bytes)
  _,err,status=Open3.capture3('gpuq','put',label,stdin_data:bytes)
  raise err unless status.success?
end
fifo=File.join(ENV.fetch('GPUQ_RUN_DIR'),'cuda-core.pipe')
raise 'mkfifo failed' unless system('mkfifo','-m','600',fifo)
reader=Thread.new do
  bytes=File.open(fifo,'rb',&:read)
  save_bytes('cuda-lightweight-core.bin',bytes) unless bytes.empty?
  bytes
end
original_pid,original_config=query('config')
save_bytes('coredump-original-config',original_config)
debug_pid=nil
data=nil
begin
  stop_core(original_pid)
  env={'CUDA_ENABLE_COREDUMP_ON_EXCEPTION'=>'1','CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION'=>'0',
       'CUDA_ENABLE_LIGHTWEIGHT_COREDUMP'=>'1','CUDA_COREDUMP_FILE'=>fifo}
  raise 'diagnostic core startup failed' unless system(env,RbConfig.ruby,RECOVER)
  debug_pid,_=query(RELOAD)
  save_bytes('coredump-diagnostic-config',query('config').last)
  ok=system({'BASELINE_JOB'=>'63668679-0f08-49f2-b283-22969e66b4bf','ALLOW_OUTPUT_DIFFERENCE'=>'1','MEASURE_PAIRS'=>'2'},RbConfig.ruby,File.join(__dir__,'measure.rb'),'753710')
  puts JSON.generate(client_completed:ok,diagnostic_core:debug_pid)
ensure
  # Unblock a reader when the requests completed without a GPU exception.
  begin
    File.open(fifo,File::WRONLY|File::NONBLOCK,&:close) if reader.alive?
  rescue Errno::ENXIO
  end
  data=reader.value if reader.join(60)
  stop_core(debug_pid) if debug_pid
  raise 'original core recovery failed' unless system(RbConfig.ruby,RECOVER)
  query(RELOAD)
  save_bytes('coredump-restored-config',query('config').last)
  File.unlink(fifo) if File.exist?(fifo)
end
raise 'no GPU core captured' if !data || data.empty?
puts JSON.generate(captured_bytes:data.bytesize)
memfd=Fiddle::Function.new(Fiddle::Handle::DEFAULT['memfd_create'],[Fiddle::TYPE_VOIDP,Fiddle::TYPE_INT],Fiddle::TYPE_INT)
fd=memfd.call("flash-cuda-core\0",0)
raise 'memfd_create failed' if fd<0
IO.open(fd,'w+b',autoclose:true) do |file|
  file.binmode.write(data); file.flush; file.rewind
  path="/proc/#{Process.pid}/fd/#{fd}"
  commands=['set pagination off',"target cudacore #{path}",'info cuda kernels','info cuda threads',
            'info registers','p/x $errorpc','x/32i $errorpc-64','x/24i $pc-48']
  debugger='/home/nate/.cache/flash-cuda-gdb-11.4/bin/cuda-gdb'
  system(debugger,'-nx','-nh','--batch',*commands.flat_map { |c| ['-ex',c] })
end
