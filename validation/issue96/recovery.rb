#!/usr/bin/env ruby
# Explicit failures must stay failures; the next request still uses GPU FA.
require 'socket'
require 'json'
require 'timeout'
require 'open3'
abort 'usage: recovery.rb BINARY PROJECTOR TEXT_GGUF IMAGE FAULT_SO' unless ARGV.length == 5
abort 'run through gpuq' unless ENV['GPUQ_JOB_ID'] && ENV['GPUQ_RUN_DIR']
binary, projector, model, image, fault = ARGV.map { |p| File.realpath(p) }
payload = File.binread(image)
$stdout.sync = true
[['cpu', {}], ['not-a-real-device', {}], ['auto', {'CUDA_VISIBLE_DEVICES' => '-1'}]].each do |device, env|
  _, diagnostics, status = Open3.capture3(env.merge('FLASHVIS_DEVICE' => device), binary, projector, model, "/tmp/fs96-invalid-#{Process.pid}.sock")
  raise "#{device} did not reject startup" unless status.exitstatus == 2 && diagnostics.match?(/not an available GPU|GPU Flash Attention is required/)
  puts JSON.generate(event: 'vision_failure_check', case: "unavailable-#{device}", exit: status.exitstatus)
end

def request(path, bytes)
  Timeout.timeout(120) do
    UNIXSocket.open(path) do |socket|
      socket.write("enc #{bytes.bytesize}\n")
      socket.write(bytes)
      header = socket.gets
      raise 'closed encoder connection' unless header
      return header if header.start_with?('err ')
      raise "bad header #{header.inspect}" unless header.match?(/\Aok \d+ \d+ \d+ \d+ \d+\n\z/)
      dims = header.split.drop(1).map(&:to_i)
      count = dims[0] * dims[4] * 4
      data, positions = socket.read(count), socket.read(dims[0] * 12)
      raise 'short or invalid successful response' unless data&.bytesize == count && positions&.bytesize == dims[0] * 12 && data.unpack('e*').all?(&:finite?)
      return dims
    end
  end
end

%w[malformed oversized error nan].each do |mode|
  path = "/tmp/fs96-recovery-#{Process.pid}.sock"
  env = {'FLASHVIS_DEVICE' => 'auto', 'LD_PRELOAD' => %w[malformed oversized].include?(mode) ? nil : fault, 'FLASHVIS_TEST_FAULT' => mode}
  reader, writer = IO.pipe
  reader.binmode
  diagnostics = +''.b
  pid = Process.spawn(env, binary, projector, model, path, '10', err: writer, pgroup: true)
  writer.close
  thread = Thread.new { reader.each_line { |line| diagnostics << line; $stderr.write(line) } }
  begin
    Timeout.timeout(120) do
      until File.socket?(path)
        raise 'encoder exited during startup' if Process.waitpid(pid, Process::WNOHANG)
        sleep 0.05
      end
    end
    input = mode == 'malformed' ? 'not an image' : payload
    if mode == 'oversized'
      w, h = 4097, 4096
      size = (w * 3 + 3) / 4 * 4 * h
      input = 'BM'.b + [54 + size, 0, 54, 40, w, h].pack('V6') + [1, 24].pack('v2') + [0, size, 0, 0, 0, 0].pack('V6') + ("\x80".b * size)
    end
    rejected = request(path, input)
    raise "#{mode} was accepted: #{rejected.inspect}" unless rejected.is_a?(String) && rejected.start_with?('err ')
    accepted = request(path, payload)
    raise "subsequent GPU request failed: #{accepted.inspect}" unless accepted.is_a?(Array)
    raise 'backend changed to CPU' if diagnostics.match?(/flashvis:.*(?:switching to CPU|backend CPU)/)
    raise 'GPU FA not observed after failure' unless diagnostics.include?('observed_gpu=27') && diagnostics.include?('backend=CUDA')
    puts JSON.generate(event: 'vision_failure_check', case: mode, rejection: rejected.strip, subsequent_dimensions: accepted, subsequent_backend: 'GPU FA')
  ensure
    Process.kill('TERM', -pid) rescue Errno::ESRCH
    Process.waitpid(pid) rescue Errno::ECHILD
    thread.join(5)
    reader.close
    File.unlink(path) if File.exist?(path)
  end
end
