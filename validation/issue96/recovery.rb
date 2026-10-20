#!/usr/bin/env ruby
# Real daemon recovery; fault.c is loaded only into diagnostic subprocesses.
require 'socket'
require 'json'
require 'digest'
require 'timeout'
require 'fileutils'
require 'open3'
abort 'usage: recovery.rb BINARY MMPROJ TEXT_GGUF IMAGE FAULT_SO [large]' unless (5..6).cover?(ARGV.length)
abort 'run through gpuq' unless ENV['GPUQ_JOB_ID'] && ENV['GPUQ_RUN_DIR']
binary, projector, text_model, image, fault = ARGV.first(5).map { |p| File.realpath(p) }
work = File.join(ENV.fetch('GPUQ_RUN_DIR'), 'vision-recovery')
FileUtils.mkdir_p(work)
$stdout.sync = true
payload = File.binread(image)

def bitmap(width, height)
  pixels = ((width * 3 + 3) / 4) * 4 * height
  'BM'.b + [54 + pixels, 0, 54, 40, width, height].pack('V6') +
    [1, 24].pack('v2') + [0, pixels, 0, 0, 0, 0].pack('V6') + ("\x80".b * pixels)
end

def read_exact(socket, count)
  data = +''.b
  while data.bytesize < count
    part = socket.read(count - data.bytesize)
    raise 'short encoder response' if part.nil? || part.empty?
    data << part
  end
  data
end

def request(path, payload, timings, backend)
  Timeout.timeout(300) do
    UNIXSocket.open(path) do |socket|
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      socket.write("enc #{payload.bytesize}\n")
      socket.write(payload)
      header = socket.gets
      raise "unexpected header: #{header.inspect}" unless header&.match?(/\Aok (\d+) (\d+) (\d+) (\d+) (\d+)\n\z/)
      dims = header.split.drop(1).map(&:to_i)
      raise "unexpected dimensions #{dims}" unless dims[0].between?(1, 4096) && dims[4] == 2560
      embeddings = read_exact(socket, dims[0] * dims[4] * 4)
      positions = read_exact(socket, dims[0] * 12)
      raise 'non-finite output escaped the daemon' unless embeddings.unpack('e*').all?(&:finite?)
      actual = Timeout.timeout(5) { timings.pop }
      raise "expected #{backend}, got #{actual}" unless actual == backend
      puts JSON.generate(event: 'recovery_response', backend: actual, dimensions: dims,
                         input_sha256: Digest::SHA256.hexdigest(payload),
                         embeddings_sha256: Digest::SHA256.hexdigest(embeddings),
                         roundtrip_seconds: Process.clock_gettime(Process::CLOCK_MONOTONIC) - start)
      [dims, embeddings, positions]
    end
  end
end

def run_case(label, env, binary, projector, text_model, work, payload, initial, fallback)
  path = File.join(work, "#{label}.sock")
  reader, writer = IO.pipe
  reader.binmode
  lines = +''.b
  timings, ready = Queue.new, Queue.new
  pid = Process.spawn(env, binary, projector, text_model, path, '10', err: writer, out: $stdout, pgroup: true)
  writer.close
  diagnostics = Thread.new do
    reader.each_line do |line|
      lines << line
      $stderr.write(line)
      ready << Regexp.last_match(1) if line.match(/\Aflashvis: ready, backend ([^,]+),/)
      timings << Regexp.last_match(1) if line.match(/encode [0-9.]+ s, backend ([^\s]+)/)
    end
  end
  begin
    Timeout.timeout(120) do
      until File.socket?(path)
        raise 'daemon exited before listening' if Process.waitpid(pid, Process::WNOHANG)
        sleep 0.05
      end
    end
    selected = Timeout.timeout(5) { ready.pop }
    correct = initial == :gpu ? selected != 'CPU' : selected == initial
    raise "wrong startup device #{selected}" unless correct
    result = request(path, payload, timings, 'CPU')
    if fallback
      request(path, bitmap(64, 64), timings, 'CPU')
      raise 'expected exactly one request-time fallback' unless lines.scan(/switching to CPU until restart/).length == 1
    else
      raise 'unexpected fallback' if lines.include?('switching to CPU until restart')
    end
    puts JSON.generate(event: 'recovery_case_passed', case: label, initial_device: selected, fallback: fallback)
    result
  ensure
    begin
      Process.kill('TERM', -pid)
      Timeout.timeout(5) { Process.waitpid(pid) }
    rescue Errno::ESRCH, Errno::ECHILD
    rescue Timeout::Error
      Process.kill('KILL', -pid) rescue Errno::ESRCH
      Process.waitpid(pid) rescue Errno::ECHILD
    end
    diagnostics.join(5)
    reader.close
    File.unlink(path) if File.exist?(path)
  end
end

if ARGV[5] == 'large'
  result = run_case('large', {'FLASHVIS_DEVICE' => 'auto'}, binary, projector, text_model,
                    work, bitmap(2048, 2048), :gpu, true)
  raise 'large image lost tokens' unless result[0] == [4096, 64, 64, 64, 2560]
else
  _, error, status = Open3.capture3({'FLASHVIS_DEVICE' => 'not-a-real-device'}, binary, projector, text_model, File.join(work, 'invalid.sock'))
  raise 'invalid GPU must exit 2' unless status.exitstatus == 2 && error.include?('not an available GPU')
  puts JSON.generate(event: 'recovery_case_passed', case: 'invalid-device', exit_code: 2)
  baseline = run_case('cpu', {'FLASHVIS_DEVICE' => 'cpu'}, binary, projector, text_model, work, payload, 'CPU', false)
  hidden = run_case('no-gpu', {'FLASHVIS_DEVICE' => 'auto', 'CUDA_VISIBLE_DEVICES' => '-1'},
                    binary, projector, text_model, work, payload, 'CPU', false)
  raise 'no-GPU fallback changed CPU output' unless baseline == hidden
  %w[error nan].each do |mode|
    recovered = run_case(mode, {'FLASHVIS_DEVICE' => 'auto', 'LD_PRELOAD' => fault, 'FLASHVIS_TEST_FAULT' => mode},
                         binary, projector, text_model, work, payload, :gpu, true)
    raise "#{mode} recovery changed CPU output" unless baseline == recovered
  end
end
puts JSON.generate(event: 'recovery_complete', mode: ARGV[5] || 'faults')
