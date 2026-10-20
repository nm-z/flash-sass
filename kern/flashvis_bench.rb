#!/usr/bin/env ruby
# Compare the same projector/image through flashvis's public Unix-socket protocol.
# Run under gpuq so timings, diagnostics, source identities and GPU samples persist.
require 'socket'
require 'json'
require 'digest'
require 'timeout'
require 'fileutils'

abort 'usage: flashvis_bench.rb BINARY MMPROJ TEXT_GGUF IMAGE [MEASURED_RUNS] [TEST_SHIM]' unless (4..6).cover?(ARGV.length)
abort 'run this comparison through gpuq run' unless ENV['GPUQ_JOB_ID'] && ENV['GPUQ_RUN_DIR']
binary, projector, text_model, image = ARGV.first(4).map { |p| File.realpath(p) }
runs = Integer(ARGV[4] || '3')
abort 'MEASURED_RUNS must be between 1 and 10' unless (1..10).cover?(runs)
shim = File.realpath(ARGV[5] || File.join(File.dirname(binary), 'flashvis_fault.so'))
payload = File.binread(image)
image_sha = Digest::SHA256.hexdigest(payload)
work = File.join(ENV.fetch('GPUQ_RUN_DIR'), 'flashvis-comparison')
FileUtils.mkdir_p(work)
clock = -> { Process.clock_gettime(Process::CLOCK_MONOTONIC) }
inputs = [projector, text_model, image].to_h { |p| [p, [File.size(p), File.mtime(p).to_f]] }
puts JSON.generate(event: 'workload', hostname: Socket.gethostname, image: image,
                   image_sha256: image_sha, image_bytes: payload.bytesize, inputs: inputs,
                   threads: 10, warmups_per_backend: 1, measured_runs: runs,
                   numerical_reference: 'CPU with standard attention via test-only interposer',
                   test_shim_sha256: Digest::SHA256.file(shim).hexdigest,
                   binary_sha256: Digest::SHA256.file(binary).hexdigest)
$stdout.sync = true

def read_exact(socket, count)
  result = +''.b
  while result.bytesize < count
    chunk = socket.read(count - result.bytesize)
    raise "short flashvis response: #{result.bytesize}/#{count}" if chunk.nil? || chunk.empty?
    result << chunk
  end
  result
end

def metric(name, value, unit, dimensions)
  puts JSON.generate(gpuq_metric: name, value: value, unit: unit, dimensions: dimensions)
end

results = {}
reference = nil
default_reference = nil
%w[cpu cpu-standard auto].each do |backend|
  socket_path = File.join(work, "#{backend}.sock")
  ready = Queue.new
  timings = Queue.new
  reader, writer = IO.pipe
  reader.binmode  # model metadata can contain UTF-8 even under the queue's C locale
  startup = clock.call
  env = { 'FLASHVIS_DEVICE' => backend == 'auto' ? 'auto' : 'cpu',
          'FLASHVIS_TEST_CPU_STANDARD' => nil, 'FLASHVIS_TEST_FAULT' => nil }
  if backend == 'cpu-standard'
    env['FLASHVIS_TEST_CPU_STANDARD'] = '1'
    env['LD_PRELOAD'] = [shim, ENV['LD_PRELOAD']].compact.reject(&:empty?).join(':')
  end
  standard_reference_enabled = false
  pid = Process.spawn(env, binary, projector, text_model, socket_path, '10',
                      out: $stdout, err: writer, pgroup: true)
  writer.close
  diagnostics = Thread.new do
    reader.each_line do |line|
      $stderr.write(line)
      standard_reference_enabled = true if line.include?('issue96-test: standard-attention CPU reference')
      ready << Regexp.last_match(1) if line.match(/\Aflashvis: ready, backend ([^,]+),/)
      if line.match(/preprocess ([0-9.]+) ms, encode ([0-9.]+) s, backend ([^\s]+)/)
        timings << [Regexp.last_match(1).to_f, Regexp.last_match(2).to_f, Regexp.last_match(3)]
      end
    end
  end
  begin
    Timeout.timeout(120) do
      until File.socket?(socket_path)
        ended = Process.waitpid(pid, Process::WNOHANG)
        raise "flashvis #{backend} exited during startup: #{$?}" if ended
        sleep 0.05
      end
    end
    selected = Timeout.timeout(5) { ready.pop }
    raise 'auto mode fell back to CPU; GPU comparison was not performed' if backend == 'auto' && selected == 'CPU'
    raise 'forced CPU mode selected a GPU' if backend != 'auto' && selected != 'CPU'
    raise 'standard-attention reference interposer was not active' if backend == 'cpu-standard' && !standard_reference_enabled
    metric('vision.startup.host_seconds', clock.call - startup, 's', backend: backend, device: selected)
    UNIXSocket.open(socket_path) do |socket|
      socket.write("enc 4\njunk")
      raise 'malformed image was not rejected' unless Timeout.timeout(5) { socket.gets } == "err cannot decode the image\n"
    end
    samples = []
    (0..(backend == 'cpu-standard' ? 0 : runs)).each do |iteration|
      response = Timeout.timeout(180) do
        UNIXSocket.open(socket_path) do |socket|
          started = clock.call
          socket.write("enc #{payload.bytesize}\n")
          socket.write(payload)
          header = socket.gets
          raise "flashvis #{backend}: #{header.inspect}" unless header&.match?(/\Aok (\d+) (\d+) (\d+) (\d+) (\d+)\n\z/)
          dimensions = header.split.drop(1).map { |x| Integer(x) }
          tokens, nx, ny, positions, width = dimensions
          raise "invalid dimensions #{dimensions}" unless tokens.positive? && width.positive? && tokens <= 16_384 && width <= 65_536
          embeddings = read_exact(socket, 4 * tokens * width)
          rope = read_exact(socket, 12 * tokens)
          elapsed = clock.call - started
          [dimensions, embeddings, rope, elapsed]
        end
      end
      dims, embeddings, rope, elapsed = response
      preprocess_ms, encode_s, actual_device = Timeout.timeout(5) { timings.pop }
      raise "encoder changed from #{selected} to #{actual_device}; comparison requires the selected backend" unless actual_device == selected
      values = embeddings.unpack('e*')
      raise 'non-finite embedding' unless values.all?(&:finite?)
      metric('vision.image_roundtrip.host_seconds', elapsed, 's',
             backend: backend, iteration: iteration, warmup: iteration.zero?,
             image_sha256: image_sha, image_tokens: dims[0])
      metric('vision.encode.host_seconds', encode_s, 's', backend: backend, device: selected, iteration: iteration,
             warmup: iteration.zero?, image_sha256: image_sha)
      metric('vision.preprocess.host_milliseconds', preprocess_ms, 'ms', backend: backend, iteration: iteration,
             warmup: iteration.zero?, image_sha256: image_sha)
      samples << elapsed unless iteration.zero?
      if backend == 'cpu' && iteration.zero?
        default_reference = [dims, values, rope]
      elsif backend == 'cpu-standard'
        reference = [dims, values, rope]
      elsif backend == 'auto' && iteration.zero?
        { 'cpu-default-attention' => default_reference, 'cpu-standard-attention' => reference }.each do |reference_name, reference|
        raise 'GPU changed token dimensions or M-RoPE positions' unless reference[0] == dims && reference[2] == rope
        norm_cpu = 0.0
        norm_gpu = 0.0
        dot = 0.0
        error = 0.0
        min_token_cosine = 1.0
        max_token_relative_l2 = 0.0
        row_cpu = row_gpu = row_dot = row_error = 0.0
        reference[1].each_with_index do |a, i|
          b = values[i]
          norm_cpu += a * a
          norm_gpu += b * b
          dot += a * b
          error += (a - b)**2
          row_cpu += a * a
          row_gpu += b * b
          row_dot += a * b
          row_error += (a - b)**2
          if (i + 1) % dims[4] == 0
            raise 'zero-norm image token' unless row_cpu.positive? && row_gpu.positive?
            min_token_cosine = [min_token_cosine, row_dot / Math.sqrt(row_cpu * row_gpu)].min
            max_token_relative_l2 = [max_token_relative_l2, Math.sqrt(row_error / row_cpu)].max
            row_cpu = row_gpu = row_dot = row_error = 0.0
          end
        end
        raise 'zero-norm embedding' unless norm_cpu.positive? && norm_gpu.positive?
        cosine = dot / Math.sqrt(norm_cpu * norm_gpu)
        relative_l2 = Math.sqrt(error / norm_cpu)
        puts JSON.generate(event: 'embedding_comparison', reference: reference_name,
                           exact_dimensions: true, exact_mrope_positions: true,
                           cosine_similarity: cosine, relative_l2: relative_l2, values: values.length,
                           min_token_cosine: min_token_cosine, max_token_relative_l2: max_token_relative_l2)
        unless cosine >= 0.99 && relative_l2 <= 0.1
          raise 'GPU/CPU global embedding comparison failed'
        end
        if reference_name == 'cpu-standard-attention' && !(min_token_cosine >= 0.99 && max_token_relative_l2 <= 0.1)
          raise 'GPU/CPU embedding comparison failed (cosine >= 0.99, relative L2 <= 0.1 required globally and per token)'
        end
        end
      end
    end
    next if backend == 'cpu-standard'  # numerical reference only; production CPU is the latency baseline
    sorted = samples.sort
    results[backend] = (sorted[(sorted.length - 1) / 2] + sorted[sorted.length / 2]) / 2.0
    metric('vision.image_roundtrip.median_host_seconds', results[backend], 's',
           backend: backend, image_sha256: image_sha, measured_runs: runs)
  ensure
    begin
      Process.kill('TERM', -pid)
      Timeout.timeout(5) { Process.waitpid(pid) }
    rescue Errno::ESRCH, Errno::ECHILD
    rescue Timeout::Error
      Process.kill('KILL', -pid) rescue Errno::ESRCH
      Process.waitpid(pid) rescue Errno::ECHILD
    end
    File.unlink(socket_path) if File.exist?(socket_path)
    diagnostics.join(5)
    reader.close
  end
end
raise 'input file changed during comparison' unless inputs.all? { |p, metadata| metadata == [File.size(p), File.mtime(p).to_f] }
speedup = results.fetch('cpu') / results.fetch('auto')
metric('vision.image_roundtrip.speedup', speedup, 'ratio', image_sha256: image_sha)
puts JSON.generate(event: 'comparison_complete', median_seconds: results, speedup: speedup)
