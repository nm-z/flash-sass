#!/usr/bin/env ruby
# Compare the same projector/image through flashvis's public Unix-socket protocol.
# Run under gpuq so timings, diagnostics, source identities and GPU samples persist.
require 'socket'
require 'json'
require 'digest'
require 'timeout'
require 'fileutils'

abort 'usage: flashvis_bench.rb BINARY MMPROJ TEXT_GGUF IMAGE [MEASURED_RUNS]' unless (4..5).cover?(ARGV.length)
abort 'run this comparison through gpuq run' unless ENV['GPUQ_JOB_ID'] && ENV['GPUQ_RUN_DIR']
binary, projector, text_model, image = ARGV.first(4).map { |p| File.realpath(p) }
runs = Integer(ARGV[4] || '3')
abort 'MEASURED_RUNS must be between 1 and 10' unless (1..10).cover?(runs)
payload = File.binread(image)
image_sha = Digest::SHA256.hexdigest(payload)
work = File.join(ENV.fetch('GPUQ_RUN_DIR'), 'flashvis-comparison')
FileUtils.mkdir_p(work)
clock = -> { Process.clock_gettime(Process::CLOCK_MONOTONIC) }
inputs = [projector, text_model, image].to_h { |p| [p, [File.size(p), File.mtime(p).to_f]] }
puts JSON.generate(event: 'workload', hostname: Socket.gethostname, image: image,
                   image_sha256: image_sha, image_bytes: payload.bytesize, inputs: inputs,
                   threads: 10, warmups_per_backend: 1, measured_runs: runs,
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
%w[cpu auto].each do |backend|
  socket_path = File.join(work, "#{backend}.sock")
  ready = Queue.new
  timings = Queue.new
  reader, writer = IO.pipe
  reader.binmode  # model metadata can contain UTF-8 even under the queue's C locale
  startup = clock.call
  pid = Process.spawn({ 'FLASHVIS_DEVICE' => backend }, binary, projector, text_model, socket_path, '10',
                      out: $stdout, err: writer, pgroup: true)
  writer.close
  diagnostics = Thread.new do
    reader.each_line do |line|
      $stderr.write(line)
      ready << Regexp.last_match(1) if line.match(/\Aflashvis: ready, backend ([^,]+),/)
      timings << [Regexp.last_match(1).to_f, Regexp.last_match(2).to_f] if line.match(/preprocess ([0-9.]+) ms, encode ([0-9.]+) s/)
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
    raise 'forced CPU mode selected a GPU' if backend == 'cpu' && selected != 'CPU'
    metric('vision.startup.host_seconds', clock.call - startup, 's', backend: backend, device: selected)
    samples = []
    (0..runs).each do |iteration|
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
      preprocess_ms, encode_s = Timeout.timeout(5) { timings.pop }
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
        reference = [dims, values, rope]
      elsif backend == 'auto' && iteration.zero?
        raise 'GPU changed token dimensions or M-RoPE positions' unless reference[0] == dims && reference[2] == rope
        norm_cpu = 0.0
        norm_gpu = 0.0
        dot = 0.0
        error = 0.0
        reference[1].each_with_index do |a, i|
          b = values[i]
          norm_cpu += a * a
          norm_gpu += b * b
          dot += a * b
          error += (a - b)**2
        end
        raise 'zero-norm embedding' unless norm_cpu.positive? && norm_gpu.positive?
        cosine = dot / Math.sqrt(norm_cpu * norm_gpu)
        relative_l2 = Math.sqrt(error / norm_cpu)
        puts JSON.generate(event: 'embedding_comparison', exact_dimensions: true, exact_mrope_positions: true,
                           cosine_similarity: cosine, relative_l2: relative_l2, values: values.length)
        raise 'GPU/CPU embedding comparison failed (cosine >= 0.99, relative L2 <= 0.1 required)' unless cosine >= 0.99 && relative_l2 <= 0.1
      end
    end
    results[backend] = samples.sort[samples.length / 2]
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
