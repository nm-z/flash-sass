#!/usr/bin/env ruby
# Direct, uncached encoder check. Output tensors stay in the queue's run dir.
require 'socket'
require 'timeout'
require 'json'
require 'digest'
abort 'usage: encode.rb BINARY PROJECTOR TEXT_GGUF IMAGE|WIDTHxHEIGHT ...' if ARGV.length < 4
abort 'run through gpuq' unless ENV['GPUQ_JOB_ID'] && ENV['GPUQ_RUN_DIR']
$stdout.sync = true
binary, projector, model = ARGV.shift(3).map { |p| File.realpath(p) }
path = "/tmp/fs96-#{Process.pid}.sock"
pid = Process.spawn(binary, projector, model, path, '10', pgroup: true)
begin
  Timeout.timeout(120) do
    until File.socket?(path)
      raise 'encoder exited before listening' if Process.waitpid(pid, Process::WNOHANG)
      sleep 0.05
    end
  end
  ARGV.each_with_index do |input, index|
    expected = nil
    if input.match?(/\A\d+x\d+\z/)
      w, h = input.split('x').map(&:to_i)
      stride = (w * 3 + 3) / 4 * 4
      # Deterministic colored stripes exercise spatial positions without noise.
      row = (0...w).map { |x| [x % 251, (x / 16) % 251, 127].pack('C3') }.join.b
      row << "\0" * (stride - row.bytesize)
      pixels = row * h
      payload = 'BM'.b + [54 + pixels.bytesize, 0, 54, 40, w, h].pack('V6') +
                [1, 24].pack('v2') + [0, pixels.bytesize, 0, 0, 0, 0].pack('V6') + pixels
      expected = [w / 32 * (h / 32), w / 32, h / 32, [w, h].max / 32, 2560] if w % 32 == 0 && h % 32 == 0 && w*h >= 65536
    else
      payload = File.binread(input)
    end
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    Timeout.timeout(3600) do
      UNIXSocket.open(path) do |socket|
        socket.write("enc #{payload.bytesize}\n")
        socket.write(payload)
        header = socket.gets
        raise "encoder response: #{header.inspect}" unless header&.match?(/\Aok (\d+) (\d+) (\d+) (\d+) (\d+)\n\z/)
        dims = header.split.drop(1).map(&:to_i)
        raise "wrong dimensions #{dims}, expected #{expected}" if expected && dims != expected
        size = dims[0] * dims[4] * 4
        embeddings = socket.read(size)
        positions = socket.read(dims[0] * 12)
        raise 'short response' unless embeddings&.bytesize == size && positions&.bytesize == dims[0] * 12
        raise 'non-finite embeddings' unless embeddings.unpack('e*').all?(&:finite?)
        positions.unpack('V*').each_slice(3).with_index do |pos, i|
          raise "wrong M-RoPE position #{i}: #{pos}" unless pos == [0, i / dims[1], i % dims[1]]
        end
        File.binwrite(File.join(ENV.fetch('GPUQ_RUN_DIR'), "embeddings-#{index}.bin"), embeddings)
        puts JSON.generate(event: 'vision_encode_passed', input: input, dimensions: dims,
                           input_sha256: Digest::SHA256.hexdigest(payload), embeddings_sha256: Digest::SHA256.hexdigest(embeddings),
                           roundtrip_seconds: Process.clock_gettime(Process::CLOCK_MONOTONIC) - start)
      end
    end
  end
ensure
  Process.kill('TERM', -pid) rescue Errno::ESRCH
  Process.waitpid(pid) rescue Errno::ECHILD
  File.unlink(path) if File.exist?(path)
end
