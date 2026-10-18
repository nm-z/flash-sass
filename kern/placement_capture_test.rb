#!/usr/bin/env ruby
# CPU-only socket and safety regression checks; no CUDA or resident service needed.
require_relative 'placement_capture'
require 'timeout'

$count = 0
def assert(value, message = 'assertion failed')
  raise message unless value
end
def check(name)
  Timeout.timeout(8) { yield }
  $count += 1
  puts "ok #{$count} - #{name}"
end
def with_proxy(timeout: 2, handler:)
  Dir.mktmpdir('placement-proxy-test-') do |dir|
    server = UNIXServer.new(File.join(dir, 'engine.sock'))
    peers, received = [], []
    engine = Thread.new do
      begin
        loop do
          peer = server.accept
          peers << peer
          handler.call(peer, received)
          peer.close unless peer.closed?
        end
      rescue IOError, Errno::EBADF, Errno::EPIPE
        # Test teardown closes only sockets created here.
      end
    end
    proxy = PlacementCapture::Proxy.new(File.join(dir, 'proxy.sock'), File.join(dir, 'engine.sock'), timeout: timeout)
    begin
      yield proxy, File.join(dir, 'proxy.sock'), received
    ensure
      proxy.close
      server.close
      peers.each { |p| p.close unless p.closed? }
      engine.join(2) || raise('fake engine did not terminate')
    end
  end
end
def normal_engine(peer, received, fail_profile: false, bad_count: false)
  while (line = peer.gets)
    received << line.chomp
    case line
    when "info\n" then peer.write("info 4096 0\n")
    when /\A(?:stop|sample|snap) / then peer.write("ok\n")
    when /\Achat /
      peer.write("tok 101\ntok 102\ndone #{bad_count ? 3 : 2} 0 3 101.0 1.0 0 0\n")
    when "config\n" then peer.write("config rev=test dies=6\n")
    when /\Aprof (\d) all\n/
      if fail_profile
        peer.write("error: profile unavailable\n")
      else
        peer.write("prof: 147 phases, 1.00 ms between the first and last timestamp\n  actq256    146 phases    1000.0 us (6.8 us each)\n  longest: #1 actq256 7 us;\n")
      end
    else raise "unexpected mock command: #{line.inspect}"
    end
  end
end

check('startup info connection closes before the next request') do
  with_proxy(handler: method(:normal_engine)) do |proxy, path, received|
    UNIXSocket.open(path) do |s|
      s.write("info\n")
      assert(s.gets == "info 4096 0\n")
    end
    UNIXSocket.open(path) do |s|
      s.write("info\n")
      assert(s.gets == "info 4096 0\n")
    end
    assert(received == %w[info info])
    assert(proxy.errors.empty?)
  end
end

check('pipelined commands retain exact IDs and capture all dies before done') do
  with_proxy(handler: method(:normal_engine)) do |proxy, path, received|
    UNIXSocket.open(path) do |s|
      s.write("stop 200 201\nsample 19 0 32 1\nsnap 2\nchat 39 1 2 3\n")
      lines = []
      lines << s.gets until lines.last&.start_with?('done ')
      assert(lines.first(3) == ["ok\n"] * 3)
      assert(lines[3..4] == ["tok 101\n", "tok 102\n"])
      assert(proxy.records.size == 1)
      capture = proxy.records.first
      assert(capture['token_ids'] == [101, 102])
      assert(capture['profiles'].map { |p| p['die'] } == (0..5).to_a)
      assert(received.last(7) == ['config'] + (0..5).map { |d| "prof #{d} all" })
      assert(capture['native_commands'].last == 'chat 39 1 2 3')
      assert(capture['status'] == 'complete')
    end
    assert(proxy.errors.empty?)
  end
end

check('forbidden resident-control commands never reach the engine') do
  with_proxy(handler: ->(s, received) { received << s.gets }) do |proxy, path, received|
    UNIXSocket.open(path) do |s|
      s.write("quit\n")
      assert(s.gets.nil?)
    end
    assert(proxy.errors.any? { |e| e.include?('refusing') })
    assert(received.compact.empty?)
  end
end

check('frontend cancellation closes upstream during silent prefill') do
  ended = Queue.new
  with_proxy(handler: ->(s, received) { received << s.gets; ended << s.read }) do |proxy, path, received|
    s = UNIXSocket.new(path)
    s.write("chat 39 1 2 3\n")
    sleep 0.01 until received.any?
    canceled_at = PlacementCapture.clock
    s.close
    assert(ended.pop == '')
    assert(PlacementCapture.clock - canceled_at < 1, 'cancellation waited for the native deadline')
    assert(proxy.errors.any? { |e| e.include?('disconnected') })
    assert(proxy.records.empty?)
  end
end

check('native timeout is bounded and retains the incomplete attempt') do
  with_proxy(timeout: 0.15, handler: ->(s, received) { received << s.gets; s.read }) do |proxy, path, received|
    UNIXSocket.open(path) do |s|
      s.write("chat 39 1 2 3\n")
      assert(s.gets.nil?)
    end
    assert(proxy.errors.any? { |e| e.include?('deadline') })
    assert(proxy.attempts.one? && proxy.attempts.first['status'] == 'failed')
  end
end

check('partial profile failure preserves tokens, done, and raw error') do
  with_proxy(handler: ->(s, r) { normal_engine(s, r, fail_profile: true) }) do |proxy, path, _|
    UNIXSocket.open(path) do |s|
      s.write("chat 39 1 2 3\n")
      assert(s.read == "tok 101\ntok 102\n")
    end
    attempt = proxy.attempts.fetch(0)
    assert(proxy.records.empty?)
    assert(attempt['token_ids'] == [101, 102])
    assert(attempt['engine_done'].start_with?('done 2 '))
    assert(attempt['native_responses'].include?("error: profile unavailable\n"))
    assert(attempt['status'] == 'failed')
  end
end

check('native done count disagreement cannot produce a capture') do
  with_proxy(handler: ->(s, r) { normal_engine(s, r, bad_count: true) }) do |proxy, path, _|
    UNIXSocket.open(path) { |s| s.write("chat 39 1 2 3\n"); s.read }
    assert(proxy.records.empty?)
    assert(proxy.errors.include?('native token count mismatch'))
  end
end

check('unknown GPU process prevents idle approval even at zero utilization') do
  original = PlacementCapture.method(:command)
  begin
    PlacementCapture.define_singleton_method(:command) do |*args|
      if args.any? { |x| x.start_with?('--query-gpu=') }
        (0..5).map { |d| "#{d}, GPU-#{d}, 100, 8000, 0, 40, 1177, 3, 16\n" }.join
      elsif args.any? { |x| x.start_with?('--query-compute-apps=') }
        "#{Process.pid}, GPU-0, core, 100\n2147483647, GPU-2, foreign, 100\n"
      else
        'PID PPID STAT COMMAND'
      end
    end
    host = PlacementCapture::Host.new([Process.pid], (0..5).to_a, [])
    snap = host.snapshot
    assert(!snap['idle'])
    assert(snap['foreign_gpu_processes'].size == 1)
  ensure
    PlacementCapture.define_singleton_method(:command, original)
  end
end

check('HTTP JSON errors retain the complete raw response') do
  server = TCPServer.new('127.0.0.1', 0)
  body = JSON.generate('error' => {'message' => 'x' * 1024})
  thread = Thread.new do
    s = server.accept
    while (line = s.gets)
      break if line == "\r\n"
    end
    s.write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: #{body.bytesize}\r\nConnection: close\r\n\r\n#{body}")
    s.close
  end
  begin
    begin
      PlacementCapture.http(server.addr[1], '/health')
      raise 'HTTP error was accepted'
    rescue PlacementCapture::HTTPError => e
      assert(e.response['body'] == body)
      assert(e.response['status'] == '200')
    end
  ensure
    server.close
    thread.join(2) || raise('HTTP fixture did not terminate')
  end
end
puts "1..#{$count}"
