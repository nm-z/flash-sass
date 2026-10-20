#!/usr/bin/env ruby
require 'open3'
require 'shellwords'
require 'socket'
require 'timeout'
require 'tmpdir'

ROOT = File.expand_path('../..', __dir__)
FIXTURE = File.join(__dir__, 'supervisor_test.c')
SUPERVISOR = File.join(ROOT, 'kern', 'vis_supervisor.c')
HEADER = File.join(ROOT, 'kern', 'vis_supervisor.h')

abort 'missing supervisor API source/header' unless File.file?(SUPERVISOR) && File.file?(HEADER)

def compile_fixture(output)
  compiler = Shellwords.split(ENV.fetch('CC', 'cc'))
  command = compiler + [
    '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-I', File.join(ROOT, 'kern'),
    FIXTURE, SUPERVISOR, '-o', output
  ]
  stdout, stderr, status = Open3.capture3(*command)
  return if status.success?
  abort "supervisor fixture compile failed:\n#{command.shelljoin}\n#{stdout}#{stderr}"
end

def wait_for_socket(pid, path, seconds = 5)
  deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + seconds
  loop do
    exited = Process.waitpid(pid, Process::WNOHANG)
    raise "supervisor exited before listening (pid #{pid})" if exited
    return if File.socket?(path)
    raise "supervisor did not create #{path}" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline
    sleep 0.01
  end
end

def wait_for_exit(pid, seconds = 5)
  deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + seconds
  loop do
    result = Process.waitpid2(pid, Process::WNOHANG)
    return result[1] if result
    raise "worker supervisor pid #{pid} did not exit" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline
    sleep 0.01
  end
end

def stop_server(pid)
  Process.kill('TERM', -pid)
  Timeout.timeout(5) { Process.waitpid(pid) }
rescue Errno::ESRCH, Errno::ECHILD
  nil
rescue Timeout::Error
  Process.kill('KILL', -pid) rescue nil
  Process.waitpid(pid) rescue nil
end

def request(path, tag)
  Timeout.timeout(5) do
    UNIXSocket.open(path) do |socket|
      payload = tag.to_s.b
      socket.write("enc #{payload.bytesize}\n")
      socket.write(payload)
      header = socket.gets
      raise "closed response for payload #{tag.inspect}" unless header
      return {kind: :err, header: header} if header.start_with?('err ')

      match = header.match(/\Aok (\d+) (\d+) (\d+) (\d+) (\d+)\n\z/)
      raise "invalid public response header: #{header.inspect}" unless match
      ntok, nx, ny, npos, dim = match.captures.map(&:to_i)
      body = socket.read(ntok * dim * 4 + ntok * 12)
      {kind: :ok, header: header, shape: [ntok, nx, ny, npos, dim], body: body}
    end
  end
end

def valid_worker_pid(response)
  raise "expected complete successful response, got #{response.inspect}" unless response[:kind] == :ok
  raise "unexpected fixture shape: #{response[:shape].inspect}" unless response[:shape] == [1, 1, 1, 1, 2560]
  raise "short fixture response: #{response[:body]&.bytesize.inspect}" unless response[:body]&.bytesize == 2560 * 4 + 12
  response[:body].unpack1('e').to_i
end

Dir.mktmpdir('fs96-supervisor-test-') do |dir|
  binary = File.join(dir, 'supervisor-fixture')
  compile_fixture(binary)

  startup_socket = File.join(dir, 'startup.sock')
  startup_log = File.join(dir, 'startup.log')
  startup_pid = Process.spawn({'FLASHVIS_DEVICE' => 'cpu'}, binary,
                              'projector-placeholder', 'text-placeholder', startup_socket,
                              err: startup_log, pgroup: true)
  startup_status = wait_for_exit(startup_pid)
  raise "GPU-off startup exit #{startup_status.exitstatus.inspect}, expected 2" unless startup_status.exitstatus == 2
  raise 'startup failure left an external socket behind' if File.exist?(startup_socket)

  socket = File.join(dir, 'supervisor.sock')
  log = File.join(dir, 'worker.log')
  pid = Process.spawn(binary, 'projector-placeholder', 'text-placeholder', socket,
                      err: log, pgroup: true)
  begin
    wait_for_socket(pid, socket)
    previous_pid = valid_worker_pid(request(socket, 'V'))

    {
      'A' => ['abort mid-dispatch', "err GPU encoder worker terminated\n"],
      'P' => ['partial body then exit', "err GPU encoder worker terminated\n"],
      'I' => ['invalid header', "err GPU encoder worker terminated\n"],
      'E' => ['in-band error before worker exit', "err fixture reported encode failure\n"]
    }.each do |tag, (label, expected_error)|
      response = request(socket, tag)
      raise "#{label} did not produce its expected explicit error: #{response.inspect}" unless
        response[:kind] == :err && response[:header] == expected_error
      fresh_pid = valid_worker_pid(request(socket, 'V'))
      raise "#{label} reused worker pid #{fresh_pid}" if fresh_pid == previous_pid
      previous_pid = fresh_pid
    end
  ensure
    stop_server(pid)
  end

  log_text = File.file?(log) ? File.read(log) : ''
  {'A' => 'abort', 'P' => 'partial-body', 'I' => 'invalid-header', 'E' => 'reported-error'}.each do |tag, label|
    count = log_text.scan(/fixture-request pid=\d+ tag=#{tag}\n/).length
    raise "#{label} image was dispatched #{count} times; expected exactly once" unless count == 1
  end
  puts 'CPU-only supervisor IPC regression passed: GPU-off exit propagation, fatal EOF, partial body, invalid header, err forwarding and retirement, fresh worker, no request replay.'
end
