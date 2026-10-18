#!/usr/bin/env ruby
# Capture the linkmatrix request protocol without changing resident processes or links.
require 'digest'
require 'fileutils'
require 'json'
require 'net/http'
require 'open3'
require 'optparse'
require 'socket'
require 'tmpdir'
require 'time'

module PlacementCapture
  LOCK = '/tmp/flash-sass-engine-validation.lock'
  REQUEST = {
    'messages' => [{'role' => 'user', 'content' => 'What is the capital of France? Answer in one word.'}],
    'max_tokens' => 40, 'temperature' => 0
  }.freeze
  GPU_FIELDS = %w[index uuid memory.used memory.total utilization.gpu temperature.gpu clocks.sm pcie.link.gen.current pcie.link.width.current].freeze
  class Error < StandardError; end
  class HTTPError < Error
    attr_reader :response
    def initialize(message, code, body)
      super(message)
      @response = {'status' => code, 'body' => body}
    end
  end

  def self.command(*args)
    out, err, status = Open3.capture3(*args)
    raise Error, "#{args.first} failed (#{status.exitstatus}): #{err.strip}" unless status.success?
    out
  end

  def self.clock
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  # Bounded line framing; watching the downstream socket makes cancellation close
  # the real engine connection too. Never send quit/reload/cubin to the resident.
  class Lines
    def initialize(io)
      @io, @buffer = io, +''
    end
    def read(deadline, watch: nil)
      loop do
        return @buffer.slice!(0, @buffer.index("\n") + 1) if @buffer.include?("\n")
        raise Error, 'native line exceeds 2 MiB' if @buffer.bytesize > 2 * 1024 * 1024
        left = deadline - PlacementCapture.clock
        raise Error, 'native socket deadline expired' unless left.positive?
        ready = IO.select([@io, watch].compact, nil, nil, left)
        raise Error, 'native socket deadline expired' unless ready
        if watch && ready[0].include?(watch)
          peek = watch.recv_nonblock(1, Socket::MSG_PEEK, exception: false)
          raise Error, 'temporary frontend disconnected' if peek.nil? || peek == ''
        end
        unless ready[0].include?(@io)
          sleep 0.001
          next
        end
        chunk = @io.read_nonblock(16384, exception: false)
        next if chunk == :wait_readable
        if chunk.nil?
          raise Error, 'native socket ended inside a line' unless @buffer.empty?
          return nil
        end
        @buffer << chunk
      end
    end
  end

  class Proxy
    attr_reader :records, :errors, :attempts
    def initialize(path, upstream_path, timeout: 120)
      @server = UNIXServer.new(path)
      @upstream_path, @timeout = upstream_path, timeout
      @records, @errors, @attempts, @sockets = [], [], [], []
      @guard = Mutex.new
      @stopping = false
      @thread = Thread.new do
        begin
          until @stopping
            client = @server.accept
            serve(client)
          end
        rescue IOError, Errno::EBADF
          raise unless @stopping
        rescue StandardError => e
          @errors << e.message unless @stopping
        end
      end
    end
    def close
      @guard.synchronize do
        @stopping = true
        @server.close unless @server.closed?
        @sockets.each { |s| s.close unless s.closed? }
      end
      @thread.join(3) || raise(Error, 'proxy did not stop after closing its sockets')
    end
    def register(socket)
      @guard.synchronize do
        if @stopping
          socket.close
          raise Error, 'proxy is stopping'
        end
        @sockets << socket
      end
    end
    def required(reader, deadline, client, record)
      line = reader.read(deadline, watch: client) || raise(Error, 'engine disconnected')
      record['native_responses'] << line
      line
    end
    def serve(client)
      deadline = PlacementCapture.clock + @timeout
      register(client)
      upstream = Socket.new(Socket::AF_UNIX, Socket::SOCK_STREAM, 0)
      register(upstream)
      address = Socket.sockaddr_un(@upstream_path)
      loop do
        begin
          break if upstream.connect_nonblock(address, exception: false) == 0
        rescue Errno::EISCONN
          break
        rescue Errno::EALREADY
          # The registered socket remains cancellable while connect is pending.
        end
        left = deadline - PlacementCapture.clock
        raise Error, 'engine connect deadline expired' unless left.positive?
        IO.select(nil, [upstream], nil, [left, 0.05].min)
      end
      input, replies = Lines.new(client), Lines.new(upstream)
      record = {'native_commands' => [], 'native_responses' => [], 'token_ids' => [], 'profiles' => [], 'status' => 'incomplete'}
      saw_chat = false
      while (line = input.read(deadline))
        raise Error, 'proxy is stopping' if @stopping
        raise Error, 'command after chat is not supported' if saw_chat
        unless line.match?(/\A(?:info|stop(?: \d+){0,2}|sample \d+ [-+\d.e]+ \d+ [-+\d.e]+|snap \d+|chat \d+(?: \d+)+)\n\z/)
          raise Error, 'refusing a command outside the benchmark protocol'
        end
        record['native_commands'] << line.chomp
        upstream.write(line)
        unless line.start_with?('chat ')
          answer = required(replies, deadline, client, record)
          raise Error, answer.strip if answer.start_with?('error')
          client.write(answer)
          next
        end
        saw_chat = true
        @attempts << record
        loop do
          answer = required(replies, deadline, client, record)
          if (m = /\Atok (\d+)\n\z/.match(answer))
            record['token_ids'] << Integer(m[1])
            raise Error, 'engine exceeded the 40-token request' if record['token_ids'].length > 40
            client.write(answer)
          elsif answer.start_with?('done ')
            unless answer.match?(/\Adone \d+ [01] \d+ [\d.]+ [\d.]+ \d+ \d+\n\z/)
              raise Error, 'malformed native done line'
            end
            raise Error, 'native token count mismatch' unless answer.split[1].to_i == record['token_ids'].length
            record['engine_done'] = answer.chomp
            upstream.write("config\n")
            record['config'] = required(replies, deadline, client, record).strip
            raise Error, 'engine lacks a config record' unless record['config'].start_with?('config ')
            6.times do |die|
              upstream.write("prof #{die} all\n")
              profile = +''
              loop do
                row = required(replies, deadline, client, record)
                raise Error, row.strip if row.start_with?('error')
                profile << row
                raise Error, 'profile exceeds 64 KiB' if profile.bytesize > 65536
                break if row.start_with?('  longest:')
              end
              raise Error, 'malformed profile header' unless profile.start_with?('prof: ')
              record['profiles'] << {'die' => die, 'text' => profile}
            end
            record['status'] = 'complete'
            @records << record
            client.write(answer) # native times were fixed before profiling
            break
          else
            raise Error, "unexpected native response: #{answer.strip}"
          end
        end
      end
    rescue StandardError => e
      record&.merge!('status' => 'failed', 'error' => e.message)
      @errors << e.message unless @stopping
    ensure
      client&.close unless client&.closed?
      upstream&.close unless upstream&.closed?
      @guard.synchronize { @sockets = [] }
    end
  end

  class Host
    attr_reader :pids, :identities
    def initialize(pids, devices, ports)
      @pids, @devices, @ports = pids, devices, ports
      @identities = pids.to_h { |pid| [pid, process_identity(pid)] }
    end
    def process_identity(pid)
      # starttime distinguishes a PID that was recycled during a long wait.
      stat = File.read("/proc/#{pid}/stat").sub(/\A.*\) /, '').split
      {'pid' => pid, 'starttime_ticks' => stat.fetch(19), 'comm' => File.read("/proc/#{pid}/comm").strip}
    end
    def snapshot(check_residents: true)
      identity_errors = []
      @identities.each do |pid, expected|
        begin
          raise Error, "resident process #{pid} changed" unless process_identity(pid) == expected
        rescue StandardError => e
          raise if check_residents
          identity_errors << e.message
        end
      end
      raw = PlacementCapture.command('nvidia-smi', "--query-gpu=#{GPU_FIELDS.join(',')}", '--format=csv,noheader,nounits')
      # These requested GPU fields contain no commas. For processes only the
      # leading PID and UUID fields are interpreted; retain the whole raw row.
      rows = raw.lines.map { |line| line.strip.split(',').map(&:strip) }
      selected = @devices.map do |index|
        row = rows.find { |r| r[0] == index.to_s } || raise(Error, "missing GPU #{index}")
        raise Error, 'malformed GPU telemetry' unless row.size == GPU_FIELDS.size && row[2..].all? { |v| v.match?(/\A\d+(?:\.\d+)?\z/) }
        GPU_FIELDS.zip(row).to_h
      end
      apps_raw = PlacementCapture.command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory', '--format=csv,noheader,nounits')
      apps = apps_raw.lines.map { |line| line.strip.split(',').map(&:strip) }
      ps_pids = (apps.map { |a| Integer(a[0], 10) } + [Process.pid]).uniq.join(',')
      ps = PlacementCapture.command('ps', '-p', ps_pids, '-o', 'pid,ppid,stat,comm')
      links = @ports.map do |port|
        base = "/sys/bus/pci/devices/#{port}"
        {'pci_address' => port, 'speed' => File.read("#{base}/current_link_speed").strip,
         'width' => File.read("#{base}/current_link_width").strip}
      end
      uuids = selected.map { |g| g.fetch('uuid') }
      foreign = apps.select { |a| uuids.include?(a[1]) && !@pids.include?(Integer(a[0], 10)) }
      {'utc' => Time.now.utc.iso8601(6), 'gpus' => selected, 'gpu_processes_csv' => apps_raw,
       'ps' => ps, 'root_links' => links, 'foreign_gpu_processes' => foreign, 'resident_identity_errors' => identity_errors,
       'idle' => (!check_residents || foreign.empty?) && selected.all? { |g| g.fetch('utilization.gpu').to_f.zero? }}
    end
    def wait_idle(log, retry_errors: false, check_residents: true)
      consecutive = 0
      loop do
        begin
          snap = snapshot(check_residents: check_residents)
        rescue StandardError => e
          raise unless retry_errors
          log << {'utc' => Time.now.utc.iso8601(6), 'error' => e.message}
          warn "placement capture: retaining lock; cannot confirm idle: #{e.message}"
          consecutive = 0
          sleep 5
          next
        end
        log << snap
        consecutive = snap['idle'] ? consecutive + 1 : 0
        return snap if consecutive >= 2
        warn 'placement capture: waiting for GPUs to be idle' unless snap['idle']
        sleep(snap['idle'] ? 0.3 : 5)
      end
    end
  end

  def self.stop_child(pid)
    return unless pid
    Process.kill('TERM', pid)
    deadline = clock + 5
    loop do
      return if Process.waitpid(pid, Process::WNOHANG)
      break if clock >= deadline
      sleep 0.05
    end
    # This child has not been reaped, so its PID cannot have been reused.
    Process.kill('KILL', pid)
    Process.waitpid(pid)
  rescue Errno::ESRCH, Errno::ECHILD
    # Already exited; never signal a resident process.
  end

  def self.http(port, path, body = nil)
    client = Net::HTTP.new('127.0.0.1', port, nil)
    client.open_timeout = 3
    client.read_timeout = 150
    client.write_timeout = 5
    request = body ? Net::HTTP::Post.new(path, {'Content-Type' => 'application/json'}) : Net::HTTP::Get.new(path)
    request.body = JSON.generate(body) if body
    response = client.start { |c| c.request(request) }
    raise HTTPError.new("HTTP #{response.code}", response.code, response.body) unless response.code == '200'
    begin
      parsed = JSON.parse(response.body)
    rescue JSON::ParserError
      raise HTTPError.new('HTTP response is not JSON', response.code, response.body)
    end
    raise HTTPError.new('HTTP JSON error response', response.code, response.body) if !parsed.is_a?(Hash) || parsed.key?('error')
    parsed
  end

  def self.main(argv)
    opts = {socket: '/tmp/flash.sock', devices: (0..5).to_a,
            ports: %w[0000:00:01.0 0000:00:02.0 0000:00:03.0], n: 5, rounds: 2}
    parser = OptionParser.new do |p|
      p.banner = 'Usage: ruby kern/placement_capture.rb --flashd EXE --model GGUF --resident-pids PIDS --output JSON --label NAME'
      p.on('--flashd PATH') { |v| opts[:flashd] = File.expand_path(v) }
      p.on('--model PATH') { |v| opts[:model] = File.realpath(v) }
      p.on('--socket PATH') { |v| opts[:socket] = v }
      p.on('--resident-pids LIST', 'Explicit existing core/hold/fence GPU PID allowlist') { |v| opts[:pids] = v.split(',').map { |x| Integer(x, 10) }.uniq }
      p.on('--devices LIST', 'Six NVML indices in engine die order (default 0,1,2,3,4,5)') { |v| opts[:devices] = v.split(',').map { |x| Integer(x, 10) } }
      p.on('--root-ports LIST', 'PCI root ports in card order (default 0000:00:01.0,0000:00:02.0,0000:00:03.0)') { |v| opts[:ports] = v.split(',') }
      p.on('--output PATH') { |v| opts[:output] = v }
      p.on('--label NAME') { |v| opts[:label] = v }
      p.on('-n N', Integer, 'Measured requests per round (default 5)') { |v| opts[:n] = v }
      p.on('-r N', Integer, 'Rounds (default 2)') { |v| opts[:rounds] = v }
      p.on('-h', '--help') { puts p; return 0 }
    end
    parser.parse!(argv)
    raise Error, 'unexpected positional arguments' unless argv.empty?
    %i[flashd model pids output label].each { |key| raise Error, "missing --#{key.to_s.tr('_','-')}" unless opts[key] }
    raise Error, 'need six distinct nonnegative device indices' unless opts[:devices].size == 6 && opts[:devices].uniq.size == 6 && opts[:devices].all? { |x| x >= 0 }
    raise Error, 'resident PIDs must be positive' unless !opts[:pids].empty? && opts[:pids].all?(&:positive?)
    raise Error, 'need three distinct PCI root-port addresses' unless opts[:ports].uniq.size == 3 && opts[:ports].all? { |v| v.match?(/\A[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\z/) }
    raise Error, 'request and round counts must be positive' unless opts[:n].positive? && opts[:rounds].positive?
    raise Error, 'frontend is not executable' unless File.file?(opts[:flashd]) && File.executable?(opts[:flashd])
    raise Error, 'model is not a readable regular file' unless File.file?(opts[:model]) && File.readable?(opts[:model])
    raise Error, 'engine socket is absent' unless File.socket?(opts[:socket])
    record = {'schema' => 'flash-sass-placement-v1', 'status' => 'incomplete', 'label' => opts[:label],
              'protocol' => {'request' => REQUEST, 'requests_per_round' => opts[:n], 'rounds' => opts[:rounds], 'warmups_per_round' => 1},
              'samples' => [], 'failures' => [], 'preflight' => [], 'postflight' => []}
    output = File.open(opts[:output], File::WRONLY | File::CREAT | File::EXCL, 0644)
    lock = proxy = child = workdir = host = monitor = nil
    monitoring = false
    begin
      lock = File.open(LOCK, File::RDWR | File::CREAT, 0600)
      until lock.flock(File::LOCK_EX | File::LOCK_NB)
        warn "placement capture: waiting for #{LOCK}"
        sleep 5
      end
      host = Host.new(opts[:pids], opts[:devices], opts[:ports])
      before = host.wait_idle(record['preflight'])
      model_stat = File.stat(opts[:model])
      frontend_env = ENV.to_h.select { |k, _| k.start_with?('FLASHD_') || k == 'FLASHVIS_SOCK' }
      record['identity'] = {
        'hostname' => Socket.gethostname, 'boot_id' => File.read('/proc/sys/kernel/random/boot_id').strip,
        'frontend_sha256' => Digest::SHA256.file(opts[:flashd]).hexdigest,
        'frontend_environment' => frontend_env, 'collector_sha256' => Digest::SHA256.file(__FILE__).hexdigest,
        'model' => {'path' => opts[:model], 'size' => model_stat.size, 'mtime' => model_stat.mtime.utc.iso8601(9)},
        'device_uuids' => before.fetch('gpus').map { |g| g.fetch('uuid') },
        'device_indices' => opts[:devices], 'resident_processes' => host.identities.values
      }
      record['hardware_before'] = {
        'nvidia_smi_q' => command('nvidia-smi', '-q', '-i', opts[:devices].join(',')),
        'pci_tree' => command('lspci', '-D', '-t')
      }
      workdir = Dir.mktmpdir('flash-sass-placement-')
      proxy = Proxy.new(File.join(workdir, 'capture.sock'), opts[:socket])
      reservation = TCPServer.new('127.0.0.1', 0)
      port = reservation.addr[1]
      reservation.close
      log_path = File.join(workdir, 'flashd.log')
      child = Process.spawn(opts[:flashd], opts[:model], File.join(workdir, 'capture.sock'), "127.0.0.1:#{port}", out: log_path, err: log_path)
      deadline = clock + 30
      loop do
        if Process.waitpid(child, Process::WNOHANG)
          child = nil
          raise Error, "temporary frontend exited during startup: #{File.read(log_path)}"
        end
        # Its own log confirms bind/listen, avoiding mistaking another listener for our child.
        break if File.read(log_path).include?("listening on 127.0.0.1:#{port}")
        raise Error, 'temporary frontend startup timed out' if clock >= deadline
        raise Error, proxy.errors.join('; ') unless proxy.errors.empty?
        sleep 0.1
      end
      record['identity']['frontend_props'] = http(port, '/props')
      record['telemetry'] = [before]
      monitoring = true
      monitor = Thread.new do
        while monitoring
          begin
            observation = host.snapshot
            record['telemetry'] << observation
            raise Error, 'foreign GPU work appeared during capture' unless observation['foreign_gpu_processes'].empty?
          rescue StandardError => e
            record['failures'] << "monitor: #{e.message}"
            break
          end
          sleep 0.2
        end
      end
      opts[:rounds].times do |r|
        (0..opts[:n]).each do |i|
          if Process.waitpid(child, Process::WNOHANG)
            child = nil
            raise Error, 'temporary frontend exited'
          end
          raise Error, record['failures'].join('; ') unless record['failures'].empty?
          sample_before = record['telemetry'].last
          count = proxy.records.length
          t0 = clock
          response = http(port, '/v1/chat/completions', REQUEST)
          wall_ms = (clock - t0) * 1000
          raise Error, proxy.errors.join('; ') unless proxy.errors.empty?
          raise Error, 'missing or duplicate native capture' unless proxy.records.length == count + 1
          sample = proxy.records.last.merge(
            'round' => r + 1, 'request' => i, 'warmup' => i.zero?, 'response' => response, 'http_wall_ms' => wall_ms,
            'host_before' => sample_before, 'host_after' => record['telemetry'].last)
          record['samples'] << sample
          record['identity']['engine_config'] ||= sample.fetch('config')
          raise Error, 'engine configuration changed during capture' unless record['identity']['engine_config'] == sample.fetch('config')
          warn "placement capture: round #{r + 1} #{i.zero? ? 'warmup' : "request #{i}"} captured"
        end
      end
      monitoring = false
      monitor.join
      raise Error, record['failures'].join('; ') unless record['failures'].empty?
      record['status'] = 'complete'
    rescue StandardError, Interrupt => e
      record['status'] = 'failed'
      record['failures'] << "#{e.class}: #{e.message}"
      record['failed_http'] = e.response if e.is_a?(HTTPError)
      warn record['failures'].last
    ensure
      # Closing the proxy closes our upstream connection and cancels our request,
      # including during prefill. The resident process is never signaled.
      monitoring = false
      cleanup = [-> { proxy&.close }, -> { stop_child(child) }, -> { monitor&.join }, -> {
        if workdir
          record['frontend_log'] = File.read(File.join(workdir, 'flashd.log'))
          FileUtils.remove_entry(workdir)
        end
      }]
      cleanup.each do |action|
        begin
          action.call
        rescue StandardError => e
          record['status'] = 'failed'
          record['failures'] << "cleanup: #{e.message}"
        end
      end
      record['native_attempts'] = proxy.attempts if proxy
      begin
        # A telemetry failure is not proof that cancellation finished. Keep the
        # shared lock while waiting for monitoring and the idle state to recover.
        host&.wait_idle(record['postflight'], retry_errors: true, check_residents: false)
        changed = record['postflight'].flat_map { |s| s.fetch('resident_identity_errors', []) }.uniq
        unless changed.empty?
          record['status'] = 'failed'
          record['failures'].concat(changed)
        end
      ensure
        begin
          output.write(JSON.pretty_generate(record) + "\n")
        ensure
          lock&.flock(File::LOCK_UN)
          lock&.close
          output.close
        end
      end
    end
    record['status'] == 'complete' ? 0 : 1
  rescue OptionParser::ParseError, ArgumentError, SystemCallError, Error => e
    warn "placement capture: #{e.message}"
    2
  end
end

exit PlacementCapture.main(ARGV) if $PROGRAM_NAME == __FILE__
