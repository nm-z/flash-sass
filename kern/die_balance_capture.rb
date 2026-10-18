#!/usr/bin/env ruby
# frozen_string_literal: true
# One warmup + matched linkmatrix requests. Never reloads the resident engine.
require "csv"
require "digest"
require "fileutils"
require "json"
require "net/http"
require "optparse"
require "socket"
require "time"
require "timeout"
require "tmpdir"
require_relative "die_balance"

module DieBalanceCapture
  REQUEST = DieBalance::REQUEST
  IO_TIMEOUT = 120
  MAX_LINE = 4 * 1024 * 1024
  FRONTEND_ENV = %w[FLASHD_NAME FLASHD_THINKING FLASHD_QUEUE_MAX FLASHD_QUEUE_WAIT_S FLASHD_IMGDIR FLASHD_GREEDY FLASHVIS_SOCK].freeze
  ROOT_PORTS = %w[00:01.0 00:02.0 00:03.0].freeze
  class CaptureError < StandardError; end
  module_function

  def monotonic
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  def line(io, eof_ok: false)
    value = Timeout.timeout(IO_TIMEOUT) { io.gets(MAX_LINE) }
    return nil if !value && eof_ok
    raise CaptureError, "unexpected protocol EOF" unless value
    raise CaptureError, "unterminated/oversized protocol line" unless value.end_with?("\n")
    raise CaptureError, "NUL in protocol line" if value.include?("\0")
    value
  end

  def write(io, value)
    Timeout.timeout(IO_TIMEOUT) { io.write(value); io.flush }
  end

  def connect_core(path, pid)
    io = UNIXSocket.new(path)
    peer = io.getsockopt(Socket::SOL_SOCKET, Socket::SO_PEERCRED).unpack("i3").first
    raise CaptureError, "core peer PID #{peer} differs from requested PID #{pid}" unless peer == pid
    io
  rescue StandardError
    io&.close
    raise
  end

  def read_config(io)
    write(io, "config\n")
    raw = line(io)
    DieBalance.config!(raw)
    raw
  end

  def query_config(path, pid)
    io = connect_core(path, pid)
    read_config(io)
  ensure
    io&.close
  end

  def process_identity(pid)
    base = "/proc/#{pid}"
    stat = File.read("#{base}/stat")
    suffix = stat[(stat.rindex(") ") || raise(CaptureError, "bad proc stat")) + 2..]
    {"pid" => pid, "start_ticks" => suffix.split.fetch(19),
     "exe" => File.readlink("#{base}/exe"), "exe_sha256" => Digest::SHA256.file("#{base}/exe").hexdigest,
     "cwd" => File.readlink("#{base}/cwd"), "argv" => File.binread("#{base}/cmdline").split("\0"),
     "comm" => File.read("#{base}/comm").strip}
  end

  def engine_identity(pid)
    rows = File.readlines("/proc/#{pid}/maps").map { |row| row.strip.split(/\s+/, 6) }
    rows = rows.select { |row| row[1].include?("x") && row[5]&.match?(%r{/libengine[^/]*\.so(?: \(deleted\))?\z}) }
    raise CaptureError, "expected one identifiable mapped libengine" unless rows.length == 1
    row = rows.first
    path = row[5]
    raise CaptureError, "mapped engine file was replaced/deleted" if path.end_with?(" (deleted)")
    stat = File.stat(path)
    raise CaptureError, "mapped engine inode differs from file" unless stat.ino == row[4].to_i
    {"path" => path, "inode" => stat.ino, "size" => stat.size,
     "mtime_ns" => (stat.mtime.to_r * 1_000_000_000).to_i, "sha256" => Digest::SHA256.file(path).hexdigest}
  end

  def model_identity(path)
    real = File.realpath(path)
    stat = File.stat(real)
    raise CaptureError, "GGUF is not a regular file" unless stat.file?
    {"identity_kind" => "stat", "path" => path, "realpath" => real, "device" => stat.dev,
     "inode" => stat.ino, "size" => stat.size, "mtime_ns" => (stat.mtime.to_r * 1_000_000_000).to_i}
  end

  # Only children spawned by this collector may be terminated.
  def stop_owned_child(pid)
    return unless pid
    return if Process.waitpid(pid, Process::WNOHANG)
    Process.kill("TERM", pid)
    deadline = monotonic + 3
    loop do
      return if Process.waitpid(pid, Process::WNOHANG)
      break if monotonic >= deadline
      sleep 0.05
    end
    Process.kill("KILL", pid)
    Process.waitpid(pid)
  rescue Errno::ESRCH, Errno::ECHILD
    nil
  end

  def command(argv, out, name, seconds: 30, chdir: nil)
    options = {out: File.join(out, "#{name}.stdout"), err: File.join(out, "#{name}.stderr"), in: File::NULL}
    options[:chdir] = chdir if chdir
    File.write(File.join(out, "#{name}.command.json"), JSON.generate(argv) + "\n")
    pid = Process.spawn(*argv, **options)
    deadline = monotonic + seconds
    loop do
      waited = Process.waitpid2(pid, Process::WNOHANG)
      if waited
        pid = nil
        File.write(File.join(out, "#{name}.exit"), "#{waited.last.exitstatus}\n")
        raise CaptureError, "command failed: #{argv.inspect}" unless waited.last.success?
        return File.read(options[:out])
      end
      raise CaptureError, "command timed out: #{argv.first}" if monotonic >= deadline
      sleep 0.05
    end
  ensure
    stop_owned_child(pid) if pid
  end

  def hardware_snapshot(out, prefix)
    identity = command(["nvidia-smi", "-L"], out, "#{prefix}-gpu-identity")
    command(["nvidia-smi", "-q"], out, "#{prefix}-gpu-telemetry")
    links = ROOT_PORTS.map do |port|
      name = "#{prefix}-pcie-#{port.tr(':.', '--')}"
      argv = ["lspci", "-vv", "-s", port]
      raw = command(argv, out, name)
      if !raw.include?("LnkSta:")
        raw = command(["sudo", "-n", *argv], out, "#{name}-privileged")
      end
      match = /LnkSta:\s+Speed\s+([0-9.]+GT\/s)(?:\s+\([^)]*\))?,\s+Width\s+(x\d+)/.match(raw)
      raise CaptureError, "cannot read root-port speed/width for #{port}" unless match
      "#{port} Speed #{match[1]} Width #{match[2]}"
    end
    {"identity" => identity, "pci_link_snapshot" => links.join("\n") + "\n"}
  end

  def idle_preflight(out, core_pid, socket)
    command(["ps", "-eo", "pid,ppid,lstart,stat,args"], out, "preflight-processes")
    metrics = command(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"], out, "preflight-gpu-load")
    rows = CSV.parse(metrics).map { |row| row.map(&:strip) }
    selected = rows.select { |row| (0...6).include?(Integer(row[0], 10)) }
    unless selected.map { |row| Integer(row[0], 10) }.sort == (0...6).to_a &&
           selected.all? { |row| Integer(row[2], 10).zero? }
      raise CaptureError, "dies 0-5 are absent or busy; wait for an idle host"
    end
    uuids = selected.map { |row| row[1] }
    apps = command(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader,nounits"], out, "preflight-compute-apps")
    CSV.parse(apps).each do |row|
      pid, _name, uuid = row.map(&:strip)
      next unless uuids.include?(uuid)
      pid = Integer(pid, 10)
      comm = File.read("/proc/#{pid}/comm").strip
      next if pid == core_pid || %w[hold fence].include?(comm)
      raise CaptureError, "unrelated compute process #{pid} (#{comm}) occupies a participating die"
    end
    sockets = command(["ss", "-xnp"], out, "preflight-sockets")
    if sockets.lines.any? { |row| row.include?(socket) && row.match?(/\bESTAB\b/) }
      raise CaptureError, "resident engine has an established client; wait until it disconnects"
    end
  end

  def owned_listener?(pid, port)
    sockets = Dir.glob("/proc/#{pid}/fd/*").filter_map do |fd|
      File.readlink(fd)[/\Asocket:\[(\d+)\]\z/, 1]
    rescue Errno::ENOENT, Errno::EACCES
      nil
    end
    address = "0100007F:#{format('%04X', port)}"
    File.readlines("/proc/net/tcp").drop(1).any? do |row|
      fields = row.split
      fields[1] == address && fields[3] == "0A" && sockets.include?(fields[9])
    end
  rescue Errno::ENOENT
    false
  end

  class Relay
    def initialize(path, upstream, core_pid, log_path)
      @path, @upstream, @core_pid = path, upstream, core_pid
      @results = Queue.new
      @log = File.open(log_path, "wx")
      @log.sync = true
      @server = UNIXServer.new(path)
      File.chmod(0o600, path)
      @stopping = false
      @thread = Thread.new { serve }
      @thread.report_on_exception = false
    end

    def check!
      raise @failure if @failure
    end

    def pop
      check!
      value = Timeout.timeout(IO_TIMEOUT) { @results.pop }
      raise value if value.is_a?(Exception)
      value
    end

    def close
      @stopping = true
      [@server, @client, @core].each do |io|
        io&.close unless io&.closed?
      rescue IOError, SystemCallError
        nil
      end
      @thread.join(3)
      raise CaptureError, "relay did not stop after sockets closed" if @thread.alive?
    ensure
      @log&.close unless @log&.closed?
      File.unlink(@path) if File.socket?(@path)
    end

    private

    def log(direction, raw)
      @log.puts(JSON.generate("at" => Time.now.utc.iso8601(6), "direction" => direction, "raw" => raw))
    end

    def send_core(raw)
      log("frontend_or_collector_to_core", raw)
      DieBalanceCapture.write(@core, raw)
    end

    def from_core
      raw = DieBalanceCapture.line(@core)
      log("core_to_relay", raw)
      raw
    end

    def serve
      until @stopping
        @client = @server.accept
        @core = DieBalanceCapture.connect_core(@upstream, @core_pid)
        session
        @client.close
        @core.close
        @client = @core = nil
      end
    rescue StandardError => error
      unless @stopping
        @failure = error
        @results << error
      end
    ensure
      [@client, @core].each do |io|
        io&.close unless io&.closed?
      rescue IOError, SystemCallError
        nil
      end
    end

    def profile(die)
      send_core("prof #{die} all\n")
      raw = from_core
      raise CaptureError, "invalid profile header" unless raw.start_with?("prof: ")
      loop do
        part = from_core
        raw << part
        raise CaptureError, "oversized profile" if raw.bytesize > MAX_LINE
        break if part.start_with?("  longest:")
        raise CaptureError, "unexpected profile record" unless part.match?(/\A  \S+\s+\d+ phases /)
      end
      entry = {"die" => die, "raw" => raw}
      DieBalance.profile!(entry)
      entry
    end

    # Consume one whole command/response at a time. flashd pipelines commands,
    # so IO.select must not be mixed with gets' hidden input buffering.
    def session
      while (raw = DieBalanceCapture.line(@client, eof_ok: true))
        parts = raw.strip.split
        name = parts.first
        expected = case name
                   when "info" then /\Ainfo \d+ \d+\n\z/ if parts.length == 1
                   when "config" then /\Aconfig rev=/ if parts.length == 1
                   when "stop"
                     /\Aok: stop / if parts.length == 3 && parts.drop(1).all? { |v| v.match?(/\A\d+\z/) }
                   when "sample"
                     /\Aok: sample / if parts.length == 5 && parts[1].match?(/\A\d+\z/) &&
                       parts.drop(2).all? { |v| v.match?(/\A#{DieBalance::FLOAT}\z/) } && Float(parts[2]).zero?
                   when "snap" then /\Aok: snap / if parts.length == 2 && parts[1].match?(/\A\d+\z/)
                   when "chat"
                     :chat if parts.length >= 3 && parts.drop(1).all? { |v| v.match?(/\A\d+\z/) } &&
                       Integer(parts[1], 10).between?(1, 39)
                   end
        raise CaptureError, "refusing frontend command #{name.inspect}" unless expected
        send_core(raw)
        if expected != :chat
          reply = from_core
          raise CaptureError, "unexpected #{name} response: #{reply.inspect}" unless expected.match?(reply)
          DieBalanceCapture.write(@client, reply)
          next
        end
        evidence = {"input_tokens" => parts.drop(2).map { |v| Integer(v, 10) }, "output_tokens" => []}
        loop do
          reply = from_core
          if (match = /\Atok (\d+)\n\z/.match(reply))
            evidence["output_tokens"] << Integer(match[1], 10)
            raise CaptureError, "too many output tokens" if evidence["output_tokens"].length > 40
          elsif reply.start_with?("done ")
            fields = reply.split
            unless fields.length == 8 && [1, 2, 3, 6, 7].all? { |i| fields[i].match?(/\A\d+\z/) } &&
                   [4, 5].all? { |i| fields[i].match?(/\A#{DieBalance::FLOAT}\z/) } &&
                   fields[1].to_i == evidence["output_tokens"].length && fields[3].to_i == evidence["input_tokens"].length
              raise CaptureError, "malformed or incomplete done record"
            end
            evidence["done"] = reply
            send_core("config\n")
            evidence["config"] = from_core
            DieBalance.config!(evidence["config"])
            evidence["profiles"] = (0...6).map { |die| profile(die) }
            @results << evidence
            DieBalanceCapture.write(@client, reply)
            break
          else
            raise CaptureError, "unexpected chat response: #{reply.inspect}"
          end
          DieBalanceCapture.write(@client, reply)
        end
      end
    end
  end

  def http_request(port, out, index)
    http = Net::HTTP.new("127.0.0.1", port, nil)
    http.open_timeout, http.read_timeout, http.write_timeout = 5, IO_TIMEOUT, 10
    request = Net::HTTP::Post.new("/v1/chat/completions")
    request["Content-Type"] = "application/json"
    request.body = JSON.generate(REQUEST)
    response = Timeout.timeout(IO_TIMEOUT) { http.request(request) }
    File.write(File.join(out, format("http-%02d.json", index)), response.body)
    raise CaptureError, "HTTP request failed: #{response.code}" unless response.is_a?(Net::HTTPSuccess)
    body = JSON.parse(response.body)
    raise CaptureError, "invalid HTTP JSON" unless body.is_a?(Hash) && !body.key?("error") && body["choices"].is_a?(Array)
    body
  end

  def run(argv)
    options = {socket: "/tmp/flash.sock", requests: 3, lock: "/tmp/flash-sass-engine-validation.lock"}
    parser = OptionParser.new do |p|
      p.banner = "Usage: ruby kern/die_balance_capture.rb --out FRESH_DIR --label LABEL --core-pid PID --flashd-pid PID"
      p.on("--out PATH") { |v| options[:out] = v }
      p.on("--label LABEL") { |v| options[:label] = v }
      p.on("--core-pid PID", Integer) { |v| options[:core_pid] = v }
      p.on("--flashd-pid PID", Integer) { |v| options[:flashd_pid] = v }
      p.on("--socket PATH") { |v| options[:socket] = v }
      p.on("--requests N", Integer) { |v| options[:requests] = v }
      p.on("--lock PATH") { |v| options[:lock] = v }
      p.on("-h", "--help") { puts p; return 0 }
    end
    parser.parse!(argv)
    raise CaptureError, "unexpected positional arguments" unless argv.empty?
    %i[out label core_pid flashd_pid].each { |key| raise CaptureError, "missing --#{key.to_s.tr('_', '-')}" unless options[key] }
    raise CaptureError, "label must not be empty" if options[:label].strip.empty?
    raise CaptureError, "at least three requests are required" unless options[:requests] >= 3
    raise CaptureError, "invalid resident PIDs" unless options[:core_pid] > 1 && options[:flashd_pid] > 1 && options[:core_pid] != options[:flashd_pid]
    lock = File.open(options[:lock], File::RDWR | File::CREAT, 0o600)
    raise CaptureError, "validation lock is busy: #{options[:lock]}" unless lock.flock(File::LOCK_EX | File::LOCK_NB)

    out = File.expand_path(options[:out])
    Dir.mkdir(out, 0o700)
    created_out = true
    created_at = Time.now.utc.iso8601(6)
    File.write(File.join(out, "request.json"), JSON.pretty_generate(REQUEST) + "\n")
    core = process_identity(options[:core_pid])
    frontend = process_identity(options[:flashd_pid])
    raise CaptureError, "requested PIDs must identify flashcore and flashd" unless core["comm"] == "flashcore" && frontend["comm"] == "flashd"
    raise CaptureError, "expected flashd argv: executable GGUF socket listen" unless frontend["argv"].length == 4
    target_socket = File.expand_path(options[:socket])
    unless File.expand_path(frontend["argv"][2], frontend["cwd"]) == target_socket
      raise CaptureError, "resident flashd socket differs from requested socket"
    end
    model_path = File.expand_path(frontend["argv"][1], frontend["cwd"])
    model = model_identity(model_path)
    engine = engine_identity(options[:core_pid])
    environment = File.binread("/proc/#{options[:flashd_pid]}/environ").split("\0").to_h { |item| item.split("=", 2) }
    behavior_env = environment.select { |key, _| FRONTEND_ENV.include?(key) }
    File.write(File.join(out, "resident-processes.json"), JSON.pretty_generate("core" => core, "frontend" => frontend, "engine" => engine) + "\n")
    idle_preflight(out, options[:core_pid], target_socket)
    hardware = hardware_snapshot(out, "before")
    source_root = File.expand_path("..", __dir__)
    revision = command(["git", "rev-parse", "HEAD"], out, "source-revision", chdir: source_root).strip
    source_status = command(["git", "status", "--porcelain=v1"], out, "source-status", chdir: source_root)
    config_before = query_config(target_socket, options[:core_pid])
    File.write(File.join(out, "config-before.txt"), config_before)

    temp = Dir.mktmpdir("flash-die-balance-")
    relay = Relay.new(File.join(temp, "relay.sock"), target_socket, options[:core_pid], File.join(out, "protocol.jsonl"))
    reservation = TCPServer.new("127.0.0.1", 0)
    port = reservation.addr[1]
    reservation.close
    clone_pid = Process.spawn(behavior_env.merge("PATH" => ENV.fetch("PATH", "/usr/bin:/bin")),
      "/proc/#{options[:flashd_pid]}/exe", model_path, File.join(temp, "relay.sock"), "127.0.0.1:#{port}",
      unsetenv_others: true, chdir: frontend["cwd"], in: File::NULL,
      out: File.join(out, "frontend.stdout"), err: File.join(out, "frontend.stderr"))
    deadline = monotonic + 30
    until owned_listener?(clone_pid, port)
      relay.check!
      if Process.waitpid2(clone_pid, Process::WNOHANG)
        clone_pid = nil
        raise CaptureError, "owned frontend exited during startup"
      end
      raise CaptureError, "owned frontend startup timeout" if monotonic >= deadline
      sleep 0.05
    end
    runs = (0..options[:requests]).map do |index|
      relay.check!
      body = http_request(port, out, index)
      relay.pop.merge("index" => index, "warmup" => index.zero?, "response" => body)
    end
    relay.check!
    stop_owned_child(clone_pid)
    clone_pid = nil
    relay.close
    relay = nil
    config_after = query_config(target_socket, options[:core_pid])
    File.write(File.join(out, "config-after.txt"), config_after)
    raise CaptureError, "core identity changed" unless process_identity(options[:core_pid]) == core
    raise CaptureError, "frontend identity changed" unless process_identity(options[:flashd_pid]) == frontend
    raise CaptureError, "engine library changed" unless engine_identity(options[:core_pid]) == engine
    raise CaptureError, "model stat identity changed" unless model_identity(model_path) == model
    raise CaptureError, "GPU identities or PCIe links changed" unless hardware_snapshot(out, "after") == hardware

    capture = {"schema" => DieBalance::SCHEMA, "label" => options[:label], "created_at" => created_at,
      "hostname" => Socket.gethostname, "request" => REQUEST, "warmup_requests" => 1,
      "manifest" => {"config_before" => config_before, "config_after" => config_after,
        "frontend_sha256" => frontend["exe_sha256"], "frontend_env" => behavior_env,
        "core_exe_sha256" => core["exe_sha256"], "engine_sha256" => engine["sha256"],
        "model" => model, "hardware" => hardware, "source_revision" => revision,
        "source_status" => source_status}, "runs" => runs}
    partial = File.join(out, "capture.json.partial")
    File.open(partial, "wx") { |file| file.puts JSON.pretty_generate(capture); file.flush; file.fsync }
    checked = DieBalance.load_capture(partial)
    raise CaptureError, "capture validation failed: #{checked[:failures].join('; ')}" unless checked[:valid]
    File.rename(partial, File.join(out, "capture.json"))
    puts File.join(out, "capture.json")
    0
  rescue StandardError => error
    File.write(File.join(out, "error.txt"), "#{error.class}: #{error.message}\n") if created_out
    warn "#{error.class}: #{error.message}"
    1
  ensure
    stop_owned_child(clone_pid) if clone_pid
    begin
      relay&.close
    rescue StandardError => cleanup_error
      warn "relay cleanup: #{cleanup_error.message}"
    end
    FileUtils.remove_entry_secure(temp) if temp && File.directory?(temp)
    lock&.close
  end
end
exit DieBalanceCapture.run(ARGV) if $PROGRAM_NAME == __FILE__
