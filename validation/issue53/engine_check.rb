#!/usr/bin/env ruby
# Issue #53 live-engine gates. Run on the GPU host, with the resident core.
# Both arms and restoration stay inside the same engine-validation lock.
require 'json'
require 'socket'
require 'timeout'
require 'optparse'
require 'open3'
require 'digest'
require 'fileutils'
require 'time'

module HCCheck
  EXPECTED = {'bf16hcup' => 97, 'hcwsum' => 48, 'bf16hcrwx' => 48, 'bf16hcrx' => 49}.freeze
  SETTINGS = {'FLASH_ENGINE_MAXT' => 'maxt', 'FLASH_KV_HOST' => 'kv_host',
              'FLASH_MAXIMG' => 'img_rows', 'FLASH_TWO_POSITION' => 'two_position',
              'FLASH_PREFILL_CHUNK' => 'prefill_chunk', 'FLASH_NBLK' => 'nblk'}.freeze
  MATCH = %w[kv_host ple_disk maxt layers nblk dies attn_dies split attn_map img_rows two_position prefill_chunk].freeze

  def self.fields(line)
    raise 'Missing config response' unless line&.start_with?('config ')
    line.split.drop(1).map { |x| x.split('=', 2) }.to_h
  end

  def self.fnv(path)
    n = 0xcbf29ce484222325
    File.binread(path).each_byte { |b| n = ((n ^ b) * 0x100000001b3) & 0xffffffffffffffff }
    format('%016x', n)
  end

  def self.profile(raw)
    dies = {}
    die = nil
    raw.each_line do |line|
      if (m = line.match(/\A== die (\d+)\s*\z/))
        die = Integer(m[1])
        raise 'Duplicate profile die' if dies.key?(die)
        dies[die] = {kinds: {}}
      elsif (m = line.match(/\Aprof: (\d+) phases, ([\d.]+) ms /))
        raise 'Profile header before die' unless die
        dies[die][:phases] = Integer(m[1])
        dies[die][:total_ms] = Float(m[2])
      elsif (m = line.match(/\A\s+(\w+)\s+(\d+) phases\s+([\d.]+) us /))
        raise 'Profile kind before die' unless die
        raise 'Duplicate profile kind' if dies[die][:kinds].key?(m[1])
        dies[die][:kinds][m[1]] = {count: Integer(m[2]), us: Float(m[3])}
      end
    end
    raise 'Expected profiles for dies 0 through 5' unless dies.keys.sort == (0..5).to_a
    dies.each_value do |d|
      raise 'Missing or invalid phase header' unless d[:phases]&.positive? && d[:total_ms]&.positive?
      EXPECTED.each do |name, count|
        raise "Missing/wrong HC phase count for #{name}" unless d[:kinds].dig(name, :count) == count
      end
      hc = d[:kinds].select { |name, _| name.start_with?('bf16hc') || name == 'hcwsum' }
      raise 'Invalid HC phase time' unless hc.values.all? { |v| v[:us].finite? && v[:us].positive? }
      d[:hc_ms] = hc.values.sum { |v| v[:us] } / 1000.0
    end
    dies
  end

  def self.tokens(raw, fixture)
    lines = raw.lines.map(&:strip)
    done = lines.select { |line| line.start_with?('done ') }
    raise 'Missing/duplicate terminal done' unless done.size == 1 && lines.last == done.first
    m = done.first.match(/\Adone (\d+) ([01]) (\d+) ([\d.]+) ([\d.]+) (\d+) (\d+)\z/)
    raise 'Malformed done response' unless m
    tokens = lines.grep(/\Atok /).map do |line|
      raise 'Malformed token ID' unless line.match?(/\Atok \d+\z/)
      Integer(line.split.last)
    end
    count, ended, prompt = m.captures.first(3).map { |s| Integer(s) }
    raise 'Empty/truncated token response' unless count.positive? && count == tokens.size && count <= fixture.fetch('max_tokens')
    raise 'Wrong prompt count' unless prompt == fixture.fetch('ids').size
    raise 'Unexpected prefix reuse' unless Integer(m[6]).zero? && Integer(m[7]).zero?
    raise 'Invalid output token ID' unless tokens.all? { |v| v >= 0 && v < 248320 }
    raise 'Unfinished response' unless ended == 1 || count == fixture.fetch('max_tokens')
    raise 'Stop status does not match last token' unless (ended == 1) == fixture.fetch('stop').include?(tokens.last)
    {ids: tokens, generated: count, ended: ended, prompt: prompt, run_ms: Float(m[4]), first_ms: Float(m[5])}
  end

  class Runner
    def initialize(opt)
      @o = opt
      @seq = 0
      raise 'Use a fresh output directory' if File.exist?(@o.fetch(:out))
      FileUtils.mkdir_p(@o[:out])
      @summary = {started_utc: Time.now.utc.iso8601, host: Socket.gethostname,
                  options: @o, runner_sha256: Digest::SHA256.file(__FILE__).hexdigest,
                  arms: {}, restoration: 'not needed', model_test: 'not run by this validator'}
      persist
    end

    def persist
      tmp = File.join(@o[:out], 'summary.json.new')
      File.write(tmp, JSON.pretty_generate(@summary) + "\n")
      File.rename(tmp, File.join(@o[:out], 'summary.json'))
    end

    def log(message)
      puts "#{Time.now.utc.iso8601} #{message}"
      STDOUT.flush
    end

    def record(name, command, env = {})
      log("START #{name}")
      command = ['timeout', '--kill-after=5s', command.first == 'bash' ? '45s' : '20s', *command] unless command.first == 'curl'
      output, error, status = Open3.capture3(env, *command)
      File.write(File.join(@o[:out], "#{name}.stdout"), output)
      File.write(File.join(@o[:out], "#{name}.stderr"), error)
      File.write(File.join(@o[:out], "#{name}.command.json"), JSON.pretty_generate({argv: command, env: env, exit: status.exitstatus}) + "\n")
      raise "#{name} exited #{status.exitstatus}" unless status.success?
      output
    end

    def exchange(name, commands, terminator: /\Aok:/, seconds: 900)
      log("START #{name}")
      File.write(File.join(@o[:out], "#{name}.commands.txt"), commands.join("\n") + "\n")
      output = +''
      begin
        Timeout.timeout(seconds) do
          UNIXSocket.open(@o.fetch(:socket)) do |s|
            s.write(commands.join("\n") + "\n")
            loop do
              line = s.gets or raise "#{name}: unexpected EOF"
              output << line
              raise "#{name}: #{line.strip}" if line.start_with?('error:')
              break if line.match?(terminator)
            end
          end
        end
      ensure
        File.write(File.join(@o[:out], "#{name}.response.txt"), output)
      end
      output
    end

    def config(name)
      HCCheck.fields(exchange(name, ['config'], terminator: /\Aconfig /, seconds: 30).lines.first)
    end

    def identity
      pid = @o.fetch(:core_pid)
      raise 'Named PID is not flashcore' unless File.read("/proc/#{pid}/comm").strip == 'flashcore'
      File.read("/proc/#{pid}/stat").split[21]
    end

    def loaded_library
      paths = File.readlines("/proc/#{@o[:core_pid]}/maps").filter_map do |line|
        path = line.split.last
        path if path&.match?(%r{/flashcore-#{@o[:core_pid]}-gen\d+\.so\z})
      end.uniq
      raise 'Cannot identify one loaded engine image' unless paths.size == 1
      paths.first
    end

    def idle(name)
      deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + @o[:wait_seconds]
      loop do
        raise 'Core process identity changed' unless identity == @core_identity
        @seq += 1
        prefix = "#{name}-idle-#{@seq}"
        rows = record(prefix + '-gpu', ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.free', '--format=csv,noheader,nounits']).lines.map { |l| l.strip.split(/\s*,\s*/) }
        raise 'Malformed GPU inventory' unless rows.size >= 6 && rows.all? { |r| r.size == 4 && r[0].match?(/\A\d+\z/) && r[1].start_with?('GPU-') && r[2..3].all? { |x| x.match?(/\A\d+\z/) } }
        selected = rows.select { |r| Integer(r[0]) < 6 }
        raise 'Missing production GPU' unless selected.map { |r| Integer(r[0]) }.sort == (0..5).to_a
        uuids = selected.map { |r| r[1] }
        apps = record(prefix + '-apps', ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits']).lines.map { |l| l.strip.split(/\s*,\s*/, 3) }
        busy = selected.select { |r| Integer(r[2]) != 0 }.map { |r| "active GPU #{r[0]} (#{r[2]}%)" }
        apps.each do |r|
          raise 'Malformed compute-process inventory' unless r.size == 3 && r[1].match?(/\A\d+\z/)
          next unless uuids.include?(r[0])
          next if Integer(r[1]) == @o[:core_pid] || %w[hold fence].include?(File.basename(r[2]))
          busy << "GPU consumer #{r[1]} #{r[2]}"
        end
        record(prefix + '-ps', ['ps', '-eo', 'pid,ppid,stat,etimes,comm'])
        sockets = record(prefix + '-sockets', ['ss', '-H', '-x', '-n', 'state', 'established'])
        busy << 'another client is connected to the engine socket' if sockets.lines.any? { |line| line.include?(@o[:socket]) }
        return if busy.empty?
        raise "Device stayed busy for #{@o[:wait_seconds]} seconds: #{busy.join(', ')}" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline
        log("WAIT #{busy.join(', ')}")
        sleep 5
      end
    end

    def assert_arm(name, directory, revision)
      conf = config(name)
      raise 'Core process identity changed' unless identity == @core_identity
      MATCH.each { |key| raise "Runtime setting changed: #{key}" unless conf.fetch(key) == @original.fetch(key) }
      raise 'Wrong loaded source revision' unless conf.fetch('rev') == revision
      raise 'Wrong loaded cubin' unless conf.fetch('cubin_fnv') == HCCheck.fnv(File.join(directory, 'die.cubin'))
      raise 'Wrong loaded library' unless Digest::SHA256.file(loaded_library).hexdigest == Digest::SHA256.file(File.join(directory, 'libengine.so')).hexdigest
      conf
    end

    def reload(name, library, cubin)
      raise 'Core process identity changed before reload' unless identity == @core_identity
      raise 'Protocol paths cannot contain whitespace' if [library, cubin].any? { |p| p.match?(/\s/) }
      settings = SETTINGS.map { |env, key| "#{env}=#{@original.fetch(key)}" }
      settings += ["FLASH_CUBIN=#{cubin}", "FLASH_LAYERS=#{@original.fetch('layers').split('/').first}"]
      @touched = true
      response = exchange(name, ["reload #{library} #{settings.join(' ')}"])
      raise 'Reload was not acknowledged' unless response.strip == 'ok: engine reloaded'
      raise 'Core process identity changed during reload' unless identity == @core_identity
    end

    def run_arm(name, directory, revision, fixtures)
      idle("#{name}-reload")
      reload("#{name}-reload", File.join(directory, 'libengine.so'), File.join(directory, 'die.cubin'))
      arm = @summary[:arms][name] = {config: assert_arm("#{name}-config", directory, revision), tokens: {}, requests: [], profiles: []}
      fixtures.each do |fixture|
        label = "#{name}-tokens-#{fixture.fetch('name')}"
        commands = ['snap 0', 'stop ' + fixture.fetch('stop').join(' '), 'sample 1 0 1 1',
                    "chat #{fixture.fetch('max_tokens') - 1} #{fixture.fetch('ids').join(' ')}"]
        raw = exchange(label, commands, terminator: /\Adone /)
        arm[:tokens][fixture['name']] = HCCheck.tokens(raw, fixture)
        persist
      end
      request = JSON.generate({messages: [{role: 'user', content: 'What is the capital of France? Answer in one word.'}], max_tokens: 40, temperature: 0})
      4.times do |i|
        assert_arm("#{name}-request-#{i}-config", directory, revision)
        label = "#{name}-http-#{i}"
        raw = record(label, ['curl', '--fail-with-body', '--silent', '--show-error', '--max-time', '180',
                             @o.fetch(:url) + '/v1/chat/completions', '-H', 'content-type: application/json', '--data-binary', request])
        response = JSON.parse(raw)
        speed = response.fetch('timings').fetch('predicted_per_second')
        count = response.fetch('usage').fetch('completion_tokens')
        raise 'Invalid HTTP timing or generated count' unless speed.is_a?(Numeric) && speed.finite? && speed.positive? && count.is_a?(Integer) && count > 1
        arm[:requests] << {warmup: i.zero?, predicted_per_second: speed, completion_tokens: count}
        next if i.zero?
        # No generation may occur between the timed response and these profiles.
        profile = record("#{name}-profile-#{i}", ['bash', File.join(@o.fetch(:candidate_dir), 'prof.sh')], {'PROF_ALL' => '1', 'FLASH_SOCK' => @o[:socket]})
        arm[:profiles] << HCCheck.profile(profile)
        persist
      end
      arm[:mean_tok_s] = arm[:requests].reject { |r| r[:warmup] }.sum { |r| r[:predicted_per_second] } / 3.0
      arm[:max_hc_ms] = arm[:profiles].flat_map { |p| p.values.map { |d| d[:hc_ms] } }.max
      assert_arm("#{name}-final-config", directory, revision)
      persist
      log("ARM #{name}: mean=#{arm[:mean_tok_s]} tok/s max_HC=#{arm[:max_hc_ms]} ms")
    end

    def lock_file
      File.open('/tmp/flash-sass-engine-validation.lock', 'a')
    end

    def run
      fixtures = JSON.parse(File.read(@o.fetch(:fixtures))).fetch('cases')
      raise 'Expected nonempty uniquely named prompt fixtures' unless fixtures.any? && fixtures.map { |f| f.fetch('name') }.uniq.size == fixtures.size
      fixtures.each do |f|
        raise 'Invalid fixture name' unless f['name'].match?(/\A[a-z0-9_]+\z/)
        raise 'Invalid fixture IDs' unless f['ids'].is_a?(Array) && f['ids'].any? && (f['ids'] + f.fetch('stop')).all? { |v| v.is_a?(Integer) && v >= 0 && v < 248320 }
        raise 'Invalid fixture generation limit' unless f['max_tokens'].is_a?(Integer) && f['max_tokens'] > 1
      end
      @summary[:fixtures_sha256] = Digest::SHA256.file(@o[:fixtures]).hexdigest
      lock = lock_file
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      until lock.flock(File::LOCK_EX | File::LOCK_NB)
        raise 'Shared lock wait budget exhausted' if Process.clock_gettime(Process::CLOCK_MONOTONIC) - started >= @o[:wait_seconds]
        log('WAIT shared engine-validation lock')
        sleep 5
      end
      @summary[:lock_wait_seconds] = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
      @core_identity = identity
      environment = File.binread("/proc/#{@o[:core_pid]}/environ").split("\0").map { |x| x.split('=', 2) }.to_h
      raise 'Core socket mismatch' unless environment['FLASH_SOCK'] == @o[:socket]
      idle('original')
      @original = config('original-config')
      raise 'Expected complete six-die 48-layer engine' unless @original['dies'] == '6' && @original['layers'] == '48/48'
      original_cubin = File.expand_path(@original.fetch('cubin'), File.readlink("/proc/#{@o[:core_pid]}/cwd"))
      raise 'Original cubin pathname no longer matches loaded bytes' unless HCCheck.fnv(original_cubin) == @original.fetch('cubin_fnv')
      saved_library = File.join(@o[:out], 'original-engine.so')
      saved_cubin = File.join(@o[:out], 'original-die.cubin')
      FileUtils.cp(loaded_library, saved_library)
      FileUtils.cp(original_cubin, saved_cubin)
      @summary[:original] = {config: @original, library_sha256: Digest::SHA256.file(saved_library).hexdigest, cubin_sha256: Digest::SHA256.file(saved_cubin).hexdigest}
      manifests = File.binread("/proc/#{@o[:core_pid]}/cmdline").split("\0").drop(3)
      @summary[:manifests] = manifests.to_h { |path| [path, Digest::SHA256.file(path).hexdigest] }
      persist
      begin
        run_arm('baseline', @o.fetch(:baseline_dir), @o.fetch(:baseline_rev), fixtures)
        run_arm('candidate', @o.fetch(:candidate_dir), @o.fetch(:candidate_rev), fixtures)
        baseline = @summary[:arms].fetch('baseline')
        candidate = @summary[:arms].fetch('candidate')
        @summary[:gates] = {token_ids_match: fixtures.all? { |f| baseline[:tokens][f['name']][:ids] == candidate[:tokens][f['name']][:ids] },
                           hc_at_most_4_ms: candidate[:max_hc_ms] <= 4.0,
                           decode_at_least_38_tok_s: candidate[:mean_tok_s] >= 38.0}
        @summary[:status] = @summary[:gates].values.all? ? 'PASS_LIVE_GATES' : 'FAIL_LIVE_GATES'
      ensure
        if @touched
          old_int = Signal.trap('INT', 'IGNORE')
          old_term = Signal.trap('TERM', 'IGNORE')
          @summary[:restoration] = 'in progress'
          persist
          idle('restore')
          reload('restore-engine', saved_library, saved_cubin)
          restored = config('restored-config')
          (MATCH + %w[rev cubin_fnv]).each { |key| raise "Restoration mismatch: #{key}" unless restored.fetch(key) == @original.fetch(key) }
          raise 'Restored library bytes differ' unless Digest::SHA256.file(loaded_library).hexdigest == @summary[:original][:library_sha256]
          @summary[:restoration] = 'PASS'
          log('PASS original engine restored')
          Signal.trap('INT', old_int)
          Signal.trap('TERM', old_term)
        end
      end
    rescue StandardError, Interrupt => e
      @summary[:status] = 'ERROR'
      @summary[:error] = "#{e.class}: #{e.message}"
      @summary[:error_chain] = []
      cause = e
      while cause
        @summary[:error_chain] << "#{cause.class}: #{cause.message}"
        cause = cause.cause
      end
      log("ERROR #{@summary[:error]}")
    ensure
      @summary[:finished_utc] = Time.now.utc.iso8601
      persist
      lock&.close
    end

    def success?
      @summary[:status] == 'PASS_LIVE_GATES' && @summary[:restoration] == 'PASS'
    end
  end
end

if $PROGRAM_NAME == __FILE__
  opt = {socket: '/tmp/flash.sock', url: 'http://127.0.0.1:18090', wait_seconds: 2700}
  OptionParser.new do |p|
    %i[baseline_dir candidate_dir baseline_rev candidate_rev fixtures out socket url].each do |key|
      p.on("--#{key.to_s.tr('_', '-')} VALUE") { |v| opt[key] = v }
    end
    p.on('--core-pid PID', Integer) { |v| opt[:core_pid] = v }
    p.on('--wait-seconds N', Integer) { |v| opt[:wait_seconds] = v }
  end.parse!
  %i[baseline_dir candidate_dir baseline_rev candidate_rev fixtures out core_pid].each { |key| abort("Missing --#{key.to_s.tr('_', '-')}") unless opt[key] }
  %i[baseline_dir candidate_dir fixtures out socket].each { |key| opt[key] = File.expand_path(opt[key]) }
  abort 'Wait must be in 1..2700 seconds' unless (1..2700).cover?(opt[:wait_seconds])
  Signal.trap('TERM') { raise Interrupt, 'SIGTERM' }
  runner = HCCheck::Runner.new(opt)
  runner.run
  exit(runner.success? ? 0 : 1)
end
