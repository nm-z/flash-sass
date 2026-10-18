#!/usr/bin/env ruby
# Reproducible issue #19 validation. All paths come from this checkout or options.
require 'optparse'
require 'json'
require 'digest'
require 'fileutils'
require 'open3'
require 'socket'
require 'tmpdir'
require 'time'

# nvidia-smi emits fixed-column query rows, not quoted general-purpose CSV.
# A split limit preserves commas in the final process-name column. Reject
# malformed rows instead of silently treating an unreadable inventory as idle.
def nvidia_rows(text, columns)
  text.lines.reject { |line| line.strip.empty? }.map do |line|
    row = line.strip.split(/\s*,\s*/, columns)
    raise "Malformed nvidia-smi row (expected #{columns} columns)" unless row.size == columns && row.none?(&:empty?)
    row
  end
end

ROOT = File.expand_path('../..', __dir__)
opt = {mode: 'cpu', cuda: ENV.fetch('CUDA', '/opt/cuda'),
       ptxas: ENV.fetch('PTXAS11', '/opt/cuda-11.4/bin/ptxas'),
       maxas: ENV.fetch('MAXAS_DIR', File.join(ROOT, 'maxas')),
       lock_dir: ENV.fetch('GPU_LOCK_DIR', File.join(Dir.tmpdir, 'flash-sass-gpu-locks')),
       engine_lock: ENV.fetch('FLASH_ENGINE_VALIDATION_LOCK', File.join(Dir.tmpdir, 'flash-sass-engine-validation.lock')),
       positions: [32768, 65536], iterations: 10000, samples: 12}
OptionParser.new do |p|
  p.banner = 'Usage: run.rb --mode cpu|build|gpu|model --out DIR [options]'
  p.on('--mode MODE') { |v| opt[:mode] = v }
  p.on('--out DIR') { |v| opt[:out] = File.expand_path(v) }
  p.on('--cuda DIR') { |v| opt[:cuda] = File.expand_path(v) }
  p.on('--ptxas FILE') { |v| opt[:ptxas] = File.expand_path(v) }
  p.on('--maxas-dir DIR', 'Assembler dependency; all source hashes are recorded') { |v| opt[:maxas] = File.expand_path(v) }
  p.on('--gpu UUID', 'Required for isolated single-GPU tests') { |v| opt[:gpu] = v }
  p.on('--lock-dir DIR') { |v| opt[:lock_dir] = File.expand_path(v) }
  p.on('--engine-lock FILE', 'Shared lock used by engine validation workers') { |v| opt[:engine_lock] = File.expand_path(v) }
  p.on('--socket FILE', 'Existing resident core; never starts/stops another worker') { |v| opt[:socket] = File.expand_path(v) }
  p.on('--core-pid PID', Integer) { |v| opt[:core_pid] = v }
  p.on('--checkpoint FILE', 'GGUF shard containing the tokenizer') { |v| opt[:checkpoint] = File.expand_path(v) }
  p.on('--corpus FILE', 'UTF-8 source text; default is tracked repository text') { |v| opt[:corpus] = File.expand_path(v) }
  p.on('--positions LIST', 'Comma-separated true prefill lengths') { |v| opt[:positions] = v.split(',').map { |x| Integer(x) } }
  p.on('--iterations N', Integer) { |v| opt[:iterations] = v }
  p.on('--samples N', Integer) { |v| opt[:samples] = v }
end.parse!
abort 'An output directory is required' unless opt[:out]
abort 'Unknown mode' unless %w[cpu build gpu model].include?(opt[:mode])
FileUtils.mkdir_p(opt[:out])
Dir.chdir(ROOT)
head, st = Open3.capture2('git', 'rev-parse', 'HEAD')
abort 'Cannot identify source revision' unless st.success?
head = head.strip
abort 'Commit tracked source changes before recording evidence' unless system('git', 'diff', '--quiet', 'HEAD', '--', '.', out: File::NULL)
committed_runner, runner_status = Open3.capture2('git', 'show', head + ':validation/issue19/run.rb')
abort 'The running validator must match the committed revision' unless runner_status.success? && committed_runner == File.read(__FILE__)
manifest_path = File.join(opt[:out], "#{opt[:mode]}.json")
manifest = {mode: opt[:mode], source_revision: head, hostname: Socket.gethostname,
            started_utc: Time.now.utc.iso8601, runner_sha256: Digest::SHA256.file(__FILE__).hexdigest,
            options: opt, results: []}
persist = lambda do
  tmp = manifest_path + '.new'
  File.write(tmp, JSON.pretty_generate(manifest) + "\n")
  File.rename(tmp, manifest_path)
end
persist.call
record = lambda do |name, command, cwd: ROOT, env: {}, expected: 0|
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  stdout = File.join(opt[:out], "#{name}.stdout")
  stderr = File.join(opt[:out], "#{name}.stderr")
  clear = ENV.keys.grep(/\A(?:QSA_|DIE_CUBIN\z|ATTN_POISON_)/).to_h { |k| [k, nil] }
  row = {name: name, command: command, cwd: cwd, env: env,
         stdout: File.basename(stdout), stderr: File.basename(stderr), expected_exit: expected}
  puts "START #{name}"; STDOUT.flush
  File.open(stdout, 'w') do |so|
    File.open(stderr, 'w') do |se|
      pid = Process.spawn(clear.merge(env), *command, chdir: cwd, out: so, err: se)
      _, status = Process.wait2(pid)
      row[:exit_code] = status.exitstatus || 128 + status.termsig
    end
  end
  row[:seconds] = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  row[:passes] = row[:exit_code] == expected
  row[:stdout_sha256] = Digest::SHA256.file(stdout).hexdigest
  row[:stderr_sha256] = Digest::SHA256.file(stderr).hexdigest
  manifest[:results] << row
  persist.call
  puts "#{row[:passes] ? 'PASS' : 'FAIL'} #{name} exit=#{row[:exit_code]} seconds=#{format('%.3f', row[:seconds])}"; STDOUT.flush
  raise "#{name}: expected exit #{expected}, got #{row[:exit_code]} (see #{stderr})" unless row[:passes]
  File.read(stdout)
end

# Builders use git-archive copies, never a hardcoded worktree and never stale objects.
variants = {bounded: {}, unbounded: {'QSA_KV_UNBOUNDED' => '1'}, trace: {'QSA_KV_TRACE' => '1'}}
build_root = File.join(opt[:out], 'build')
kern = ->(variant) { File.join(build_root, variant.to_s, 'kern') }
build_meta = File.join(opt[:out], 'build.json')

begin
  case opt[:mode]
  when 'cpu'
    exe = File.join(opt[:out], 'quant_range_test')
    record.call('cpu-build', [ENV.fetch('CC', 'cc'), '-std=gnu99', '-O2', '-ffp-contract=off', '-UNDEBUG', '-Wall', '-Wextra',
      '-o', exe, 'kern/quant_range_test.c', 'ref/qsa.c', 'ref/fmath.c', '-Iref', '-lm'])
    record.call('cpu-boundaries', [exe])
    manifest[:executable_sha256] = Digest::SHA256.file(exe).hexdigest
  when 'build'
    raise '--maxas-dir must contain bin/maxas.pl' unless File.file?(File.join(opt[:maxas], 'bin', 'maxas.pl'))
    abort 'Use a new output directory for a fresh build' if File.exist?(build_root)
    FileUtils.mkdir_p(build_root)
    archive = File.join(build_root, 'source.tar')
    record.call('source-archive', ['git', 'archive', '--format=tar', "--output=#{archive}", head])
    record.call('toolchain-cc', [ENV.fetch('CC', 'cc'), '--version'])
    record.call('toolchain-ptxas', [opt[:ptxas], '--version'])
    variants.each do |variant, flags|
      dir = File.dirname(kern.call(variant))
      FileUtils.mkdir_p(dir)
      record.call("extract-#{variant}", ['tar', '-xf', archive, '-C', dir])
      FileUtils.cp_r(opt[:maxas], File.join(dir, 'maxas'))
      record.call("build-#{variant}", ['make', '-B', '-C', kern.call(variant), "CUDA=#{opt[:cuda]}", "PTXAS11=#{opt[:ptxas]}",
        "FLASH_REV=#{head}", 'quant_range_test', 'qsa_test', 'quant_bench', 'die.cubin', 'libengine.so', 'flashcore', 'tok_test'], env: flags)
    end
    manifest[:artifacts] = variants.keys.to_h do |v|
      [v, %w[quant_range_test qsa_test quant_bench die.cubin libengine.so flashcore tok_test die.sass].to_h do |f|
        path = File.join(kern.call(v), f)
        [f, {sha256: Digest::SHA256.file(path).hexdigest, bytes: File.size(path)}]
      end]
    end
    manifest[:maxas] = Dir.glob(File.join(opt[:maxas], '{bin,lib}', '**', '*')).select { |f| File.file?(f) }.sort.to_h do |f|
      [f.delete_prefix(opt[:maxas] + '/'), Digest::SHA256.file(f).hexdigest]
    end
  else
    prior = JSON.parse(File.read(build_meta))
    raise 'Build revision does not equal current head' unless prior['source_revision'] == head && prior['status'] == 'PASS'
    prior.fetch('artifacts').each do |v, files|
      files.each do |f, data|
        raise "Artifact changed: #{v}/#{f}" unless Digest::SHA256.file(File.join(kern.call(v), f)).hexdigest == data['sha256']
      end
    end
    manifest[:build_manifest_sha256] = Digest::SHA256.file(build_meta).hexdigest
    FileUtils.mkdir_p(opt[:lock_dir])
    locks = []
    FileUtils.mkdir_p(File.dirname(opt[:engine_lock]))
    engine_lock = File.open(opt[:engine_lock], 'a')
    until engine_lock.flock(File::LOCK_EX | File::LOCK_NB)
      puts 'WAIT shared engine-validation lock is held'; STDOUT.flush; sleep 10
    end
    locks << engine_lock
    if opt[:mode] == 'gpu'
      raise '--gpu must specify a GPU UUID' unless opt[:gpu]&.start_with?('GPU-')
      lock = File.open(File.join(opt[:lock_dir], opt[:gpu] + '.lock'), 'a')
      until lock.flock(File::LOCK_EX | File::LOCK_NB)
        puts 'WAIT GPU advisory lock is held'; STDOUT.flush; sleep 10
      end
      locks << lock
    else
      raise 'model requires --socket, --core-pid and --checkpoint' unless opt.values_at(:socket, :core_pid, :checkpoint).all?
      raise 'Invalid positions' unless opt[:positions].all? { |n| n >= 1 && n < 262144 }
    end
    seq = 0
    idle = lambda do |name|
      loop do
        seq += 1
        prefix = "idle-#{seq}-#{name}"
        processes = record.call(prefix + '-nvidia-processes', ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'])
        gpu = record.call(prefix + '-nvidia-gpus', ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,clocks.sm', '--format=csv,noheader,nounits'])
        ps = record.call(prefix + '-ps', ['ps', '-eo', 'pid,ppid,stat,wchan:25,etimes,time,comm'])
        rows = nvidia_rows(processes, 3)
        gpu_rows = nvidia_rows(gpu, 7)
        raise 'The GPU inventory is empty' if gpu_rows.empty?
        busy = []
        rows.each do |r|
          pid = r[1].to_i
          if opt[:mode] == 'gpu' && r[0] == opt[:gpu]
            busy << "test GPU owned by pid #{pid} #{r[2]}"
            next
          end
          comm = File.read("/proc/#{pid}/comm").strip rescue 'exited'
          line = ps.lines.find { |l| l.split.first == pid.to_s }
          sleeping = line && line.split[2].start_with?('S')
          holder = %w[hold fence].include?(comm) && line && line.split[3] == 'do_sys_pause'
          resident = (pid == opt[:core_pid] && comm == 'flashcore' && opt[:socket]) || holder
          busy << "pid #{pid} #{comm} is not an idle resident" unless resident && sleeping
        end
        if opt[:socket]
          sockets = record.call(prefix + '-sockets', ['ss', '-xnp'])
          busy << 'resident core has a connected client' if sockets.lines.any? { |l| l.include?(opt[:socket]) || l.include?("pid=#{opt[:core_pid]},") }
        end
        # Also wait for active computation on other dies: no interference with workers.
        gpu_rows.each do |r|
          util = r[5].to_s.strip
          raise 'Cannot establish GPU utilization' unless util.match?(/\A\d+\z/)
          busy << "active GPU #{r[0]} utilization #{util}" if util.to_i > 0
        end
        raise 'Requested GPU UUID was not found' if opt[:mode] == 'gpu' && !gpu_rows.any? { |r| r[1] == opt[:gpu] }
        if busy.empty?
          break
        end
        puts "WAIT #{busy.join('; ')}"; STDOUT.flush
        sleep 10
      end
    end
    if opt[:mode] == 'gpu'
      env = {'CUDA_VISIBLE_DEVICES' => opt[:gpu]}
      record.call('cpu-boundaries', [File.join(kern.call(:bounded), 'quant_range_test')])
      [false, true].each do |host|
        suffix = host ? 'host' : 'vram'
        base = env.merge(host ? {'QSA_HOSTKV' => '1'} : {})
        %i[unbounded bounded trace].each do |variant|
          [false, true].each do |extreme|
            name = "#{variant}-#{extreme ? 'boundaries' : 'normal'}-#{suffix}"
            idle.call(name)
            e = base.merge('DIE_CUBIN' => File.join(kern.call(variant), 'die.cubin'))
            e['QSA_KV_EXTREMES'] = '1' if extreme
            e['QSA_KV_TRACE_CHECK'] = '1' if variant == :trace
            rc = variant == :unbounded && extreme ? 1 : 0
            out = record.call(name, [File.join(kern.call(:bounded), 'qsa_test'), 'kv', '16', '2'], cwd: kern.call(:bounded), env: e, expected: rc)
            raise 'Control failed without the quantizer witness' if rc == 1 && !out.match?(/nonfinite|finite|scale/i)
          end
        end
        idle.call('quant-benchmark-' + suffix)
        record.call('quant-benchmark-' + suffix, [File.join(kern.call(:bounded), 'quant_bench'),
          '--bounded', File.join(kern.call(:bounded), 'die.cubin'), '--unbounded', File.join(kern.call(:unbounded), 'die.cubin'),
          '--trace', File.join(kern.call(:trace), 'die.cubin'), '--iterations', opt[:iterations].to_s, '--samples', opt[:samples].to_s,
          '--host', host ? '1' : '0'], env: env, cwd: kern.call(:bounded))
      end
    else
      # The core serializes clients. Keep each preflight and request on one
      # connection; never send quit, kill a holder, or create another core.
      exchange = lambda do |name, commands, terminator: nil, validate: nil|
        so_path = File.join(opt[:out], name + '.stdout')
        in_path = File.join(opt[:out], name + '.commands')
        File.write(in_path, ((validate ? ['config'] : []) + commands).join("\n") + "\n")
        start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        puts "START #{name}"; STDOUT.flush
        sock = UNIXSocket.new(opt[:socket])
        reached = false
        begin
          File.open(so_path, 'w') do |so|
            so.sync = true
            if validate
              sock.puts('config')
              line = sock.gets or raise 'Core closed before config'
              so.write(line)
              validate.call(line)
            end
            commands.each { |c| sock.puts(c) }
            while (line = sock.gets)
              so.write(line)
              if terminator && line.match?(terminator)
                reached = true
                break
              end
            end
          end
        ensure
          sock.close
        end
        output = File.read(so_path)
        row = {name: name, transport: 'resident Unix socket', commands_file: File.basename(in_path),
               commands_sha256: Digest::SHA256.file(in_path).hexdigest,
               stdout: File.basename(so_path), stdout_sha256: Digest::SHA256.file(so_path).hexdigest,
               seconds: Process.clock_gettime(Process::CLOCK_MONOTONIC) - start,
               passes: !output.match?(/^error:/) && (!terminator || reached)}
        manifest[:results] << row
        persist.call
        raise "#{name}: incomplete/error response" unless row[:passes]
        puts "PASS #{name} seconds=#{format('%.3f', row[:seconds])}"; STDOUT.flush
        output
      end
      fields = ->(line) { line.split.filter_map { |s| s.include?('=') ? s.split('=', 2) : nil }.to_h }
      idle.call('model-inventory')
      raise 'The named PID is not flashcore' unless File.read("/proc/#{opt[:core_pid]}/comm").strip == 'flashcore'
      core_env = File.read("/proc/#{opt[:core_pid]}/environ").split("\0").map { |s| s.split('=', 2) }.to_h
      raise 'PID does not own the requested socket' unless core_env['FLASH_SOCK'] == opt[:socket]
      original_text = exchange.call('original-config', ['config', 'info'], terminator: /^info /)
      original = fields.call(original_text.lines.find { |l| l.start_with?('config ') } || '')
      raise 'Expected the full six-die, 48-layer QSA checkpoint engine' unless original['layers'] == '48/48' && original['dies'] == '6' && original['nblk'] == '512'
      cmdline = File.read("/proc/#{opt[:core_pid]}/cmdline").split("\0")
      ple = File.expand_path(cmdline.fetch(1), File.readlink("/proc/#{opt[:core_pid]}/cwd"))
      raise 'A real PLE checkpoint file is required' unless File.file?(ple)
      loaded_libs = File.readlines("/proc/#{opt[:core_pid]}/maps").filter_map do |line|
        path = line.split.last
        path if path&.match?(%r{/flashcore-#{opt[:core_pid]}-gen\d+\.so\z})
      end.uniq
      raise 'Cannot identify exactly one loaded engine image for restoration' unless loaded_libs.size == 1
      original_cubin = File.expand_path(original.fetch('cubin'), File.readlink("/proc/#{opt[:core_pid]}/cwd"))
      fnv = 0xcbf29ce484222325
      File.binread(original_cubin).each_byte { |b| fnv = ((fnv ^ b) * 0x100000001b3) & 0xffffffffffffffff }
      raise 'The original cubin file differs from the loaded image; refusing a non-reversible reload' unless format('%016x', fnv) == original['cubin_fnv']
      restore_lib = File.join(opt[:out], 'original-engine.so')
      restore_cubin = File.join(opt[:out], 'original-die.cubin')
      FileUtils.cp(loaded_libs.first, restore_lib)
      FileUtils.cp(original_cubin, restore_cubin)
      manifest[:original_engine] = {config: original, library_sha256: Digest::SHA256.file(restore_lib).hexdigest,
                                    cubin_sha256: Digest::SHA256.file(restore_cubin).hexdigest}
      identities = [opt[:checkpoint], ple] + cmdline.drop(3)
      identities.each { |f| raise "Missing checkpoint/holder input #{f}" unless File.file?(f) }
      record.call('checkpoint-and-holder-sha256', ['sha256sum', *identities])
      manifest[:checkpoint_files] = identities.map { |f| s = File.stat(f); {path: f, bytes: s.size, mtime_utc: s.mtime.utc.iso8601} }
      corpus = opt[:corpus] || File.join(opt[:out], 'model-corpus.txt')
      unless opt[:corpus]
        tracked, status = Open3.capture2('git', 'ls-files')
        raise 'Cannot enumerate corpus files' unless status.success?
        sources = tracked.lines.map(&:strip).select { |f| %w[README.md SPEC.md].include?(f) || f.match?(%r{\A(?:kern|ref)/[^/]+\.(?:c|sass|pl)\z}) }.sort
        File.open(corpus, 'w') { |f| sources.each { |src| f.puts("\nFile: #{src}\n"); f.write(File.read(src)); f.puts } }
        manifest[:corpus_sources] = sources.to_h { |f| [f, Digest::SHA256.file(f).hexdigest] }
      end
      manifest[:corpus_sha256] = Digest::SHA256.file(corpus).hexdigest
      all_ids = JSON.parse(record.call('tokenize-corpus', [File.join(kern.call(:bounded), 'tok_test'), opt[:checkpoint], corpus, 'raw']))
      raise 'Corpus has too few tokens for the requested contexts' unless all_ids.size >= opt[:positions].max
      raise 'Tokenizer returned invalid IDs' unless all_ids.all? { |x| x.is_a?(Integer) && x >= 0 && x <= 0xffffffff }
      restore = {'FLASH_CUBIN' => restore_cubin, 'FLASH_ENGINE_MAXT' => original.fetch('maxt'),
                 'FLASH_KV_HOST' => original.fetch('kv_host'), 'FLASH_MAXIMG' => original.fetch('img_rows'),
                 'FLASH_TWO_POSITION' => original.fetch('two_position'), 'FLASH_LAYERS' => '48',
                 'FLASH_NBLK' => original.fetch('nblk'), 'FLASH_KV_TRACE' => '0'}
      settings = {'FLASH_CUBIN' => File.join(kern.call(:trace), 'die.cubin'),
                  'FLASH_ENGINE_MAXT' => (opt[:positions].max + 16).to_s, 'FLASH_KV_HOST' => '1',
                  'FLASH_MAXIMG' => '1', 'FLASH_TWO_POSITION' => '0', 'FLASH_LAYERS' => '48',
                  'FLASH_NBLK' => '512', 'FLASH_KV_TRACE' => '1'}
      raise 'Engine protocol cannot encode whitespace in paths' if [restore_lib, *restore.values, *settings.values].any? { |v| v.match?(/\s/) }
      touched = false
      begin
        idle.call('model-reload')
        touched = true
        result = exchange.call('model-reload', ['reload ' + File.join(kern.call(:trace), 'libengine.so') + ' ' + settings.map { |k, v| "#{k}=#{v}" }.join(' ')])
        raise 'Trace engine was not loaded' unless result.include?('ok: engine reloaded')
        manifest[:model_contexts] = []
        opt[:positions].each_with_index do |n, index|
          # Both are real full-prefix runs. Distinct source slices avoid merely
          # repeating a short token sequence or resuming an old KV snapshot.
          offset = index.zero? ? 0 : (all_ids.size - n) / 2
          ids = all_ids.slice(offset, n)
          ids_path = File.join(opt[:out], "model-#{n}.tokens.json")
          File.write(ids_path, JSON.generate(ids) + "\n")
          idle.call("model-#{n}")
          check_config = lambda do |line|
            conf = fields.call(line)
            raise 'Wrong engine/cubin/configuration before request' unless conf['rev'] == head && conf['layers'] == '48/48' && conf['nblk'] == '512' && conf['kv_host'] == '1' && conf['kv_trace'] == '1' && conf['cubin'] == settings['FLASH_CUBIN']
          end
          output = exchange.call("model-#{n}", ['snap 0', 'stop', 'gen 0 ' + ids.join(' '), 'kvstats', 'statehash', 'info'],
                                 terminator: /^info /, validate: check_config)
          time = output.lines.find { |l| l.start_with?('time: ') }
          raise 'The full requested context was not processed from position zero' unless time&.match?(/for #{n} positions run \(0 from the snapshot;/)
          stats = output.lines.select { |l| l.start_with?('kvstats layer=') }.map { |l| fields.call(l) }
          raise 'Expected 24 per-layer K/V rows' unless stats.size == 24
          expected_rows = (3...48).step(4).flat_map { |l| %w[K V].map { |kind| [l.to_s, kind] } }
          raise 'Duplicate/missing layer metrics' unless stats.map { |r| r.values_at('layer', 'kind') }.sort == expected_rows.sort
          stats.each do |r|
            raise 'Incorrect group coverage' unless r['groups'].to_i == n * 16
            %w[missing_slots nonfinite_input_groups nonfinite_stored_scales scale_mismatches].each do |key|
              raise "Invalid model trace: #{key}" unless r.fetch(key).to_i.zero?
            end
            raise 'Per-layer trace rejected its own coverage' unless r.fetch('status') == 'PASS'
          end
          raise 'Trace did not complete successfully' unless output.lines.any? { |l| l.strip == "kvstats done status=PASS rows=24 positions=#{n} start=0" }
          manifest[:model_contexts] << {positions: n, corpus_token_offset: offset, tokens_sha256: Digest::SHA256.file(ids_path).hexdigest,
                                        time: time.strip, layers: stats}
          persist.call
        end
      ensure
        if touched
          idle.call('restore-engine')
          restored = exchange.call('restore-engine', ['reload ' + restore_lib + ' ' + restore.map { |k, v| "#{k}=#{v}" }.join(' ')])
          raise 'Could not restore the original resident engine' unless restored.include?('ok: engine reloaded')
          after = exchange.call('restored-config', ['config', 'info'], terminator: /^info /)
          conf = fields.call(after.lines.find { |l| l.start_with?('config ') } || '')
          %w[rev kv_host maxt layers nblk dies img_rows two_position cubin_fnv].each do |key|
            raise "Restored #{key} differs from the original" unless conf[key] == original[key]
          end
          manifest[:original_engine_restored] = true
        end
      end
    end
  end
  manifest[:status] = 'PASS'
rescue StandardError => e
  manifest[:status] = 'FAIL'
  manifest[:error] = "#{e.class}: #{e.message}"
  warn manifest[:error]
ensure
  locks&.reverse_each { |lock| lock.flock(File::LOCK_UN); lock.close }
  manifest[:finished_utc] = Time.now.utc.iso8601
  persist.call
end
puts JSON.generate(manifest.slice(:mode, :source_revision, :status, :error))
exit(manifest[:status] == 'PASS' ? 0 : 1)
