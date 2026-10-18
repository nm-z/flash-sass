#!/usr/bin/env ruby
# Controlled ABBA comparison against the actual flashcore line protocol.
require 'base64'
require 'digest'
require 'fileutils'
require 'json'
require 'open3'
require 'optparse'
require 'socket'
require 'time'
require 'zlib'

module TraceValidation
  class Error < StandardError; end
  class Busy < Error; end
  ABBA = %w[ordinary traced traced ordinary].freeze
  IMAGE_FIELDS = %w[cubin cubin_fnv trace_tokens].freeze
  def self.ns
    Process.clock_gettime(Process::CLOCK_MONOTONIC,:nanosecond)
  end
  def self.save(path,value)
    File.write(path,JSON.pretty_generate(value) + "\n")
  end
  def self.stats(values)
    a = values.sort; n = a.length; return nil if n.zero?
    mean = a.sum/n.to_f
    {n: n,mean: mean,median: n.odd? ? a[n/2] : (a[n/2-1]+a[n/2])/2.0,
     min: a.first,max: a.last,sample_sd: n > 1 ? Math.sqrt(a.sum { |v| (v-mean)**2 }/(n-1)) : nil}
  end
  def self.fields(line)
    raise Error,'expected config' unless line.start_with?('config ')
    pairs = line.split.drop(1).map { |s| s.split('=',2) }
    raise Error,'malformed config' unless pairs.all? { |p| p.length == 2 } && pairs.map(&:first).uniq.length == pairs.length
    pairs.to_h
  end
  class Wire
    def initialize(socket,prefix,timeout,compressed: false)
      @socket = UNIXSocket.new(socket); @timeout = timeout; @buffer = ''.b; @lines = []
      @tx = File.open(prefix + '.request.txt','w')
      @rx = compressed ? Zlib::GzipWriter.open(prefix + '.raw.gz') : File.open(prefix + '.response.txt','wb')
      @events = compressed ? nil : File.open(prefix + '.events.jsonl','w')
    end
    def send_line(text)
      bytes = text + "\n"; @socket.write(bytes); @tx.write(bytes); @tx.flush
      @events&.puts(JSON.generate(direction: 'send',monotonic_ns: TraceValidation.ns,base64: Base64.strict_encode64(bytes)))
    end
    def line
      deadline = TraceValidation.ns + (@timeout*1e9).to_i
      loop do
        return @lines.shift unless @lines.empty?
        left = (deadline-TraceValidation.ns)/1e9
        raise Error,'protocol timeout' if left <= 0 || !IO.select([@socket],nil,nil,left)
        bytes = @socket.read_nonblock(65536,exception: false)
        next if bytes == :wait_readable
        raise Error,'socket closed before response terminator' unless bytes
        at = TraceValidation.ns; @rx.write(bytes)
        @events&.puts(JSON.generate(direction: 'receive',monotonic_ns: at,base64: Base64.strict_encode64(bytes)))
        @buffer << bytes
        while (i = @buffer.index("\n"))
          @lines << [@buffer.slice!(0,i+1),at]
        end
      end
    end
    def close
      [@socket,@tx,@rx,@events].compact.each { |io| io.close unless io.closed? }
    end
  end
  class Runner
    def initialize(config_path,output=nil)
      @config_path = File.realpath(config_path); @base = File.dirname(@config_path)
      @bytes = File.binread(@config_path); @c = JSON.parse(@bytes)
      @output = output && File.expand_path(output); @rows = []; @tokens = nil; @common = nil
      %w[socket modes restore_argv expected_common identity_paths allowed_holders].each { |k| @c.fetch(k) }
      %w[library source_manifest weights_manifest ordinary_cubin traced_cubin].each { |k| @c['identity_paths'].fetch(k) }
      %w[ordinary traced].each do |mode|
        %w[cubin cubin_fnv trace_tokens prepare_argv].each { |k| @c['modes'].fetch(mode).fetch(k) }
      end
      @c['expected_common'].fetch('rev')
      raise Error,'need at least two output tokens' unless Integer(@c.fetch('new',7)) >= 1
      raise Error,'bad repeat/warmup count' unless Integer(@c.fetch('repeats',3)) > 0 && Integer(@c.fetch('warmups',2)) >= 0
      @c.fetch('prompt_ids',[10,11]).each { |n| raise Error,'invalid prompt ID' unless n.is_a?(Integer) && n >= 0 }
    end
    def path(p)
      File.expand_path(p,@base)
    end
    def identities
      @c['identity_paths'].transform_values do |p|
        f = File.realpath(path(p))
        {path: f,bytes: File.size(f),sha256: Digest::SHA256.file(f).hexdigest}
      end
    end
    def capture(argv)
      out,err,s = Open3.capture3(*argv)
      {argv: argv,stdout: out,stderr: err,status: s.exitstatus,success: s.success?}
    end
    def snapshot(dir,label,guard: false)
      d = {
        utc: Time.now.utc.iso8601(6),
        nvidia_smi: capture(['nvidia-smi']),
        gpu: capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']),
        utilization: capture(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader,nounits']),
        compute: capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
        ps: capture(['ps','-eLo','pid=,tid=,stat=,comm='])
      }
      TraceValidation.save(File.join(dir,label+'.json'),d)
      return unless guard
      %i[nvidia_smi gpu utilization compute ps].each { |k| raise Error,"#{k} query failed" unless d[k][:success] }
      allowed = @c['allowed_holders'].to_h do |h|
        pid = Integer(h.fetch('pid')); hash = Digest::SHA256.file("/proc/#{pid}/cmdline").hexdigest
        raise Error,"resident PID #{pid} changed identity" unless hash == h.fetch('cmdline_sha256')
        [pid,true]
      end
      d[:compute][:stdout].each_line do |line|
        next if line.strip.empty?
        pid = Integer(line.split(',').first.strip)
        raise Busy,"unrecognized GPU process #{pid}" unless allowed[pid]
      end
      d[:utilization][:stdout].each_line do |line|
        next if line.strip.empty?
        f = line.split(',').map(&:strip)
        raise Busy,"GPU #{f[0]} busy: #{f[2]}%" unless Float(f.fetch(2)).zero?
      end
      pattern = Regexp.new(@c.fetch('busy_process_pattern','flashcore|_test$|bench|trace_lifecycle'))
      d[:ps][:stdout].each_line do |line|
        pid,tid,state,comm = line.split
        next if Integer(pid) == Process.pid
        raise Busy,"active #{comm} PID #{pid} TID #{tid}" if state.start_with?('R') && pattern.match?(comm)
      end
    end
    def idle!(dir,label)
      FileUtils.mkdir_p(dir); started = TraceValidation.ns; attempt = 0
      loop do
        attempt += 1
        begin
          snapshot(dir,format('%s-%03d',label,attempt),guard: true); return
        rescue Busy => e
          elapsed = (TraceValidation.ns-started)/1e9
          raise Error,"GPU remains busy: #{e.message}" if elapsed >= Float(@c.fetch('busy_wait_seconds',300))
          warn "Waiting for GPU: #{e.message}"; sleep 5
        end
      end
    end
    def helper(argv,dir,label)
      raise Error,'helper must be argument array' unless argv.is_a?(Array) && argv.all? { |a| a.is_a?(String) }
      out,err,s = Open3.capture3({'FLASH_TRACE_VALIDATION_LOCK_HELD'=>'1'},*argv,chdir: @base)
      File.write(File.join(dir,label+'.stdout'),out); File.write(File.join(dir,label+'.stderr'),err)
      TraceValidation.save(File.join(dir,label+'.json'),{argv: argv,status: s.exitstatus})
      raise Error,"#{label} failed: #{s.exitstatus}" unless s.success?
    end
    def wire(dir,label,compressed: false)
      w = Wire.new(path(@c['socket']),File.join(dir,label),Float(@c.fetch('timeout_seconds',120)),compressed: compressed)
      yield w
    ensure
      w&.close
    end
    def config!(mode,dir,label)
      line = nil
      wire(dir,label) { |w| w.send_line('config'); line = w.line[0].strip }
      f = TraceValidation.fields(line)
      expected = @c['expected_common'].transform_values(&:to_s)
      m = @c['modes'].fetch(mode)
      expected.merge!('cubin'=>path(m.fetch('cubin')),'cubin_fnv'=>m.fetch('cubin_fnv'),'trace_tokens'=>m.fetch('trace_tokens').to_s)
      expected.each { |k,v| raise Error,"config #{k}=#{f[k].inspect}, expected #{v.inspect}" unless f[k] == v }
      common = f.reject { |k,_| IMAGE_FIELDS.include?(k) }; @common ||= common
      raise Error,'common runtime config changed' unless common == @common
      TraceValidation.save(File.join(dir,label+'.json'),f); f
    end
    def trace(dir,kind,die,enabled,expected)
      command = die ? "#{kind} #{die}" : kind; label = die ? "#{kind}-#{die}" : kind
      header = nil
      wire(dir,label,compressed: true) do |w|
        w.send_line(command); header = w.line[0]
        if enabled
          raise Error,"missing #{kind} header: #{header}" unless header.start_with?(kind+'_begin ')
          loop do
            line = w.line[0]; break if line.strip == kind+'_end'
          end
        else
          raise Error,'ordinary image unexpectedly returned a trace' unless header.start_with?('error:')
        end
      end
      meta = header.split.drop(1).filter_map { |s| s.split('=',2) if s.include?('=') }.to_h
      expected.each { |k,v| raise Error,"trace lifecycle #{k}=#{meta[k]}" unless meta[k] == v.to_s }
      if enabled
        raise Error,'trace schema mismatch' unless meta['schema'] == '2'
        raise Error,'wrong trace die' if die && meta['die'] != die.to_s
      end
      {file: label+'.raw.gz',header: header.strip,metadata: meta,
       sha256: Digest::SHA256.file(File.join(dir,label+'.raw.gz')).hexdigest}
    end
    def sample(slot,kind,number,slot_dir)
      dir = File.join(slot_dir,format('%s-%02d',kind,number)); FileUtils.mkdir_p(dir)
      idle!(dir,'before-chat'); runtime = config!(slot['mode'],dir,'config-before')
      prompt = @c.fetch('prompt_ids',[10,11]); n = Integer(@c.fetch('new',7)); tokens = []; arrivals = []; done = nil
      started = nil; finished = nil
      wire(dir,'chat') do |w|
        started = TraceValidation.ns; w.send_line("chat #{n} #{prompt.join(' ')}")
        loop do
          line,at = w.line
          if (m = /\Atok (\d+)\n\z/.match(line))
            tokens << Integer(m[1]); arrivals << at
          elsif line.start_with?('done ')
            done = line.split; finished = at; break
          else
            raise Error,"unexpected chat response: #{line.inspect}"
          end
        end
      end
      raise Error,'incomplete/ended/wrong-length request' unless done.length == 8 && tokens.length == n+1 &&
        done[1..3] == [(n+1).to_s,'0',prompt.length.to_s] && done[6..7] == %w[0 0]
      @tokens ||= tokens; raise Error,'generated tokens differ across modes/repeats' unless tokens == @tokens
      row = slot.merge('kind'=>kind,'sample'=>number,'directory'=>dir,'config'=>runtime,'tokens'=>tokens,
        'token_arrival_ns'=>arrivals,'started_ns'=>started,'finished_ns'=>finished,
        'wall_ns'=>finished-started,'first_token_ns'=>arrivals.first-started,
        'steady_decode_ns_per_token'=>(arrivals.last-arrivals.first).to_f/n,
        'done'=>done.join(' '),'run_ms_rounded'=>Float(done[4]),'first_ms_rounded'=>Float(done[5]))
      if kind == 'measured'
        expected = {status: 'completed',aborted: 0,ended: 0,requested: prompt.length+n,completed: prompt.length+n}.transform_keys(&:to_s)
        enabled = slot['mode'] == 'traced'
        traces = (0...Integer(runtime.fetch('dies'))).map { |d| trace(dir,'trace',d,enabled,expected) }
        traces << trace(dir,'hosttrace',nil,enabled,expected)
        raise Error,'different requests across raw traces' unless traces.map { |r| r[:metadata].fetch('request') }.uniq.length == 1
        row['traces'] = traces
      end
      config!(slot['mode'],dir,'config-after'); snapshot(dir,'after-chat')
      TraceValidation.save(File.join(dir,'sample.json'),row)
      File.open(File.join(@output,'samples.jsonl'),'a') { |f| f.puts(JSON.generate(row)) }
      @rows << row
      puts "#{kind} slot=#{slot['slot']} mode=#{slot['mode']} wall_ns=#{row['wall_ns']}"
      $stdout.flush
    end
    def plan
      Integer(@c.fetch('repeats',3)).times.flat_map do |b|
        ABBA.each_with_index.map { |mode,i| {'block'=>b+1,'position'=>i+1,'slot'=>b*4+i+1,'mode'=>mode} }
      end
    end
    def summary
      measured = @rows.select { |r| r['kind'] == 'measured' }
      %w[wall_ns first_token_ns steady_decode_ns_per_token run_ms_rounded].to_h do |metric|
        modes = %w[ordinary traced].to_h { |m| [m,TraceValidation.stats(measured.select { |r| r['mode']==m }.map { |r| r.fetch(metric) })] }
        blocks = measured.map { |r| r['block'] }.uniq.map do |b|
          means = %w[ordinary traced].to_h do |m|
            a = measured.select { |r| r['block']==b && r['mode']==m }.map { |r| r.fetch(metric) }
            [m,a.sum/a.length.to_f]
          end
          raise Error,'nonpositive baseline' unless means['ordinary'].positive?
          {block: b,means: means,overhead_percent: 100*(means['traced']/means['ordinary']-1)}
        end
        [metric,{modes: modes,abba_blocks: blocks,overhead_percent: TraceValidation.stats(blocks.map { |b| b[:overhead_percent] })}]
      end
    end
    def run
      raise Error,'new output directory required' unless @output && !File.exist?(@output)
      FileUtils.mkdir_p(@output); File.write(File.join(@output,'config.json'),@bytes)
      original_ids = identities
      TraceValidation.save(File.join(@output,'metadata.json'),{utc: Time.now.utc.iso8601(6),hostname: Socket.gethostname,
        ruby: RUBY_DESCRIPTION,pid: Process.pid,identities: original_ids,plan: plan,runner_sha256: Digest::SHA256.file(__FILE__).hexdigest})
      File.open(path(@c.fetch('lock_path','/tmp/flash-sass-engine-validation.lock')),File::RDWR|File::CREAT,0600) do |lock|
        raise Busy,'validation lock held' unless lock.flock(File::LOCK_EX|File::LOCK_NB)
        failure = nil; restore = false
        begin
          plan.each do |slot|
            dir = File.join(@output,format('slot-%02d-%s',slot['slot'],slot['mode'])); FileUtils.mkdir_p(dir)
            idle!(dir,'before-prepare')
            raise Error,'input identities changed' unless identities == original_ids
            restore = true; helper(@c['modes'][slot['mode']]['prepare_argv'],dir,'prepare')
            config!(slot['mode'],dir,'config-prepared')
            Integer(@c.fetch('warmups',2)).times { |i| sample(slot,'warmup',i+1,dir) }
            sample(slot,'measured',1,dir)
          end
          raise Error,'input identities changed' unless identities == original_ids
          TraceValidation.save(File.join(@output,'summary.json'),summary)
        rescue StandardError,Interrupt => e
          failure = e; TraceValidation.save(File.join(@output,'failure.json'),{class: e.class.name,message: e.message})
        ensure
          if restore
            begin
              helper(@c['restore_argv'],@output,'restore')
            rescue StandardError,Interrupt => e
              failure ||= e; TraceValidation.save(File.join(@output,'restore-failure.json'),{message: e.message})
            end
          end
          snapshot(@output,'after')
          lock.flock(File::LOCK_UN)
        end
        raise failure if failure
      end
      puts JSON.pretty_generate(summary)
    end
  end
end

if $PROGRAM_NAME == __FILE__
  opts = {}
  OptionParser.new do |p|
    p.on('--config FILE') { |v| opts[:config] = v }
    p.on('--output DIR') { |v| opts[:output] = v }
    p.on('--preflight DIR') { |v| opts[:preflight] = v }
    p.on('--plan') { opts[:plan] = true }
    p.on('--check-config') { opts[:check] = true }
  end.parse!
  begin
    raise TraceValidation::Error,'unexpected arguments' unless ARGV.empty?
    runner = TraceValidation::Runner.new(opts.fetch(:config),opts[:output])
    if opts[:preflight]
      runner.idle!(File.expand_path(opts[:preflight]),'gpu')
    elsif opts[:plan]
      puts JSON.pretty_generate(runner.plan)
    elsif opts[:check]
      puts 'configuration structure PASS; no GPU or socket accessed'
    else
      runner.run
    end
  rescue StandardError,Interrupt => e
    warn "#{e.class}: #{e.message}"; exit 1
  end
end
