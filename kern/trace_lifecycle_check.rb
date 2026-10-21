#!/usr/bin/env ruby
# Real resident-core lifecycle regression. Raw wire evidence is saved first.
require 'socket'
require 'optparse'
require 'json'
require 'zlib'
require 'digest'
require 'securerandom'
require 'fileutils'
require 'open3'

module TraceChecks
  module_function
  def need(ok, message)
    raise message unless ok
  end
  def fields(line)
    line.split.drop(1).filter_map { |s| s.split('=', 2) if s.include?('=') }.to_h
  end
  def number(row, key)
    Integer(row.fetch(key), 10)
  end
  def ordered(row, keys)
    values = keys.map { |k| number(row, k) }
    need(values.first.positive? && values.each_cons(2).all? { |a,b| a <= b },
         "missing/nonmonotonic #{keys.join(' <= ')}: #{row}")
  end
  def validate_file(path, kind, meta, ple_disk)
    need(meta['schema'] == '2', "expected schema 2: #{meta}")
    count = number(meta, 'count'); first = number(meta, 'first')
    nph = kind == 'trace' ? number(meta, 'nph') : 1
    ctas = kind == 'trace' ? number(meta, 'ctas') : 1
    need(count.positive? && nph.positive? && [1,16].include?(ctas), 'bad dimensions')
    rows = 0; last_end = {}; cancel_checked = false
    Zlib::GzipReader.open(path) do |io|
      io.readline
      columns = io.readline.chomp.split(',')
      required = if kind == 'trace'
        %w[position phase cta flags start_ns wait_start_ns wait_end_ns body_start_ns body_end_ns end_ns ple_wait_start_ns ple_wait_end_ns publication_ns send_before_ns send_after_ns exchange_generation]
      else
        %w[position observed_ns callback_start_ns callback_end_ns ple_staging_start_ns ple_staging_end_ns ple_ready_ns ple_bytes ple_short_reads ple_read_start_ns ple_read_end_ns ple_ready_start_ns poll_start_ns poll_end_ns readback_start_ns readback_end_ns]
      end
      need((required - columns).empty? && columns.uniq == columns, 'missing/duplicate columns')
      terminated = false
      io.each_line do |line|
        if line == "#{kind}_end\n"
          terminated = true
          break
        end
        cells = line.chomp.split(',', -1)
        need(cells.length == columns.length, "wrong CSV width at #{rows}")
        row = columns.zip(cells).to_h; position = number(row, 'position')
        need(position == first + rows / (nph * ctas), "wrong position at #{rows}")
        if kind == 'trace'
          phase = number(row, 'phase'); cta = number(row, 'cta')
          need(phase == rows / ctas % nph && cta == rows % ctas, 'wrong phase/CTA')
          ordered(row, %w[start_ns wait_start_ns wait_end_ns body_start_ns body_end_ns end_ns])
          need(number(row,'start_ns') >= last_end.fetch(cta, 0), 'overlapping local phases')
          last_end[cta] = number(row, 'end_ns')
          if number(row, 'ple_wait_start_ns').positive?
            ordered(row, %w[body_start_ns ple_wait_start_ns ple_wait_end_ns body_end_ns])
          else
            need(number(row,'ple_wait_end_ns').zero?, 'PLE end without start')
          end
          if (number(row,'flags') & 1).positive?
            ordered(row, %w[body_end_ns send_before_ns send_after_ns end_ns])
          else
            need(number(row,'send_before_ns').zero? && number(row,'send_after_ns').zero?,
                 'send samples on nonsend phase')
          end
          publication = number(row, 'publication_ns')
          need(phase == nph - 1 ? publication >= number(row,'end_ns') : publication.zero?,
               'wrong publication sample')
          need(number(row,'exchange_generation') >= 0, 'negative generation')
        else
          ordered(row, %w[poll_start_ns poll_end_ns readback_start_ns readback_end_ns observed_ns callback_start_ns callback_end_ns])
          need(number(row,'observed_ns') == number(row,'readback_end_ns'), 'wrong observation boundary')
          staging = number(row,'ple_staging_start_ns')
          need(!ple_disk || staging.positive?, 'missing disk PLE staging samples')
          if staging.positive?
            ordered(row, %w[ple_staging_start_ns ple_read_start_ns ple_read_end_ns ple_staging_end_ns ple_ready_start_ns ple_ready_ns])
            need((0..1440).cover?(number(row,'ple_bytes')), 'invalid PLE bytes')
            need((0..16).cover?(number(row,'ple_short_reads')), 'invalid PLE short reads')
          else
            need(%w[ple_read_start_ns ple_read_end_ns ple_staging_end_ns ple_ready_start_ns ple_ready_ns ple_bytes ple_short_reads].all? { |k| number(row,k).zero? }, 'PLE fields without start')
          end
          if meta['abort_reason'] == 'callback' && number(meta,'abort_position') == position
            need(number(row,'callback_end_ns') <= number(meta,'abort_start_ns'),
                 'callback_end includes abort publication/synchronization')
            cancel_checked = true
          end
        end
        rows += 1
      end
      need(terminated && io.read.empty?, 'missing terminator or trailing data')
    end
    need(rows == count * nph * ctas, "wrong record count #{rows}")
    if kind == 'hosttrace'
      a = number(meta,'abort_start_ns'); b = number(meta,'abort_end_ns')
      if meta['aborted'] == '1'
        need(a.positive? && a <= b && %w[callback alive].include?(meta['abort_reason']), 'bad abort timing/reason')
      else
        need(a.zero? && b.zero? && number(meta,'abort_position') == -1 && meta['abort_reason'] == 'none',
             'abort fields on completed request')
      end
      need(number(meta,'request_poll_calls').positive? && number(meta,'request_readback_calls').positive?,
           'missing whole-request polling/readback counters')
    end
    {rows: rows, ctas: ctas, cancellation_callback_checked: cancel_checked}
  end
end

class Core
  def initialize(path, timeout)
    @socket = UNIXSocket.new(path); @timeout = timeout; @buffer = ''.b
  end
  def send(command)
    @socket.write(command + "\n")
  end
  def line
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + @timeout
    loop do
      if (i = @buffer.index("\n"))
        return @buffer.slice!(0, i + 1).chomp("\n")
      end
      left = deadline - Process.clock_gettime(Process::CLOCK_MONOTONIC)
      raise 'core response timeout' if left <= 0 || !IO.select([@socket], nil, nil, left)
      chunk = @socket.read_nonblock(65536, exception: false)
      next if chunk == :wait_readable
      raise 'core disconnected' unless chunk
      @buffer << chunk
    end
  end
  def close(abort: false)
    begin
      @socket.shutdown(Socket::SHUT_RDWR) if abort
    rescue SystemCallError, IOError
    ensure
      @socket.close unless @socket.closed?
    end
  end
  def stop_reading
    @socket.shutdown(Socket::SHUT_RD)
  end
end

class LifecycleCheck
  include TraceChecks
  def initialize(a)
    @a = a; @out = File.expand_path(a.fetch(:out))
    FileUtils.mkdir_p(File.dirname(@out)); Dir.mkdir(@out)
    @results = []; @last_request = -1; @run_number = 0; checkpoint
  end
  def log(s)
    File.open(File.join(@out,'protocol.txt'),'a') { |f| f.puts(s) }
  end
  def json(name, value)
    path = File.join(@out,name)
    File.write(path + '.tmp', JSON.pretty_generate(value) + "\n")
    File.rename(path + '.tmp', path)
  end
  def checkpoint(status = 'RUNNING', error = nil)
    json('results.json', {status: status, cases: @results, error: error})
  end
  def session
    core = Core.new(@a[:socket], @a[:timeout]); yield core
  ensure
    core&.close
  end
  def one(core, command)
    log('> ' + command); core.send(command); reply = core.line; log(reply); reply
  end
  def guard
    @run_number += 1
    return unless @a[:preflight]
    argv = [@a[:preflight], File.join(@out, format('preflight-%02d', @run_number))]
    log('preflight argv=' + JSON.generate(argv))
    need(system(*argv), 'GPU preflight refused the request')
  end
  def run(core, n, ids)
    guard; command = "chat #{n} #{ids.join(' ')}"; log('> ' + command); core.send(command)
    tokens = []
    loop do
      line = core.line; log(line)
      if line.start_with?('tok ')
        tokens << Integer(line.split.fetch(1),10)
      elsif line.start_with?('done ','error:')
        return [tokens,line]
      else
        raise 'unexpected response ' + line
      end
    end
  end
  def capture(core, name, kind, die)
    command = die.nil? ? kind : "#{kind} #{die}"
    log('> ' + command); core.send(command); first = core.line; log(first)
    file = "#{name}-#{kind}#{die.nil? ? '' : "-#{die}"}.txt.gz"
    Zlib::GzipWriter.open(File.join(@out,file)) do |io|
      io.puts(first)
      unless first.start_with?('error:')
        need(first.start_with?(kind + '_begin '), 'missing trace header')
        loop do
          line = core.line; io.puts(line)
          if line == kind + '_end'
            log(line); break
          end
        end
      end
    end
    [first, fields(first), file, kind, die]
  end
  def statehash(core, name)
    log('> statehash'); core.send('statehash'); first = core.line; log(first)
    file = name + '-statehash.txt'
    File.open(File.join(@out,file),'w') do |io|
      io.puts(first); need(first.start_with?('statehash positions='), 'missing statehash header')
      loop do
        line = core.line; io.puts(line); log(line); break if line == 'statehash done'
        need(line.start_with?('statehash '), 'bad statehash response')
      end
    end
    meta = fields(first)
    need(meta['positions'] == '7' && meta['paired_positions'] == '2', 'paired prefix was not executed')
    {header: first, raw_file: file}
  end
  def collect(core, name, status, aborted, ended, requested, completed = nil,
              error: false, first_position: 1, statehash_record: nil)
    captures = ((0...@dies).map { |d| ['trace',d] } + [['hosttrace',nil]]).map do |kind,die|
      capture(core,name,kind,die)
    end
    common = nil; validation = []
    captures.each do |line,meta,file,kind,die|
      need(line.start_with?('error:') == error, 'wrong trace availability: ' + line)
      need(meta['status'] == status && meta['aborted'] == aborted.to_s &&
           (ended.nil? || meta['ended'] == ended.to_s), 'wrong lifecycle: ' + line)
      need(%w[0 1].include?(meta['ended']), 'invalid ended flag: ' + line)
      need(number(meta,'requested') == requested, 'wrong requested positions: ' + line)
      actual = number(meta,'completed')
      need(completed.nil? ? (first_position + 1 <= actual && actual < requested) : actual == completed,
           'wrong completed positions: ' + line)
      shared = %w[request status aborted ended completed requested].to_h { |k| [k,meta.fetch(k)] }
      need(common.nil? || common == shared, 'inconsistent lifecycle across dies/host'); common = shared
      unless error
        need(number(meta,'first') == first_position, 'wrong first traced position')
        need(number(meta,'count') == [actual-first_position,@trace_cap].min, 'wrong bounded prefix')
        need(number(meta,'die') == die, 'wrong die') if kind == 'trace'
        validation << TraceChecks.validate_file(File.join(@out,file),kind,meta,@ple_disk).merge(file: file)
      end
    end
    request = number(common,'request'); need(request > @last_request, 'stale request ID'); @last_request = request
    @results << {case: name, headers: captures.map(&:first), files: captures.map { |x| x[2] },
                 lifecycle: common, validation: validation, statehash: statehash_record}
    checkpoint
  end
  def ok(core, command)
    need(one(core,command).start_with?('ok:'), 'command failed: ' + command)
  end
  def execute
    session do |core|
      config = one(core,'config'); cfg = fields(config)
      need(cfg['rev'] == @a[:source_sha], 'loaded engine revision differs from --source-sha')
      @trace_cap = number(cfg,'trace_tokens'); @dies = number(cfg,'dies'); @maxt = number(cfg,'maxt')
      @ple_disk = cfg['ple_disk'] == '1'; @pair_enabled = cfg['two_position'] == '1'
      need(@trace_cap >= 4 && @dies.positive? && @maxt >= 80, 'requires trace>=4, dies>=1, maxt>=80')
      @manifest = {hostname: Socket.gethostname, source_sha: @a[:source_sha], config: config,
        cubin_sha256: Digest::SHA256.file(cfg.fetch('cubin')).hexdigest, socket: @a[:socket],
        device_fault_injected: false, pair_check_requested: @a[:pair_check], pair_check_executed: false,
        checker_sha256: Digest::SHA256.file(__FILE__).hexdigest}
      json('manifest.json',@manifest)
      ok(core,'stop')
      single,done = run(core,0,[10,11])
      need(done.start_with?('done ') && single.length == 1, 'single-output request failed')
      collect(core,'single','completed',0,0,2,2)
      ok(core,'snap 1')
      tokens,done = run(core,3,[10,11])
      need(done.start_with?('done ') && tokens.length == 4 && done.split[6,2] == %w[0 1], 'normal snapshot failed')
      collect(core,'normal','completed',0,0,5,5)
      first_token = tokens.first
      [["stop #{first_token}",'stop-single'],["stop #{first_token.zero? ? 1 : 0} #{first_token}",'stop-multi']].each do |command,name|
        ok(core,command); stopped,done = run(core,3,[10,11])
        need(done.start_with?('done ') && stopped == [first_token] && done.split[6] == '1', 'stop/resume failed')
        collect(core,name,'completed',0,1,5,2)
      end
    end
    core = Core.new(@a[:socket],@a[:timeout])
    begin
      ok(core,'stop'); guard; log('> chat 40 10 11'); core.send('chat 40 10 11')
      first = core.line; log(first); need(first.start_with?('tok '), 'no token before disconnect')
      # Keep the write side open so the next token write fails before an
      # ordinary full disconnect is observed by the liveness callback.
      core.stop_reading
      sleep 1
    ensure
      core.close
    end
    log('< client disconnected after first token >')
    session do |core|
      # Host cancellation can race the device's normal stop bit. The aborted
      # status is authoritative; ended may be either value but must agree on
      # every die and the host trace.
      collect(core,'disconnect','aborted',1,nil,42)
      need(@results.last[:headers].last.include?('abort_reason=callback'),
           'disconnect did not exercise callback-triggered cancellation')
      need(@results.last[:validation].any? { |v| v[:cancellation_callback_checked] },
           'callback cancellation position was outside the validated trace window')
      tokens,error = run(core,@maxt,[10,11])
      need(tokens.empty? && error.start_with?('error:'), 'oversized request accepted')
      collect(core,'argument-error','argument_error',0,0,@maxt+2,0,error: true)
      missing = "/tmp/flash-trace-missing-#{SecureRandom.hex(16)}.bin"
      need(!File.exist?(missing), 'missing fixture exists'); ok(core,"img #{missing} 0")
      tokens,error = run(core,3,[10,11])
      need(tokens.empty? && error.start_with?('error:'), 'missing image accepted')
      collect(core,'image-error','image_error',0,0,5,0,error: true)
      ok(core,'stop'); tokens,done = run(core,3,[10,11])
      need(done.start_with?('done ') && tokens.length == 4, 'recovery failed')
      collect(core,'recovery','completed',0,0,5,5)
      if @a[:pair_check]
        need(@pair_enabled, '--pair-check requires two_position=1')
        ids = [20,21,22,23]; ok(core,'snap 1')
        tokens,done = run(core,3,ids)
        need(done.start_with?('done ') && tokens.length == 4 && done.split[6,2] == %w[0 1], 'paired snapshot failed')
        collect(core,'pair-first','completed',0,0,7,7,first_position: 3,statehash_record: statehash(core,'pair-first'))
        resumed,done = run(core,3,ids)
        need(done.start_with?('done ') && resumed == tokens && done.split[6] == '1', 'paired resume changed tokens')
        collect(core,'pair-resume','completed',0,0,7,7,first_position: 3,statehash_record: statehash(core,'pair-resume'))
        @manifest[:pair_check_executed] = true
      end
    end
    json('manifest.json',@manifest); checkpoint('PASS')
    puts "PASS #{@results.length} real lifecycle cases"
  rescue StandardError => e
    checkpoint('FAIL',"#{e.class}: #{e.message}"); raise
  end
end

if $PROGRAM_NAME == __FILE__
  a = {timeout: 120.0, pair_check: false}
  parser = OptionParser.new do |p|
    p.banner = 'Usage: trace_lifecycle_check.rb --socket PATH --out NEW_DIR --source-sha REV [--pair-check] [--preflight EXECUTABLE]'
    p.on('--socket PATH') { |v| a[:socket] = v }
    p.on('--out PATH') { |v| a[:out] = v }
    p.on('--source-sha REV') { |v| a[:source_sha] = v }
    p.on('--timeout SECONDS',Float) { |v| a[:timeout] = v }
    p.on('--pair-check') { a[:pair_check] = true }
    p.on('--preflight EXECUTABLE') { |v| a[:preflight] = File.expand_path(v) }
  end
  begin
    parser.parse!; TraceChecks.need(ARGV.empty?, 'unexpected arguments')
    %i[socket out source_sha].each { |k| TraceChecks.need(a[k], "--#{k.to_s.tr('_','-')} required") }
    TraceChecks.need(a[:timeout].finite? && a[:timeout].positive?, 'invalid timeout')
    LifecycleCheck.new(a).execute
  rescue StandardError => e
    warn "FAIL: #{e.class}: #{e.message}"; exit 1
  end
end
