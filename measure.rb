#!/usr/bin/env ruby
# Use the running HTTP binary and engine, recording their unmodified wire protocol.
require 'socket'
require 'net/http'
require 'json'
require 'digest'
require 'open3'
require 'timeout'

raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
def json_record(value)
  case value
  when String
    utf8 = value.dup.force_encoding(Encoding::UTF_8)
    utf8.valid_encoding? ? utf8 : {byte_encoding: 'base64', data: [value].pack('m0')}
  when Array
    value.map { |item| json_record(item) }
  when Hash
    value.transform_values { |item| json_record(item) }
  else
    value
  end
end
def put(label, value)
  _, err, status = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(json_record(value)))
  raise err unless status.success?
end
def put_raw(label, value)
  _, err, status = Open3.capture3('gpuq', 'put', label, stdin_data: value)
  raise err unless status.success?
end
def metric(name, unit, value)
  raise 'metric recording failed' unless system('gpuq', 'emit', name, unit, JSON.generate(value))
end
def baseline_document(job, source)
  raise 'invalid baseline job ID' unless job.match?(/\A[0-9a-f-]{36}\z/)
  query = "SELECT id FROM documents WHERE job_id='#{job}' AND kind='supplied' AND source='#{source}' ORDER BY id DESC LIMIT 1"
  out, err, status = Open3.capture3('gpuq','query',query)
  raise err unless status.success?
  id = JSON.parse(out).fetch(0).fetch('id')
  out, err, status = Open3.capture3('gpuq','document',id.to_s,'--raw')
  raise err unless status.success?
  JSON.parse(out)
end
def core(command)
  Timeout.timeout(30) { UNIXSocket.open('/tmp/flash.sock') { |s| s.puts(command); s.gets || raise('engine EOF') } }
end
def request(body, port)
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  response = Net::HTTP.start('127.0.0.1', port, open_timeout: 5, read_timeout: 910) do |http|
    http.post('/v1/chat/completions', JSON.generate(body), {'Content-Type'=>'application/json'})
  end
  {status: response.code.to_i, body: JSON.parse(response.body), client_ms: (Process.clock_gettime(Process::CLOCK_MONOTONIC)-start)*1000}
end

pid = Integer(ARGV.fetch(0, '753710'))
# the serving frontend restarts with a new pid (a power cycle, serve.sh flashd): fall back to the live flashd on /tmp/flash.sock
serving = ->(p) { (File.binread("/proc/#{p}/cmdline").split("\0") rescue []).then { |a| File.basename(a.first.to_s) == 'flashd' && a.include?('/tmp/flash.sock') } }
unless serving.(pid)
  pid = IO.popen(['pgrep', '-x', 'flashd'], &:read).split.map(&:to_i).find { |p| serving.(p) } or raise 'no serving flashd on /tmp/flash.sock'
end
port = Integer(ENV.fetch('MEASURE_PORT', '18092'))
socket_path = "/tmp/flash-goal-#{Process.pid}.sock"
server = UNIXServer.new(socket_path)
mode = :prepare
records = Queue.new
threads = []
proxy = Thread.new do
  loop do
    client = server.accept
    threads << Thread.new(client, mode) do |down, current_mode|
      sent = []; received = []; up = nil
      begin
        if current_mode == :prepare
          while (line = down.gets)
            sent << line
            if line.start_with?('info')
              down.puts('info 230000 4096')
            elsif line.start_with?('chat ')
              records << {sent: sent, received: [], dry: true}
              down.puts('error tokenize_only'); break
            else
              down.puts('ok')
            end
          end
        else
          up = UNIXSocket.new('/tmp/flash.sock')
          send_thread = Thread.new do
            while (line = down.gets)
              sent << line; up.write(line)
            end
            up.close_write rescue nil
          rescue IOError, SystemCallError
          end
          while (line = up.gets)
            received << line; down.write(line)
            break if line.start_with?('done ', 'error')
          end
          send_thread.kill; send_thread.join
          records << {sent: sent, received: received, dry: false} if sent.any? { |line| line.start_with?('chat ') }
        end
      rescue IOError, SystemCallError => e
        records << {sent: sent, received: received, error: e.message} if sent.any? { |line| line.start_with?('chat ') }
      ensure
        up&.close rescue nil
        down.close rescue nil
      end
    end
  end
rescue IOError, Errno::EBADF
end
frontend = nil
begin
  env = File.binread("/proc/#{pid}/environ").split("\0").filter_map do |s|
    k,v=s.split('=',2); [k,v] if k.start_with?('FLASHD_')
  end.to_h
  env['FLASHD_REQUEST_MAX_S']='900'
  argv = File.binread("/proc/#{pid}/cmdline").split("\0")
  binary = "/proc/#{pid}/exe"
  identity = {frontend_sha256: Digest::SHA256.file(binary).hexdigest, argv: argv, environment: env, config: core('config')}
  core_pid = IO.popen(['pgrep','-x','flashcore'], &:read).split.map(&:to_i).find { |p| File.binread("/proc/#{p}/environ").include?('FLASH_SOCK=/tmp/flash.sock') }
  if core_pid
    paths = File.readlines("/proc/#{core_pid}/maps").filter_map { |l| l.split.last if l.include?("/tmp/flashcore-#{core_pid}-") }.uniq
    identity[:engine_libraries] = paths.to_h { |p| [p, Digest::SHA256.file(p).hexdigest] }
  end
  put('goal-baseline-identity', identity)
  frontend = Process.spawn(env, binary, argv.fetch(1), socket_path, "127.0.0.1:#{port}")
  Timeout.timeout(30) do
    loop do
      begin
        TCPSocket.open('127.0.0.1',port,&:close); break
      rescue Errno::ECONNREFUSED
        sleep 0.05
      end
    end
  end
  short = 'Why does salt melt ice? Explain briefly in everyday language.'
  paragraph = 'A garden contains apple trees, pear trees, and plum trees. In spring the trees bloom, bees visit the flowers, and gardeners water the soil. In summer the fruit grows. In autumn the fruit is harvested and stored in wooden boxes. '
  long = paragraph * 80 + 'Which insects visited the flowers? Answer in one short sentence.'
  common = {stream:false, max_tokens:128, temperature:0, seed:7}
  make_body = ->(prompt) { common.merge(messages:[{role:'user',content:prompt}]) }
  tokenize = lambda do |prompt|
    request(make_body.call(prompt),port)
    rec = Timeout.timeout(5) { records.pop }
    rec[:sent].find { |l| l.start_with?('chat ') }.split.drop(2).map(&:to_i)
  end
  if ENV['BASELINE_JOB']
    fixture = baseline_document(ENV['BASELINE_JOB'],'goal-request-fixtures')
    short = fixture.fetch('short').fetch('messages').first.fetch('content')
    long = fixture.fetch('long').fetch('messages').first.fetch('content')
    common = fixture.fetch('short').reject { |k,_| k == 'messages' }.transform_keys(&:to_sym)
    ids = tokenize.call(long)
    raise 'baseline tokenization changed' unless ids == fixture.fetch('long_prompt_ids')
  else
    ids = tokenize.call(long)
    while ids.length > 8192
      long = long.sub(paragraph, ''); ids = tokenize.call(long)
    end
    long += ' tree' * (8192-ids.length)
    ids = tokenize.call(long)
  end
  raise "long prompt has #{ids.length}, expected 8192" unless ids.length == 8192
  put('goal-request-fixtures', {short:make_body.call(short), long:make_body.call(long), long_prompt_ids:ids})
  mode = :live
  consecutive = 0
  cases = [['decode',short],['prefill',long]]
  if ENV['MEASURE_KIND']
    raise 'invalid measurement kind' unless %w[decode prefill].include?(ENV['MEASURE_KIND'])
    cases = cases.select { |kind,_| kind == ENV['MEASURE_KIND'] }
  end
  Integer(ENV.fetch('MEASURE_PAIRS','1')).times do |pair|
    pair_pass = cases.length == 2
    cases.each do |kind,prompt|
      # Opt-in repeat freshness: a one-token request with an unrelated prompt replaces the
      # prefix snapshot, so the next measured request cannot resume from the previous one.
      if ENV['MEASURE_DEFLECT'] == '1' && (pair > 0 || kind != cases.first.first)
        request(common.merge(max_tokens:1, messages:[{role:'user',content:"Unrelated request #{pair} #{kind}: name a color."}]),port)
        Timeout.timeout(10) { records.pop }
      end
      response = request(make_body.call(prompt),port)
      wire = Timeout.timeout(10) { records.pop }
      result = {pair:pair,kind:kind,response:response,wire:wire}
      put("goal-#{kind}-#{pair}",result)
      raise "HTTP failure #{response[:status]}" unless response[:status] == 200
      timing = response[:body].fetch('timings')
      output_ids = wire[:received].filter_map { |l| l.split[1].to_i if l.start_with?('tok ') }
      raise 'missing completed wire response' unless wire[:received].last&.start_with?('done ')
      raise 'cached request is not fresh' unless timing.fetch('cache_n') == 0
      raise 'output token count mismatch' unless output_ids.length == timing.fetch('predicted_n')
      raise 'prompt length mismatch' if kind == 'prefill' && timing.fetch('prompt_n') != 8192
      metric("goal.#{kind}.tokens_s",'tokens/s',timing.fetch(kind == 'decode' ? 'predicted_per_second' : 'prompt_per_second'))
      metric("goal.#{kind}.output_ids",'token_ids',output_ids)
      metric("goal.#{kind}.cached_tokens",'count',timing.fetch('cache_n'))
      same = nil
      if ENV['BASELINE_JOB']
        baseline = baseline_document(ENV['BASELINE_JOB'],"goal-#{kind}-0")
        expected = baseline.fetch('wire').fetch('received').filter_map { |l| l.split[1].to_i if l.start_with?('tok ') }
        same = output_ids == expected
        metric("goal.#{kind}.same_output",'boolean',same)
        unless same
          first=(0...[expected.length,output_ids.length].max).find { |i| expected[i]!=output_ids[i] }
          put("goal-#{kind}-#{pair}-output-difference",{first_index:first,expected:expected[first],actual:output_ids[first],expected_count:expected.length,actual_count:output_ids.length})
        end
      end
      rate = timing.fetch(kind == 'decode' ? 'predicted_per_second' : 'prompt_per_second')
      pair_pass &&= same == true && rate >= (kind == 'decode' ? 140 : 1200)
      puts JSON.generate(pair:pair,kind:kind,timings:timing,output_sha256:Digest::SHA256.hexdigest(output_ids.pack('V*')))
      STDOUT.flush
      if ENV['CAPTURE_PREFILL_PROFILE'] == '1'
        profiles=Timeout.timeout(30) do
          UNIXSocket.open('/tmp/flash.sock') do |s|
            6.times.map do |die|
              s.puts("prefillprof #{die}")
              lines=[]
              while (line=s.gets)
                lines << line
                break if line=="prefillprof done\n"
                raise line if line.start_with?('error')
              end
              raise 'missing prompt profile' unless lines.first&.include?('valid=1 ') && lines.last=="prefillprof done\n"
              {die:die,lines:lines}
            end
          end
        end
        put("goal-prompt-batch-profile-#{kind}-#{pair}",profiles)
      end
      if ENV['CAPTURE_PROFILE'] == '1'
        profiles = Timeout.timeout(30) do
          UNIXSocket.open('/tmp/flash.sock') do |s|
            6.times.map do |die|
              s.puts("prof #{die} all"); s.puts('config')
              lines = []
              while (line = s.gets)
                break if line.start_with?('config ')
                raise line if line.start_with?('error')
                lines << line
              end
              {die:die,lines:lines}
            end
          end
        end
        put("goal-profile-#{kind}-#{pair}",profiles)
      end
      if ENV['CAPTURE_TRACE'] == '1'
        Timeout.timeout(60) do
          UNIXSocket.open('/tmp/flash.sock') do |s|
            [*(0...6).map { |d| "trace #{d}" }, 'hosttrace'].each do |cmd|
              s.puts(cmd); s.puts('config')
              lines = []
              while (line = s.gets)
                break if line.start_with?('config ')
                raise line if line.start_with?('error')
                lines << line
              end
              put_raw("goal-#{kind}-#{pair}-#{cmd.tr(' ','-')}.csv",lines.join)
            end
          end
        end
      end
      if ENV['CAPTURE_DIAG'] == '1'
        lines=Timeout.timeout(30) do
          UNIXSocket.open('/tmp/flash.sock') do |s|
            s.puts('diag'); s.puts('config')
            data=[]
            while (line=s.gets)
              break if line.start_with?('config ')
              raise line if line.start_with?('error')
              data << line
            end
            data.join
          end
        end
        put_raw("goal-#{kind}-#{pair}-diagnostic",lines)
      end
      raise "#{kind} output differs from baseline" if same == false && ENV['ALLOW_OUTPUT_DIFFERENCE'] != '1'
    end
    consecutive = pair_pass ? consecutive + 1 : 0
    metric('goal.consecutive_passing_pairs','count',consecutive)
  end
  final_config = core('config')
  put('goal-final-config', final_config)
  # hcr_groups and actq_bodies record the last program build, not settings.
  # Keep it in the evidence while comparing every actual setting unchanged.
  settings = ->(line) { line.split.reject { |word| word.start_with?('hcr_groups=','actq_bodies=','rms_bodies=') } }
  raise 'engine configuration changed during measurement' unless settings.call(final_config) == settings.call(identity[:config])
  raise 'three passing pairs not achieved' if ENV['CHECK_TARGETS'] == '1' && consecutive < 3
ensure
  if frontend
    Process.kill('TERM',frontend) rescue nil
    Process.wait(frontend) rescue nil
  end
  server.close rescue nil
  proxy.kill; proxy.join
  threads.each { |t| t.kill; t.join }
  File.unlink(socket_path) if File.socket?(socket_path)
end
