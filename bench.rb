#!/usr/bin/env ruby
require 'socket'
require 'json'
require 'fileutils'
require 'timeout'
require 'time'
root, socket_path, label = ARGV
abort 'usage: bench.rb OUTPUT_DIRECTORY SOCKET LABEL' unless label
FileUtils.mkdir_p(root)
prompt = JSON.parse(File.read(File.join(__dir__, 'prompt-ids.json')))
raise 'invalid prompt IDs' unless prompt.is_a?(Array) && prompt.all? { |v| v.is_a?(Integer) && v >= 0 && v < 248320 }
raw = File.open(File.join(root, 'protocol.jsonl'), 'a'); raw.sync = true
clock = -> { Process.clock_gettime(Process::CLOCK_MONOTONIC) }
result = { label: label, started_utc: Time.now.utc.iso8601, runs: [] }
File.write(File.join(root, 'gpu-before.csv'), IO.popen(['nvidia-smi','--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.free','--format=csv'], &:read))
UNIXSocket.open(socket_path) do |s|
  write = ->(cmd) { raw.puts(JSON.generate(at: clock.call, direction: 'send', text: cmd)); s.write(cmd + "\n") }
  read = -> { line = s.gets or raise 'unexpected core EOF'; raw.puts(JSON.generate(at: clock.call, direction: 'receive', text: line)); raise line if line.start_with?('error'); line }
  command = ->(cmd) { write.call(cmd); read.call }
  result[:config_before] = command.call('config')
  command.call('stop 248046 -1')
  command.call('sample 7 0 20 0.95')
  command.call('snap 0')
  # A distinct one-position request invalidates any inherited prefix snapshot.
  write.call('gen 0 32')
  Timeout.timeout(120) do
    loop { break if read.call.start_with?('time:') }
  end
  cases = [['chat64', prompt, 4], ['repeat512', [32] * 512, 3], ['repeat2048', [32] * 2048, 2]]
  if ENV['BENCH_REPEATS']
    repetitions = Integer(ENV['BENCH_REPEATS'])
    raise 'BENCH_REPEATS must be 1..8' unless (1..8).cover?(repetitions)
    cases.each { |row| row[2] = repetitions }
  end
  cases = cases.select { |name,_,_| name == ENV['BENCH_CASE'] } if ENV['BENCH_CASE']
  cases.each do |name, ids, repetitions|
    repetitions.times do |rep|
      command.call('snap 0')
      tokens=[]; times=[]; done=nil
      start=clock.call
      Timeout.timeout(600) do
        write.call("chat 128 #{ids.join(' ')}")
        loop do
          line=read.call
          if line =~ /^tok (\d+)/
            tokens << Integer(Regexp.last_match(1)); times << clock.call
          elsif line.start_with?('done ')
            done=line; break
          end
        end
      end
      finish=clock.call
      fields=done.split
      raise "wrong prompt/count: #{done}" unless fields[1].to_i==tokens.length && fields[3].to_i==ids.length
      raise "unexpected snapshot: #{done}" unless fields[6].to_i==0
      profile=[]
      6.times do |die|
        write.call("prof #{die} all"); write.call('config')
        lines=[]
        loop do
          line=read.call
          break if line.start_with?('config ')
          lines << line
        end
        profile << {die: die, raw: lines.join}
      end
      intervals=tokens.length-1
      run={case: name, repetition: rep, warmup: rep==0, prompt_ids: ids, output_ids: tokens,
           done: done, wall_ms: (finish-start)*1000, first_observed_ms: (times.first-start)*1000,
           decode_intervals: intervals, decode_ms: intervals>0 ? (times.last-times.first)*1000 : nil,
           decode_tps: intervals>0 ? intervals/(times.last-times.first) : nil,
           engine_decode_tps: intervals>0 && fields[4].to_f>fields[5].to_f ? 1000.0*intervals/(fields[4].to_f-fields[5].to_f) : nil,
           engine_prefill_ms: fields[5].to_f, engine_run_ms: fields[4].to_f,
           token_arrival_ms: times.map { |t| (t-start)*1000 }, profiles: profile}
      result[:runs] << run
      File.write(File.join(root,'results.json'),JSON.pretty_generate(result)+"\n")
      puts JSON.generate(run.reject { |k,_| [:prompt_ids,:output_ids,:token_arrival_ms,:profiles].include?(k) })
      STDOUT.flush
    end
  end
  result[:config_after]=command.call('config')
end
result[:finished_utc]=Time.now.utc.iso8601
File.write(File.join(root,'results.json'),JSON.pretty_generate(result)+"\n")
File.write(File.join(root, 'gpu-after.csv'), IO.popen(['nvidia-smi','--query-gpu=index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.free','--format=csv'], &:read))
