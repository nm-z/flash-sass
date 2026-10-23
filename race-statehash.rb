#!/usr/bin/env ruby
# Race localization, finite diagnostic: replay the baseline decode request RUNS times on the live core,
# each time after a one-token unrelated request that replaces the prefix snapshot (so every replay is
# fresh), and record the output tokens plus the engine's per-die, per-layer recurrent state hashes.
# Runs whose tokens match the baseline but whose hashes differ expose a perturbation that did not flip
# a token; the lowest layer whose hash differs bounds where it entered.
# Usage (inside gpuq): ruby race-statehash.rb [RUNS]
require 'socket'
require 'json'
require 'open3'
require 'timeout'

raise 'run inside gpuq' unless ENV['GPUQ_JOB_ID']
BASELINE_JOB = '63668679-0f08-49f2-b283-22969e66b4bf'
runs = Integer(ARGV.fetch(0, '30'))

def document(job, source)
  sql = "SELECT id FROM documents WHERE job_id='#{job}' AND kind='supplied' AND source='#{source}' ORDER BY id DESC LIMIT 1"
  out, err, st = Open3.capture3('gpuq', 'query', sql)
  raise err unless st.success?
  id = JSON.parse(out).fetch(0).fetch('id')
  out, err, st = Open3.capture3('gpuq', 'document', id.to_s, '--raw')
  raise err unless st.success?
  JSON.parse(out)
end

def put(label, value)
  _, err, st = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(value))
  raise err unless st.success?
end

def talk(commands, last)
  Timeout.timeout(900) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      commands.each { |c| s.write(c) }
      out = []
      while (line = s.gets)
        out << line
        break if line =~ last
      end
      out
    end
  end
end

base = document(BASELINE_JOB, 'goal-decode-0')
sent = base.fetch('wire').fetch('sent')
expected = base.fetch('wire').fetch('received').filter_map { |l| l.split[1].to_i if l.start_with?('tok ') }
config = talk(["config\n"], /\Aconfig /).last
results = []
runs.times do |i|
  deflect = ["stop 248046 4294967295\n", "sample 7 0 20 0.95\n", "snap 3\n", "chat 1 248045 #{1000 + i} 198 248046\n"]
  talk(deflect, /\A(done|error)/)
  out = talk(sent, /\A(done|error)/)
  raise "request failed: #{out.last}" unless out.last.start_with?('done ')
  toks = out.filter_map { |l| l.split[1].to_i if l.start_with?('tok ') }
  hashes = talk(["statehash\n"], /\Astatehash done/).grep(/\Astatehash die=/).map(&:chomp)
  first = (0...[toks.length, expected.length].max).find { |k| toks[k] != expected[k] }
  results << { run: i, same: toks == expected, first_diff: first, got: first && toks[first], done: out.last.chomp, hashes: hashes }
end
reference = results.find { |r| r[:same] }&.dig(:hashes)
summary = results.map do |r|
  diff = reference ? r[:hashes].each_index.reject { |k| r[:hashes][k] == reference[k] }.map { |k| r[:hashes][k][/die=\d+ (layer=\d+|ple)/] } : []
  layers = diff.filter_map { |s| s[/layer=(\d+)/, 1]&.to_i }
  { run: r[:run], same: r[:same], first_diff: r[:first_diff], got: r[:got], differing_states: diff.size,
    lowest_layer: layers.min, dies_at_lowest: diff.select { |s| s.end_with?("layer=#{layers.min}") }.map { |s| s[/die=(\d+)/, 1].to_i } }
end
put('race-statehash-results', { config: config, expected: expected, results: results })
put('race-statehash-summary', summary)
summary.each { |s| puts JSON.generate(s) }
