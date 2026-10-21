#!/usr/bin/env ruby
# frozen_string_literal: true
# Offline comparison of measured placements. No unmeasured shares are proposed.
require "json"
require "optparse"
require "time"

module DieBalance
  SCHEMA = "flash-sass-die-balance-v1"
  REQUEST = {
    "messages" => [{"role" => "user", "content" => "What is the capital of France? Answer in one word."}],
    "max_tokens" => 40, "temperature" => 0
  }.freeze
  FLOAT = '[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
  REQUIRED_CONFIG = %w[rev kv_host ple_disk maxt layers nblk dies attn_dies split attn_map img_rows two_position prefill_chunk cubin cubin_fnv].freeze
  VARIABLE_CONFIG = %w[split attn_map attn_dies cubin].freeze
  LIMITATION = "The last-token profiler omits phase 0; parallel die spans are never added."
  class Invalid < StandardError; end
  class UniqueHash < Hash
    def []=(key, value)
      raise Invalid, "duplicate JSON key #{key.inspect}" if key?(key)
      super
    end
  end
  module_function

  def check(condition, message)
    raise Invalid, message unless condition
  end

  def text!(value, name)
    check(value.is_a?(String) && !value.strip.empty?, "#{name}: missing/empty string")
    value
  end

  def number!(value, name, positive: false)
    check(value.is_a?(Numeric) && value.finite?, "#{name}: nonfinite/nonnumeric")
    check(positive ? value > 0 : value >= 0, "#{name}: invalid sign")
    value
  end

  def integer!(value, name, positive: false)
    check(value.is_a?(Integer), "#{name}: not an integer")
    number!(value, name, positive: positive)
  end

  def ids!(value, name)
    check(value.is_a?(Array) && !value.empty?, "#{name}: missing token IDs")
    value.each { |id| integer!(id, name); check(id < 2**32, "#{name}: token outside uint32") }
    value
  end

  def integers(value, name)
    check(value.is_a?(String) && value.match?(/\A\d+(?:,\d+)*\z/), "#{name}: malformed integer list")
    value.split(",").map(&:to_i)
  end

  def config!(raw)
    fields = text!(raw, "config").strip.split(/\s+/)
    check(fields.shift == "config", "config: missing prefix")
    parsed = {}
    fields.each do |field|
      key, value = field.split("=", 2)
      check(value && !value.empty? && !parsed.key?(key), "config: malformed/duplicate field")
      parsed[key] = value
    end
    check((REQUIRED_CONFIG - parsed.keys).empty?, "config: missing required fields")
    check(parsed["rev"].match?(/\A[0-9a-f]{7,40}(?:-dirty)?\z/i), "config: malformed source revision")
    check(parsed["cubin_fnv"].match?(/\A[0-9a-f]{16}\z/i), "config: missing cubin identity")
    check(parsed["layers"] == "48/48" && parsed["dies"] == "6" && parsed["nblk"] == "512",
          "config: requires six dies, 48/48 layers, nblk=512")
    check(parsed["maxt"].match?(/\A[1-9]\d*\z/), "config: invalid maxt")
    %w[kv_host ple_disk two_position].each { |key| check(%w[0 1].include?(parsed[key]), "config: invalid #{key}") }
    %w[img_rows prefill_chunk].each { |key| check(parsed[key].match?(/\A\d+\z/), "config: invalid #{key}") }
    shares = integers(parsed["split"], "split")
    attn = integers(parsed["attn_map"], "attn_map")
    dies = integers(parsed["attn_dies"], "attn_dies")
    check(shares.length == 6 && shares.all?(&:positive?), "split: require six positive shares")
    check(attn.length == 12 && attn.all? { |die| (0...6).cover?(die) }, "attn_map: require twelve valid dies")
    check(dies.uniq.length == dies.length && dies.sort == attn.uniq.sort, "attn_dies: inconsistent map")
    parsed
  end

  def profile!(entry)
    check(entry.is_a?(Hash), "profile: expected object")
    die = integer!(entry["die"], "profile die")
    check((0...6).cover?(die), "profile: invalid die")
    lines = text!(entry["raw"], "profile raw").lines.map(&:strip).reject(&:empty?)
    header = /\Aprof:\s+(\d+)\s+phases,\s+(#{FLOAT})\s+ms between the first and last timestamp\z/.match(lines.shift.to_s)
    check(header, "die #{die}: malformed profile header")
    phases = header[1].to_i
    total = number!(Float(header[2]), "die #{die} total_ms", positive: true)
    check(phases > 1 && lines.last.to_s.start_with?("longest:"), "die #{die}: incomplete profile")
    lines.pop
    operations = {}
    row = /\A([A-Za-z0-9_.:+-]+)\s+(\d+)\s+phases\s+(#{FLOAT})\s+us\s+\(\s*(#{FLOAT})\s+us each\)\z/
    lines.each do |line|
      match = row.match(line)
      check(match, "die #{die}: malformed operation row")
      name = match[1]
      check(!operations.key?(name), "die #{die}: duplicate operation #{name}")
      count = match[2].to_i
      check(count.positive?, "die #{die}: zero operation phase count")
      us = number!(Float(match[3]), "die #{die} #{name} us")
      average = number!(Float(match[4]), "die #{die} #{name} average")
      check((average - us / count).abs <= 0.050001 + 0.050001 / count, "die #{die}: inconsistent operation average")
      operations[name] = {count: count, us: us}
    end
    check(operations.key?("actq256"), "die #{die}: missing actq256")
    check(operations.values.sum { |op| op[:count] } == phases - 1, "die #{die}: incomplete phase counts")
    sum_ms = operations.values.sum { |op| op[:us] } / 1000.0
    check((sum_ms - total).abs <= 0.005001 + operations.length * 0.000051,
          "die #{die}: operation times do not reconcile to header")
    {die: die, total_ms: total, phases: phases, actq256_us: operations["actq256"][:us],
     counts: operations.transform_values { |op| op[:count] }}
  end

  def manifest!(data)
    manifest = data["manifest"]
    check(manifest.is_a?(Hash), "manifest: missing object")
    %w[frontend_sha256 core_exe_sha256 engine_sha256].each do |name|
      check(manifest[name].is_a?(String) && manifest[name].match?(/\A[0-9a-f]{64}\z/i), "manifest: invalid #{name}")
    end
    env = manifest["frontend_env"]
    check(env.is_a?(Hash) && env.all? { |key, value| key.is_a?(String) && value.is_a?(String) }, "manifest: malformed frontend_env")
    # This is the collector checkout, not the runtime engine source revision.
    check(text!(manifest["source_revision"], "collector source revision").match?(/\A[0-9a-f]{40}\z/i),
          "manifest: malformed collector revision")
    model = manifest["model"]
    check(model.is_a?(Hash) && model["identity_kind"] == "stat", "manifest: missing model stat identity")
    %w[path realpath].each { |name| check(text!(model[name], "model #{name}").start_with?("/"), "model: path must be absolute") }
    %w[device inode size mtime_ns].each { |name| integer!(model[name], "model #{name}", positive: name != "device") }
    hardware = manifest["hardware"]
    check(hardware.is_a?(Hash), "manifest: missing hardware")
    %w[identity pci_link_snapshot].each { |name| text!(hardware[name], "hardware #{name}") }
    manifest
  end

  def run!(run, index, raw_config)
    check(run.is_a?(Hash) && run["index"] == index, "run #{index}: invalid/out-of-order index")
    check(run["warmup"] == index.zero?, "run #{index}: incorrect warmup flag")
    check(run["config"] == raw_config, "run #{index}: runtime config changed")
    input = ids!(run["input_tokens"], "run #{index} input")
    output = ids!(run["output_tokens"], "run #{index} output")
    check((2..40).cover?(output.length), "run #{index}: require 2..40 output IDs")
    response = run["response"]
    check(response.is_a?(Hash) && !response.key?("error"), "run #{index}: invalid HTTP response")
    usage, timings = response.values_at("usage", "timings")
    check(usage.is_a?(Hash) && timings.is_a?(Hash), "run #{index}: missing usage/timings")
    check(usage["prompt_tokens"] == input.length, "run #{index}: prompt count differs from IDs")
    check(usage["completion_tokens"] == output.length && timings["predicted_n"] == output.length, "run #{index}: output count differs from IDs")
    rate = number!(timings["predicted_per_second"], "run #{index} rate", positive: true)
    decode_ms = number!(timings["decode_ms"], "run #{index} decode_ms", positive: true)
    check(timings["decode_intervals"] == output.length - 1, "run #{index}: decode intervals differ")
    low = (output.length - 1) * 1000.0 / (decode_ms + 0.05) - 0.005001
    high = decode_ms > 0.05 ? (output.length - 1) * 1000.0 / (decode_ms - 0.05) + 0.005001 : Float::INFINITY
    check(rate >= low && rate <= high, "run #{index}: inconsistent decode timing/rate")
    done = text!(run["done"], "run #{index} done").split
    check(done.length == 8 && done[0] == "done" &&
          [1, 2, 3, 6, 7].all? { |i| done[i].match?(/\A\d+\z/) } &&
          [4, 5].all? { |i| done[i].match?(/\A#{FLOAT}\z/) }, "run #{index}: malformed done")
    check(done[1].to_i == output.length && done[3].to_i == input.length && [0, 1].include?(done[2].to_i),
          "run #{index}: done counts/stop flag differ")
    run_ms = number!(Float(done[4]), "run #{index} raw run_ms", positive: true)
    first_ms = number!(Float(done[5]), "run #{index} raw first_ms")
    check(run_ms > first_ms && (run_ms - first_ms - decode_ms).abs <= 0.150001,
          "run #{index}: raw and HTTP decode times differ")
    profiles = run["profiles"]
    check(profiles.is_a?(Array) && profiles.length == 6, "run #{index}: require six profiles")
    parsed = profiles.map { |entry| profile!(entry) }.sort_by { |entry| entry[:die] }
    check(parsed.map { |entry| entry[:die] } == (0...6).to_a, "run #{index}: duplicate/missing die")
    actq = parsed.map { |entry| entry[:actq256_us] }
    {input: input, output: output, rate: rate, profiles: parsed, spread_ms: (actq.max - actq.min) / 1000.0,
     max_total_ms: parsed.map { |entry| entry[:total_ms] }.max,
     signature: parsed.map { |entry| [entry[:phases], entry[:counts]] }}
  end

  def load_capture(path)
    data = JSON.parse(File.read(path), object_class: UniqueHash)
    check(data.is_a?(Hash) && data["schema"] == SCHEMA, "unsupported capture schema")
    %w[label created_at hostname].each { |key| text!(data[key], key) }
    Time.iso8601(data["created_at"])
    check(data["request"] == REQUEST, "request differs from exact linkmatrix protocol")
    check(data["warmup_requests"] == 1, "require exactly one warmup")
    manifest = manifest!(data)
    raw_config = manifest["config_before"]
    config = config!(raw_config)
    check(manifest["config_after"] == raw_config, "runtime config changed during capture")
    runs = data["runs"]
    check(runs.is_a?(Array) && runs.length >= 4, "require warmup and >=3 measured requests")
    parsed = runs.each_with_index.map { |run, index| run!(run, index, raw_config) }
    reference = parsed.first
    parsed.each_with_index do |run, index|
      check(run[:input] == reference[:input] && run[:output] == reference[:output], "run #{index}: token IDs changed within capture")
      check(run[:signature] == reference[:signature], "run #{index}: phase counts changed within capture")
    end
    measured = parsed.drop(1)
    {path: path, label: data["label"], hostname: data["hostname"], config: config, manifest: manifest,
     input: reference[:input], output: reference[:output], runs: parsed, repetitions: measured.length,
     objective_ms: measured.map { |run| run[:max_total_ms] }.max,
     worst_spread_ms: measured.map { |run| run[:spread_ms] }.max,
     mean_decode_tok_s: measured.sum { |run| run[:rate] } / measured.length,
     per_die_max_total_ms: (0...6).map { |die| measured.map { |run| run[:profiles][die][:total_ms] }.max },
     failures: [], valid: true}
  rescue Invalid, JSON::ParserError, SystemCallError, ArgumentError, TypeError => error
    {path: path, label: File.basename(path), valid: false, failures: [error.message]}
  end

  def gate_failures(capture, baseline)
    failures = []
    failures << "actq256 worst per-run spread exceeds 0.5 ms" if capture[:worst_spread_ms] > 0.5 + 1e-9
    failures << "arithmetic mean decode is below 35.9 tok/s" if capture[:mean_decode_tok_s] < 35.9 - 1e-9
    failures << "input/output token IDs differ from baseline" unless capture[:input] == baseline[:input] && capture[:output] == baseline[:output]
    stable = ->(config) { config.reject { |key, _| VARIABLE_CONFIG.include?(key) } }
    failures << "runtime differs beyond placement/cubin pathname" unless stable.call(capture[:config]) == stable.call(baseline[:config])
    failures << "hostname differs from baseline" unless capture[:hostname] == baseline[:hostname]
    %w[frontend_sha256 frontend_env core_exe_sha256 engine_sha256 model hardware source_revision].each do |key|
      failures << "manifest #{key} differs from baseline" unless capture[:manifest][key] == baseline[:manifest][key]
    end
    failures
  end

  def public_row(capture, role)
    capture.reject { |key, _| %i[manifest input output runs hostname].include?(key) }.merge(role: role)
  end

  def analyze(baseline_path, candidate_paths)
    baseline = load_capture(baseline_path)
    candidates = candidate_paths.map { |path| load_capture(path) }
    candidates.each do |capture|
      if baseline[:valid]
        capture[:failures] = gate_failures(capture, baseline) if capture[:valid]
      else
        capture[:failures] << "baseline is invalid"
      end
      capture[:eligible] = baseline[:valid] && capture[:valid] && capture[:failures].empty?
    end
    ranked = candidates.select { |capture| capture[:eligible] }.sort_by { |capture| [capture[:objective_ms], capture[:path]] }
    ranked.each_with_index { |capture, index| capture[:rank] = index + 1 }
    best = ranked.first
    {schema: "flash-sass-die-balance-analysis-v1",
     status: !baseline[:valid] ? "rejected" : candidate_paths.empty? ? "pending" : best ? "accepted" : "rejected",
     objective: "minimize max_run(max_die(profile header total_ms)); measured requests only",
     limits: [LIMITATION, "Model identity uses path/stat metadata, not a complete content hash."],
     baseline: public_row(baseline, "baseline"),
     candidates: candidates.map { |capture| public_row(capture, "candidate") },
     selected: best && {path: best[:path], label: best[:label], objective_ms: best[:objective_ms],
                        split_shares: best[:config]["split"], attn_map: best[:config]["attn_map"]}}
  end

  def print_report(report, io = $stdout)
    io.puts "Measured die-balance analysis: #{report[:status].upcase}"
    io.puts report[:objective], LIMITATION
    io.puts format("%-10s %-24s %5s %12s %12s %12s %s", "ROLE", "LABEL", "RUNS", "MAX TOTAL ms", "SPREAD ms", "MEAN tok/s", "RESULT")
    ([report[:baseline]] + report[:candidates]).each do |row|
      values = %i[objective_ms worst_spread_ms mean_decode_tok_s].map { |key| row[key] ? format("%.6f", row[key]) : "-" }
      result = row[:role] == "baseline" ? "REFERENCE / NOT SELECTED" : row[:eligible] ? "ELIGIBLE rank=#{row[:rank]}" : "REJECTED"
      io.puts format("%-10s %-24s %5s %12s %12s %12s %s", row[:role].upcase, row[:label], row[:repetitions] || "-", *values, result)
      row[:failures].each { |failure| io.puts "  failure: #{failure}" }
    end
    if report[:selected]
      io.puts "Best measured candidate: #{report[:selected][:path]}"
      io.puts "SPLIT_SHARES=#{report[:selected][:split_shares]}", "ATTN_MAP=#{report[:selected][:attn_map]}"
    else
      io.puts "No measured candidate selected; no acceptance claim."
    end
  end

  def cli(argv)
    options = {}
    parser = OptionParser.new do |opts|
      opts.banner = "Usage: ruby kern/die_balance.rb --baseline capture.json [--json NEW_REPORT] [candidate.json ...]"
      opts.on("--baseline PATH") { |path| options[:baseline] = path }
      opts.on("--json PATH") { |path| options[:json] = path }
      opts.on("-h", "--help") { puts opts; return 0 }
    end
    parser.parse!(argv)
    raise OptionParser::MissingArgument, "--baseline" unless options[:baseline]
    report = analyze(options[:baseline], argv)
    print_report(report)
    File.open(options[:json], "wx") { |file| file.puts JSON.pretty_generate(report) } if options[:json]
    report[:status] == "rejected" ? 1 : 0
  rescue OptionParser::ParseError, SystemCallError => error
    warn error.message
    2
  end
end
exit DieBalance.cli(ARGV) if $PROGRAM_NAME == __FILE__
