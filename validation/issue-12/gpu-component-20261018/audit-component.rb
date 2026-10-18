#!/usr/bin/env ruby
# Audit the preserved issue-12 component evidence without initializing CUDA.
require 'json'
require 'csv'
require 'digest'

def check(condition, message)
  raise message unless condition
end

def median(values)
  v = values.sort
  check(!v.empty?, 'empty distribution')
  v.length.odd? ? v[v.length / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2.0
end

root = File.expand_path(ARGV.fetch(0))
out = File.expand_path(ARGV.fetch(1))
manifest = JSON.parse(File.read(File.join(root, 'manifest.json')))
check(File.read(File.join(root, 'status.txt')).strip == 'PASS', 'run did not finish PASS')
check(manifest.fetch('whole_model') == false, 'component evidence mislabeled whole-model')
check(manifest.fetch('host_lock') == '/tmp/flash-sass-engine-validation.lock', 'wrong host reservation')
check(manifest.fetch('harness_source') == '6ae0a092f2a5a0f6a472066152bcdd12daef8303', 'unexpected harness source')
counts = [1, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051, 2052, 4095, 4096, 4097, 8191, 8192, 8193]
seeds = [0, 77, 4242]
modes = %w[historical matched traced]
check(manifest.fetch('keys').split(',').map { |x| Integer(x) } == counts, 'wrong key coverage')
check(manifest.fetch('seeds').split(',').map { |x| Integer(x) } == seeds, 'wrong seed coverage')
check(manifest.fetch('modes').split(',') == modes, 'wrong mapping/trace coverage')
check(manifest.fetch('repeats') == 32, 'wrong repetition count')
expected = modes.product(counts, seeds).map { |mode, n, seed| "#{mode}-keys#{n}-seed#{seed}" }
files = Dir.glob(File.join(root, '*-keys*-seed*.json')).map { |p| File.basename(p, '.json') }
check(files.sort == expected.sort, 'missing, extra, or duplicate component cases')

def table(path)
  rows = File.readlines(path, chomp: true).map { |line| line.split("\t") }
  check(rows.all? { |r| r.length == 2 }, "invalid table #{path}")
  check(rows.map(&:first).uniq.length == rows.length, "duplicate table row #{path}")
  rows.to_h.transform_values { |x| Integer(x) }
end

exits = table(File.join(root, 'exits.tsv'))
children = table(File.join(root, 'children.tsv'))
check(exits.keys.sort == (expected + ['uninstrumented-rejection']).sort, 'incomplete exit coverage')
check(children.keys.sort == exits.keys.sort, 'incomplete child identity coverage')
check(children.values.all?(&:positive?), 'invalid child PID')
check(expected.all? { |k| exits.fetch(k) == 0 }, 'positive case failed')
check(exits.fetch('uninstrumented-rejection') == 1, 'negative case did not exit 1')
rejection = File.read(File.join(root, 'uninstrumented-rejection.stderr'))
check(rejection.match?(/^trace rejected variant=/), 'negative test did not reject a trace')
check(!rejection.match?(/undefined symbol|CUDA_ERROR|->/), 'loader or CUDA failure mistaken for negative trace test')

results = {}
raw_records = 0
work = {}
modes.product(counts, seeds).each do |mode, n, seed|
  name = "#{mode}-keys#{n}-seed#{seed}"
  v = JSON.parse(File.read(File.join(root, "#{name}.json")))
  checks = {
    'keys' => n, 'seed' => seed, 'device' => 'Tesla M60',
    'bit_exact' => true, 'boundary_canaries' => true, 'boundary_canary_words' => 64,
    'guard_checked_launches' => 148, 'partial_values' => 49_536, 'final_values' => 6144,
    'steps_per_sample' => 64, 'trace_verified' => mode == 'traced',
    'timing_trace_enabled' => mode == 'traced',
    'trace_launches' => mode == 'traced' ? 148 : 0,
    'trace_records' => mode == 'traced' ? 7168 : 0
  }
  checks.each { |key, wanted| check(v.fetch(key) == wanted, "#{name}: wrong #{key}") }
  %w[old new].product(%w[event partial_program merge]).each do |arm, phase|
    d = v.fetch(arm).fetch(phase)
    samples = d.fetch('samples_us')
    check(samples.length == 32 && samples.all? { |x| x.is_a?(Numeric) && x.finite? && x.positive? }, "#{name}: invalid #{arm}/#{phase} samples")
    check((median(samples) - d.fetch('median_us')).abs <= 0.002, "#{name}: inconsistent median")
    check((samples.sort[(0.95 * samples.length).ceil - 1] - d.fetch('p95_us')).abs <= 0.002, "#{name}: inconsistent p95")
  end
  results[name] = v
  next unless mode == 'traced'

  lines = File.readlines(File.join(root, "#{name}.stderr"), chomp: true)
  check(lines.length == 256 && lines.all? { |line| line.start_with?('TRACE ') }, "#{name}: incomplete raw trace")
  records = {}
  lines.each do |line|
    parts = line.split.drop(1).map { |part| part.split('=', 2) }
    check(parts.all? { |p| p.length == 2 && p[1].match?(/\A\d+\z/) }, "#{name}: malformed trace field")
    check(parts.map(&:first).uniq.length == parts.length, "#{name}: duplicate trace field")
    r = parts.to_h.transform_values { |x| Integer(x) }
    launch = r.fetch('launch')
    token = r.fetch('token')
    phase = r.fetch('phase')
    cta = r.fetch('cta')
    identity = [launch, token, phase, cta]
    check(!records.key?(identity), "#{name}: duplicate CTA record")
    check((1..4).cover?(launch) && (0..1).cover?(token) && (0..1).cover?(phase) && (0..15).cover?(cta), "#{name}: invalid CTA identity")
    variant = [0, 1, 1, 0][launch - 1]
    check(r.fetch('variant') == variant && r.fetch('nonce') == launch, "#{name}: stale or wrong launch identity")
    check(r.fetch('opcode') == (phase.zero? ? 67 : 22), "#{name}: wrong operation code")
    chunk_size = ((n + 8191) / 8192) * 1024
    units = phase.zero? ? [cta, cta + 16].count { |unit| (variant.zero? ? unit & 7 : unit >> 2) * chunk_size < n } : 0
    check(r.fetch('nonempty_units') == units, "#{name}: wrong physical CTA work count")
    times = %w[wait_start wait_end body_start body_end end].map { |key| r.fetch(key) }
    check(times.first.positive? && times.last > times.first && times.each_cons(2).all? { |a, b| a <= b }, "#{name}: invalid timestamp order")
    %w[exchange_wait_ns dispatch_ns body_ns grid_wait_ns].each_with_index do |key, i|
      check(r.fetch(key) == times[i + 1] - times[i], "#{name}: inconsistent timing term")
    end
    records[identity] = r
    raw_records += 1
  end
  (1..4).each do |launch|
    (0..15).each do |cta|
      previous = nil
      (0..1).each do |token|
        (0..1).each do |phase|
          r = records.fetch([launch, token, phase, cta])
          check(previous.nil? || previous <= r.fetch('wait_start'), "#{name}: backwards phase/token transition")
          previous = r.fetch('end')
        end
      end
    end
    next if launch == 1
    earlier = records.select { |id, _| id[0] == launch - 1 }.values.map { |r| r.fetch('end') }.max
    now = records.select { |id, _| id[0] == launch }.values.map { |r| r.fetch('wait_start') }.min
    check(now > earlier, "#{name}: stale timestamps across launches")
  end
  if seed == 77 && [2048, 2051].include?(n)
    work[n] = {}
    {'old' => 1, 'new' => 2}.each do |arm, launch|
      units = (0..15).map { |cta| records.fetch([launch, 0, 0, cta]).fetch('nonempty_units') }
      work[n][arm] = {nonempty_units_by_physical_cta: units, active_ctas: units.count(&:positive?), total_nonempty_units: units.sum}
    end
  end
end

# Audit exactly the monitoring contract declared by the shared runner. GPU
# utilization is sampled and may retain this validator's recent work; on the
# measured GPU the compute-app inventory, rather than a stale percentage,
# distinguishes the validator's own PID from another process.
target = manifest.fetch('gpu')
telemetry_count = 0
target_rows = []
max_other_util = 0
Dir.glob(File.join(root, 'telemetry-*.gpu.csv')).sort.each do |path|
  prefix = path.delete_suffix('.gpu.csv')
  tag = File.basename(prefix).delete_prefix('telemetry-')
  rows = CSV.read(path).map { |row| row.map { |x| x&.strip } }
  check(rows.length == 8 && rows.all? { |row| row.length == 9 }, "#{tag}: incomplete GPU telemetry")
  check(rows.map(&:first).uniq.length == 8, "#{tag}: duplicate GPU UUID")
  check(rows.count { |row| row[0] == target } == 1, "#{tag}: target missing")
  rows.each do |row|
    utilization = Integer(row[6])
    check((0..100).cover?(utilization), "#{tag}: invalid utilization")
    if row[0] != target
      max_other_util = [max_other_util, utilization].max
      check(utilization.zero?, "#{tag}: competing off-device activity")
    else
      check(utilization.zero?, 'initial target not idle') if tag == 'initial'
      target_rows << row
    end
  end
  allowed = if tag == 'initial' || tag.end_with?('-before', '-after')
              0
            elsif tag.match?(/\Arejection-\d+\z/)
              children.fetch('uninstrumented-rejection')
            else
              m = tag.match(/\A(.+-keys\d+-seed\d+)-\d+\z/)
              check(!m.nil?, "unrecognized telemetry sample #{tag}")
              children.fetch(m[1])
            end
  CSV.read("#{prefix}.apps.csv").each do |row|
    next if row.empty? || row.all?(&:nil?)
    row = row.map { |x| x&.strip }
    check(row.length == 4, "#{tag}: malformed compute-app row")
    pid = Integer(row[1])
    check(allowed.positive? && pid == allowed, "#{tag}: another target compute process") if row[0] == target
  end
  check(File.read("#{prefix}.stderr").empty?, "#{tag}: telemetry query stderr")
  telemetry_count += 1
end
check(telemetry_count.positive?, 'no telemetry')

summaries = modes.to_h do |mode|
  [mode, counts.map do |n|
    before = median(seeds.map { |seed| results.fetch("#{mode}-keys#{n}-seed#{seed}").fetch('old').fetch('event').fetch('median_us') })
    after = median(seeds.map { |seed| results.fetch("#{mode}-keys#{n}-seed#{seed}").fetch('new').fetch('event').fetch('median_us') })
    {keys: n, seeds: 3, old_partial_merge_us: before, new_partial_merge_us: after,
     delta_us: after - before, reduction_percent: 100.0 * (before - after) / before}
  end]
end
result = {
  status: 'PASS', cpu_only_audit: true, gpu_executed_by_this_audit: false,
  source: manifest, component_cases: results.length,
  retained_event_samples_checked: results.length * 2 * 3 * 32,
  raw_cta_records_checked: raw_records, negative_trace_rejection: 'PASS',
  telemetry_samples_checked: telemetry_count, maximum_other_gpu_utilization_percent: max_other_util,
  telemetry_limitation: 'One-second sampled evidence cannot exclude sub-second interference.',
  target: {uuid: target, drivers: target_rows.map { |r| r[2] }.uniq,
           temperature_c_range: target_rows.map { |r| Integer(r[3]) }.minmax,
           sm_clock_mhz_range: target_rows.map { |r| Integer(r[4]) }.minmax,
           memory_clock_mhz_range: target_rows.map { |r| Integer(r[5]) }.minmax},
  physical_cta_work: work, summaries: summaries,
  whole_model_validated: false
}
File.write(out, JSON.pretty_generate(result) + "\n")
puts "PASS: #{results.length} component cases, #{raw_records} raw CTA records, #{telemetry_count} telemetry snapshots; whole-model not validated."
summaries.each do |mode, rows|
  rows.select { |row| [2048, 2051, 8192].include?(row[:keys]) }.each do |row|
    puts format('%s keys=%d: %.3f -> %.3f us; reduction %.3f%%', mode, row[:keys], row[:old_partial_merge_us], row[:new_partial_merge_us], row[:reduction_percent])
  end
end
