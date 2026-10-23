#!/usr/bin/env ruby
# Summarize captured evidence only; this script never connects to a GPU or core.
require 'csv'
require 'json'
require 'zlib'

raw,output = ARGV
abort 'usage: summarize_evidence.rb RAW_DIRECTORY OUTPUT_DIRECTORY' unless output
def read_json(path)
  JSON.parse(path.end_with?('.gz') ? Zlib::GzipReader.open(path,&:read) : File.read(path))
end
def save(path,data)
  File.write(path,JSON.pretty_generate(data)+"\n")
end
def need(ok,message)
  raise message unless ok
end
matrices = {}
sample_rows = []
[["cta1","overhead-cta1",4],["cta16","overhead-cta16-two",2]].each do |label,dir,count|
  root = File.join(raw,dir)
  samples = File.readlines(File.join(root,'samples.jsonl')).map { |line| JSON.parse(line) }
  measured = samples.select { |row| row['kind'] == 'measured' }
  need(samples.length == 36 && measured.length == 12, "#{label}: wrong sample count")
  need(samples.map { |row| row['tokens'] }.uniq.length == 1, "#{label}: token mismatch")
  need(samples.all? { |row| row['tokens'].length == count }, "#{label}: wrong output count")
  %w[ordinary traced].each do |mode|
    need(measured.count { |row| row['mode'] == mode } == 6, "#{label}: wrong mode count")
  end
  clocks = Hash.new { |h,k| h[k] = {sm: [],memory: [],temperature: []} }
  accepted = 0; waits = 0
  Dir.glob(File.join(root,'slot-*','*-*','sample.json')).sort.each do |file|
    dir = File.dirname(file)
    checks = Dir.glob(File.join(dir,'before-chat-*.json')).sort
    need(!checks.empty?, 'missing request preflight')
    waits += checks.length - 1; accepted += 1
    before = read_json(checks.last)
    before.fetch('utilization').fetch('stdout').lines.each do |line|
      need(Float(line.split(',').last.strip).zero?, 'accepted busy GPU preflight')
    end
    need(before.fetch('ps').fetch('success'), 'missing successful ps check')
    [before,read_json(File.join(dir,'after-chat.json'))].each do |capture|
      capture.fetch('gpu').fetch('stdout').lines.each do |line|
        fields = CSV.parse_line(line).map(&:strip)
        die = Integer(fields[0]); next unless die < 6
        clocks[die][:sm] << Integer(fields[8])
        clocks[die][:memory] << Integer(fields[9])
        clocks[die][:temperature] << Integer(fields[5])
      end
    end
  end
  need(accepted == 36, 'request preflight count mismatch')
  matrices[label] = {output_tokens: count,prompt_ids: [10,11],warmups: 24,
    measured_per_mode: 6,abba_blocks: 3,tokens: samples.first.fetch('tokens'),
    statistics: read_json(File.join(root,'summary.json')),
    accepted_idle_preflights: accepted,busy_wait_observations: waits,
    gpu_observations: clocks.transform_values { |v| {
      sm_clock_mhz: v[:sm].uniq.sort,memory_clock_mhz: v[:memory].uniq.sort,
      temperature_c_range: v[:temperature].minmax} },
    metadata: read_json(File.join(root,'metadata.json')),
    original_configuration_restored: File.read(File.join(root,'restore.stdout')).include?('PASS original resident configuration restored')}
  need(matrices[label][:original_configuration_restored], 'missing restoration evidence')
  measured.each do |row|
    sample_rows << [label,row['block'],row['position'],row['mode'],row['wall_ns'],
      row['first_token_ns'],row['steady_decode_ns_per_token'],row['run_ms_rounded']]
  end
end
CSV.open(File.join(output,'overhead-samples.csv'),'w') do |csv|
  csv << %w[layout block abba_position mode client_wall_ns client_first_token_ns client_steady_ns_per_token engine_run_ms_rounded]
  sample_rows.each { |row| csv << row }
end

windows = []; per_die = []; representatives = {}
%w[cta1 cta16].each do |layout|
  paths = Dir.glob(File.join(raw,'reconciliation',layout+'-slot-*-window.json.gz')).sort
  need(paths.length == 6, "#{layout}: require every measured traced window")
  paths.each do |path|
    report = read_json(path); causal = report.fetch('causal_analysis')
    local = report.fetch('local_cta_reconciliation'); envelopes = report.fetch('same_die_envelopes')
    need(local.all? { |r| r['residual_ns'].zero? && (r['period_residual_ns'].nil? || r['period_residual_ns'].zero?) }, 'nonzero local residual')
    need(envelopes.all? { |r| r['residual_ns'].zero? }, 'nonzero per-die residual')
    need(causal['enclosing_span_upper_bound_ns'], 'unbounded cross-die span')
    original = report.fetch('analysis_scope').fetch('original_input_metadata')
    windows << {file: File.basename(path),layout: layout,positions: report['positions'],
      original_gpu_phase_records: original.select { |m| m.key?('die') }.sum { |m| m['rows'] },
      local_partitions: local.length,die_envelopes: envelopes.length,local_residual_ns: 0,
      die_residual_ns: 0,span_lower_ns: causal['makespan_lower_bound_ns'],
      span_upper_ns: causal['enclosing_span_upper_bound_ns'],
      span_interval_width_ns: causal['unresolved_span_gap_upper_bound_ns'],
      matched_generations: causal['causal_generations'],
      grid_barrier_joins: causal['grid_barrier_joins'],
      sampled_chain_ns: causal['chain_sampled_ns_after_crossing_allowance']}
    envelopes.each do |row|
      per_die << [File.basename(path),layout,row['die'],row['position'],row['ctas'],
        row['duration_ns'],row['phase_union_ns'],row['phase_overlap_ns'],
        row['phase_gap_ns'],row['publication_tail_ns'],row['residual_ns']]
    end
    if File.basename(path).include?('slot-02-')
      representatives[layout] = {
        lifecycle: report['lifecycle'],causal: causal.reject { |k,_| k == 'witness' },
        per_die: envelopes,host: report['host'],
        phase_counts: original.select { |m| m.key?('die') }.map { |m| [m['die'],m['metadata']['nph']] }.to_h,
        opcodes: report['opcode_sampled_totals'].map { |m| m['opcode'] }.uniq.sort}
    end
  end
end
CSV.open(File.join(output,'per-die-reconciliation.csv'),'w') do |csv|
  csv << %w[report layout die position sampled_ctas duration_ns phase_union_ns overlap_not_added_ns gap_ns publication_tail_ns residual_ns]
  per_die.each { |row| csv << row }
end
life = read_json(File.join(raw,'lifecycle','results.json'))
need(life['status'] == 'PASS' && life['cases'].length == 10,'lifecycle did not pass')
cancel = life['cases'].find { |row| row['case'] == 'disconnect' }
need(cancel['validation'].any? { |v| v['cancellation_callback_checked'] },'callback boundary was not observed')
text = Zlib::GzipReader.open(File.join(raw,'lifecycle','disconnect-hosttrace.txt.gz'),&:read)
lines = text.lines
meta = lines.shift.split.drop(1).map { |v| v.split('=',2) }.to_h
lines.pop
table = CSV.parse(lines.join,headers: true)
row = table.find { |r| r['position'] == meta.fetch('abort_position') }
callback = {position: Integer(row['position']),callback_start_ns: Integer(row['callback_start_ns']),
  callback_end_ns: Integer(row['callback_end_ns']),abort_start_ns: Integer(meta['abort_start_ns']),
  abort_end_ns: Integer(meta['abort_end_ns']),reason: meta['abort_reason']}
callback[:callback_duration_ns] = callback[:callback_end_ns]-callback[:callback_start_ns]
callback[:return_to_abort_start_ns] = callback[:abort_start_ns]-callback[:callback_end_ns]
callback[:abort_duration_ns] = callback[:abort_end_ns]-callback[:abort_start_ns]
need(callback[:return_to_abort_start_ns] >= 0 && callback[:reason] == 'callback','bad callback/abort order')
summary = {
  engine_source: 'c7d0a6e14e99edd9e91ea16c010ae7938ac647e0',
  memory_guard_source: '99c7134a0ef68ae9a37a827de05e8ab2fd8df17b',
  schema: 2,quantum_ns: 1024,
  clock_assumption: 'Unknown constant relative GPU offsets during each capture; no independently measured drift bound.',
  metric_scope: 'Short-generation client-observed delivery; complete request differs from the bounded GPU output window.',
  matrices: matrices,windows: windows,representative_windows: representatives,
  measured_local_partitions: windows.sum { |w| w[:local_partitions] },
  measured_die_envelopes: windows.sum { |w| w[:die_envelopes] },
  measured_gpu_phase_records: windows.sum { |w| w[:original_gpu_phase_records] },
  lifecycle_cases: life['cases'].map { |r| {name: r['case'],lifecycle: r['lifecycle']} },
  callback_cancellation: callback,
  operational_limit: 'The first four-output CTA16 reload exceeded free VRAM and its fallback exited. The original service was restarted with unchanged baseline artifacts and configuration. The new guard rejected that request before reload. Successful CTA16 evidence uses two outputs, a 16 MiB buffer per die, and an 8 MiB headroom reserve.'
}
save(File.join(output,'summary.json'),summary)
puts JSON.pretty_generate(summary.reject { |k,_| %i[matrices representative_windows].include?(k) })
