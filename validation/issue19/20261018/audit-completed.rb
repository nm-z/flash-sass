#!/usr/bin/env ruby
# Offline audit of completed issue #19 evidence; performs no GPU work.
require 'json'
require 'digest'

OLD_REV = '6da421bccb68e613ecbdc8142104935deecccb29'
NEW_REV = 'd46d188a435c5d959993f16dc1f78c837cc9f70d'
EVENTS = %w[zero_groups underflow_groups subnormal_scale_groups clipped_groups would_overflow_groups].freeze
ERRORS = %w[nonfinite_input_groups nonfinite_stored_scales scale_mismatches missing_slots].freeze

def need(ok, message)
  raise message unless ok
end

def uint(value)
  need(value.to_s.match?(/\A(?:0|[1-9][0-9]*)\z/), "Invalid nonnegative integer: #{value.inspect}")
  Integer(value.to_s, 10)
end

def fields(line)
  pairs = line.split.filter_map { |part| part.include?('=') ? part.split('=', 2) : nil }
  need(pairs.map(&:first).uniq.size == pairs.size, "Duplicate output field: #{line}")
  pairs.to_h
end

def one_line(text, prefix)
  lines = text.lines(chomp: true).select { |line| line.start_with?(prefix) }
  need(lines.size == 1, "Expected one #{prefix.inspect} record; got #{lines.size}")
  lines.first
end

def file_hash(root, file, expected)
  need(file == File.basename(file), "Expected a flat evidence filename: #{file}")
  need(expected.match?(/\A[0-9a-f]{64}\z/), "Invalid SHA-256 for #{file}")
  actual = Digest::SHA256.file(File.join(root, file)).hexdigest
  need(actual == expected, "Hash mismatch: #{root}/#{file}")
end

def audit_records(root, mode, revision)
  doc = JSON.parse(File.read(File.join(root, "#{mode}.json")))
  need(doc.fetch('status') == 'PASS', "#{root}/#{mode} has not passed")
  need(doc.fetch('source_revision') == revision, "Unexpected #{mode} source")
  records = doc.fetch('results')
  need(!records.empty? && records.map { |r| r.fetch('name') }.uniq.size == records.size,
       "Empty or duplicate #{mode} records")
  hashes = 0
  records.each do |row|
    need(row.fetch('passes') == true, "Failed recorded command: #{row['name']}")
    if row.key?('command')
      need(row.fetch('exit_code') == row.fetch('expected_exit'), "Unexpected exit: #{row['name']}")
    end
    {'stdout' => 'stdout_sha256', 'stderr' => 'stderr_sha256',
     'commands_file' => 'commands_sha256'}.each do |key, hash_key|
      next unless row.key?(key)
      file_hash(root, row.fetch(key), row.fetch(hash_key))
      hashes += 1
    end
  end
  if doc.key?('build_manifest_sha256')
    file_hash(root, 'build.json', doc.fetch('build_manifest_sha256'))
    hashes += 1
  end
  [doc, hashes]
end

def audit_model(root, revision, positions)
  %w[cpu build gpu].each { |mode| audit_records(root, mode, revision) }
  doc, hashes = audit_records(root, 'model', revision)
  need(doc.fetch('original_engine_restored') == true, 'Original engine was not restored')
  contexts = doc.fetch('model_contexts')
  need(contexts.map { |c| c.fetch('positions') } == positions, 'Missing, duplicate, or reordered contexts')
  corpus = File.file?(File.join(root, 'model-corpus.txt')) ? File.join(root, 'model-corpus.txt') : doc.fetch('options').fetch('corpus')
  need(Digest::SHA256.file(corpus).hexdigest == doc.fetch('corpus_sha256'), 'Corpus hash mismatch')
  all_ids = JSON.parse(File.read(File.join(root, 'tokenize-corpus.stdout')))
  need(all_ids.all? { |x| x.is_a?(Integer) && x >= 0 && x <= 0xffffffff }, 'Invalid corpus token IDs')
  expected_pairs = (3...48).step(4).flat_map { |layer| %w[K V].map { |kind| [layer.to_s, kind] } }.sort
  summaries = contexts.map do |context|
    n = uint(context.fetch('positions'))
    name = "model-#{n}"
    file_hash(root, "#{name}.tokens.json", context.fetch('tokens_sha256'))
    ids = JSON.parse(File.read(File.join(root, "#{name}.tokens.json")))
    need(ids.size == n && ids == all_ids.slice(uint(context.fetch('corpus_token_offset')), n), 'Token slice mismatch')
    raw = File.read(File.join(root, "#{name}.stdout"))
    cfg = fields(one_line(raw, 'config '))
    required = {'rev' => revision, 'kv_host' => '1', 'kv_trace' => '1', 'maxt' => '65552',
                'layers' => '48/48', 'nblk' => '512', 'dies' => '6', 'img_rows' => '1', 'two_position' => '0'}
    required['prefill_chunk'] = '0' if revision == NEW_REV
    required.each { |key, value| need(cfg.fetch(key) == value, "Unexpected config #{key}") }
    need(!raw.match?(/^error:/), 'Error in model output')
    commands = File.readlines(File.join(root, "#{name}.commands"), chomp: true)
    need(commands == ['config', 'snap 0', 'stop', 'gen 0 ' + ids.join(' '), 'kvstats', 'statehash', 'info'],
         'Unexpected model command sequence')
    time = one_line(raw, 'time: ')
    need(time == context.fetch('time') && time.match?(/for #{n} positions run \(0 from the snapshot;/),
         'Context was not fully processed from zero')
    begin_row = fields(one_line(raw, 'kvstats begin '))
    {'rev' => revision, 'cubin_fnv' => cfg.fetch('cubin_fnv'), 'positions' => n.to_s,
     'start' => '0', 'complete' => '1', 'rounding' => 'nearest_even', 'counts' => 'groups',
     'raw' => 'pre_clamp_pre_floor'}.each { |key, value| need(begin_row.fetch(key) == value, "Bad trace header #{key}") }
    need(one_line(raw, 'kvstats done ') == "kvstats done status=PASS rows=24 positions=#{n} start=0",
         'Incomplete trace')
    need(one_line(raw, 'info ') == 'info 65552 1', 'Unexpected terminal engine info')
    rows = raw.lines.select { |line| line.start_with?('kvstats layer=') }.map { |line| fields(line) }
    need(rows.size == 24 && rows.map { |row| row.values_at('layer', 'kind') }.sort == expected_pairs,
         'Duplicate or missing layer rows')
    need(rows == context.fetch('layers'), 'Manifest rows differ from raw output')
    attn_map = cfg.fetch('attn_map').split(',').map { |x| uint(x) }
    need(attn_map.size == 12, 'Invalid attention die map')
    rows.each do |row|
      groups = uint(row.fetch('groups'))
      need(groups == n * 16 && uint(row.fetch('expected_groups')) == groups, 'Wrong group coverage')
      need(row.fetch('status') == 'PASS', 'Per-layer trace failed')
      need(uint(row.fetch('die')) == attn_map.fetch((uint(row.fetch('layer')) - 3) / 4), 'Layer on wrong die')
      counts = (EVENTS + ERRORS).to_h { |key| [key, uint(row.fetch(key))] }
      need(ERRORS.all? { |key| counts.fetch(key).zero? }, 'Invalid data in trace')
      need(EVENTS.all? { |key| counts.fetch(key) <= groups }, 'Event count exceeds coverage')
      need(%w[zero_groups underflow_groups subnormal_scale_groups clipped_groups].sum { |key| counts.fetch(key) } <= groups,
           'Disjoint event counts exceed coverage')
      need(counts.fetch('would_overflow_groups') <= counts.fetch('clipped_groups'), 'Overflow count exceeds clipping count')
      bits_text = row.fetch('max_bits')
      need(bits_text.match?(/\A[0-9a-f]{8}\z/), 'Malformed maximum bits')
      bits = Integer(bits_text, 16)
      need(bits < 0x7f800000, 'Nonfinite maximum')
      exact = [bits].pack('L<').unpack1('e')
      need([Float(row.fetch('max_abs'))].pack('e').unpack1('L<') == bits, 'Decimal maximum does not round-trip')
      need(Float(row.fetch('max_hex')) == exact, 'Hex maximum differs from bits')
      need((bits == 0) == (counts.fetch('zero_groups') == groups), 'Zero maximum/count disagreement')
      need((exact > 8_319_008) == (counts.fetch('clipped_groups') > 0), 'Clipping maximum/count disagreement')
      need(uint(row.fetch('max_position')) < n && uint(row.fetch('max_head')) < 2 && uint(row.fetch('max_group')) < 8,
           'Maximum coordinate out of range')
    end
    {positions: n, rows: rows.size, groups: rows.sum { |r| uint(r.fetch('groups')) },
     max_k: rows.select { |r| r['kind'] == 'K' }.max_by { |r| Integer(r['max_bits'], 16) }.fetch('max_abs'),
     max_v: rows.select { |r| r['kind'] == 'V' }.max_by { |r| Integer(r['max_bits'], 16) }.fetch('max_abs'),
     counts: (EVENTS + ERRORS).to_h { |key| [key, rows.sum { |r| uint(r.fetch(key)) }] },
     tokens_sha256: context.fetch('tokens_sha256'), time: time}
  end
  original = doc.fetch('original_engine').fetch('config')
  restored = fields(one_line(File.read(File.join(root, 'restored-config.stdout')), 'config '))
  original.each do |key, value|
    next if key == 'cubin' # Restored byte-identical saved copy has a different path.
    need(restored.fetch(key) == value, "Restored #{key} differs")
  end
  report = {status: 'PASS', source_revision: revision, checked_model_record_hashes: hashes,
            corpus_sha256: doc.fetch('corpus_sha256'), original_engine_restored: true, contexts: summaries}
  File.write(File.join(root, 'model-audit.json'), JSON.pretty_generate(report) + "\n")
  [doc, report]
end

old_root, new_root = ARGV
abort 'Usage: audit-completed.rb OLD_EVIDENCE_DIR MERGED_EVIDENCE_DIR' unless old_root && new_root
old_doc, old_report = audit_model(old_root, OLD_REV, [128, 32768, 65536])
new_doc, new_report = audit_model(new_root, NEW_REV, [128])
old_context = old_doc.fetch('model_contexts').first
new_context = new_doc.fetch('model_contexts').first
need(old_doc.fetch('corpus_sha256') == new_doc.fetch('corpus_sha256'), 'Merged corpus differs')
need(old_context.fetch('tokens_sha256') == new_context.fetch('tokens_sha256'), 'Merged input tokens differ')
need(old_context.fetch('layers') == new_context.fetch('layers'), 'Merged 128-position layer metrics differ')
need(File.binread(File.join(old_root, 'checkpoint-and-holder-sha256.stdout')) ==
     File.binread(File.join(new_root, 'checkpoint-and-holder-sha256.stdout')), 'Checkpoint or holder identities differ')
old_output = File.read(File.join(old_root, 'model-128.stdout'))
new_output = File.read(File.join(new_root, 'model-128.stdout'))
old_cfg = fields(one_line(old_output, 'config '))
new_cfg = fields(one_line(new_output, 'config '))
%w[kv_host kv_trace ple_disk maxt layers nblk dies attn_dies split attn_map img_rows two_position cubin_fnv].each do |key|
  need(old_cfg.fetch(key) == new_cfg.fetch(key), "Merged configuration differs: #{key}")
end
need(one_line(old_output, 'tokens:') == one_line(new_output, 'tokens:'), 'Merged output tokens differ')
old_state = File.readlines(File.join(old_root, 'model-128.stdout'), chomp: true).grep(/\Astatehash(?:\s|$)/)
new_state = File.readlines(File.join(new_root, 'model-128.stdout'), chomp: true).grep(/\Astatehash(?:\s|$)/)
need(old_state.first == 'statehash positions=128 paired_positions=0', 'Unexpected old state header')
need(new_state.first == 'statehash positions=128 paired_positions=0 prefill_chunks=0', 'Unexpected merged state header')
need(old_state.size > 2 && old_state.last == 'statehash done', 'Missing state payload')
need(old_state.drop(1) == new_state.drop(1), 'Merged state-hash payload differs')
old_build = JSON.parse(File.read(File.join(old_root, 'build.json')))
new_build = JSON.parse(File.read(File.join(new_root, 'build.json')))
cubin_checks = %w[bounded unbounded trace].to_h do |variant|
  before = old_build.fetch('artifacts').fetch(variant).fetch('die.cubin').fetch('sha256')
  after = new_build.fetch('artifacts').fetch(variant).fetch('die.cubin').fetch('sha256')
  need(before == after, "Merged #{variant} cubin differs")
  [variant, before]
end
parity = {status: 'PASS', long_context_revision: OLD_REV, integration_revision: NEW_REV,
          positions_actually_compared: 128, capacity: 65552, layers_compared: 24,
          checkpoint_and_holder_hashes_equal: true, output_tokens_equal: true,
          state_payload_lines_compared: old_state.size - 2, tokens_sha256: old_context.fetch('tokens_sha256'),
          corpus_sha256: old_doc.fetch('corpus_sha256'), cubin_sha256: cubin_checks,
          limitation: 'Long contexts were measured only at the long-context revision; this comparison measures 128 positions at the merged revision.'}
File.write(File.join(new_root, 'parity-128.json'), JSON.pretty_generate(parity) + "\n")
lines = ["# Measured checkpoint layer maxima and event counts", '',
         "All measurements in these tables were collected at \`#{OLD_REV}\`.", '',
         'Each K or V group contains 32 components; each row covers 16 groups per position. Maxima are pre-clamp and pre-floor. Counts are groups, not individual clipped components.', '']
old_doc.fetch('model_contexts').each do |context|
  n = context.fetch('positions')
  lines += ["## #{n} positions", '', context.fetch('time'), '',
            "| Layer | Die | K maximum | V maximum | Underflow K / V | Clipped K / V | Would overflow K / V |",
            '|---:|---:|---:|---:|---:|---:|---:|']
  context.fetch('layers').each_slice(2) do |k, v|
    need(k['kind'] == 'K' && v['kind'] == 'V' && k['layer'] == v['layer'], 'Unexpected report row order')
    lines << "| #{k['layer']} | #{k['die']} | #{k['max_abs']} | #{v['max_abs']} | #{k['underflow_groups']} / #{v['underflow_groups']} | #{k['clipped_groups']} / #{v['clipped_groups']} | #{k['would_overflow_groups']} / #{v['would_overflow_groups']} |"
  end
  summary = old_report.fetch(:contexts).find { |c| c[:positions] == n }
  lines += ['', "Coverage: #{summary.fetch(:groups)} groups. Other totals: " +
    %w[zero_groups subnormal_scale_groups nonfinite_input_groups nonfinite_stored_scales scale_mismatches missing_slots].map { |key| "#{key}=#{summary.fetch(:counts).fetch(key)}" }.join(', ') + '.', '']
end
File.write(File.join(old_root, 'model-layer-tables.md'), lines.join("\n") + "\n")
puts JSON.pretty_generate({status: 'PASS', long_model: old_report, merged_model: new_report, parity: parity})
