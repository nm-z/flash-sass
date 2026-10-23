#!/usr/bin/env ruby
# Finite diagnostic only. The parent runs this inside gpuq; no throughput claims.
# Usage: ruby state-continuation-compare.rb WIDE_LIBENGINE WIDE_CUBIN
# STATE_CONTINUATION_BATCH=16 (default) or 32 selects the wide comparison.
require_relative 'state-compare'

module StateContinuationCompare
  STEPS = 24
  CONFIG_KEYS = %w[maxt layers nblk dies split attn_map rope_owner kv_host ple_disk ple_cache ple_readers img_rows attn_q_split attn_q_mask trace_tokens].freeze

  def self.token_difference(left, right)
    (0...[left.size, right.size].max).find { |index| left[index] != right[index] }
  end

  def self.identify(record, inputs, next_token)
    record.merge('consumed_ids' => inputs, 'consumed_ids_sha256' => Digest::SHA256.hexdigest(inputs.pack('V*')),
                 'next_token' => next_token)
  end

  def self.compare(left, right)
    index = token_difference(left.fetch('consumed_ids'), right.fetch('consumed_ids'))
    if index
      return { 'comparable' => false, 'reason' => 'consumed input token IDs differ',
               'first_different_input_index' => index,
               'left_input' => left.fetch('consumed_ids')[index], 'right_input' => right.fetch('consumed_ids')[index] }
    end
    result = StateCompare.compare(left.merge('tokens' => [left.fetch('next_token')]),
                                  right.merge('tokens' => [right.fetch('next_token')]))
    result['next_token_equal'] = result.delete('output_tokens_equal') if result.key?('output_tokens_equal')
    result['left_next_token'] = left.fetch('next_token')
    result['right_next_token'] = right.fetch('next_token')
    if result['comparable']
      result['logits_equal'] = left.fetch('state').fetch('logits') == right.fetch('state').fetch('logits')
    end
    result
  end

  def self.validate_batch(record, batch, prompt_size, label)
    executed = Integer(record.fetch('state').fetch('metadata').fetch('batched_positions', '0'))
    expected = batch.zero? ? 0 : ((prompt_size - 1) / batch) * batch
    raise "#{label} executed #{executed} wide positions, expected #{expected}" unless executed == expected
  end

  class Core < StateCompare::Core
    def continuation(ids, reference_outputs, request_id:, dies: 6, layers: 48)
      prepare
      lines = exchange("chat #{STEPS} " + ids.join(' ')) { |line| line.start_with?('done ') }
      timing_line = lines.find { |line| line.start_with?('timings ') } or raise 'missing continuation timing record'
      timing = JSON.parse(timing_line.delete_prefix('timings '))
      raise 'another request interleaved with this diagnostic' unless Integer(timing.fetch('request_id')) == request_id
      raise 'device error or incomplete counters' unless timing['counts_final'] && timing['error_code'] == 0 && !timing['aborted']
      raise 'snapshot was reused' unless timing.fetch('resumed_positions') == 0
      raise 'continuation prompt changed' unless timing.fetch('prompt_tokens') == ids.size
      tokens = lines.filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
      hashes = exchange('statehash', timeout: 240) { |line| line == 'statehash done' }
      state = StateCompare.state(hashes, dies, layers)
      boundary = ids.size + STEPS
      # chat24 consumes 24 generated IDs and returns the next choice as ID24.
      # Its final choice has not been consumed and must not enter this prefix.
      inputs = ids + tokens.first(STEPS)
      mismatch = StateContinuationCompare.token_difference(tokens.first(STEPS), reference_outputs.first(STEPS))
      valid_boundary = timing['positions_completed'] == boundary && timing['stop_token_observed'] == false &&
                       tokens.size == STEPS + 1 && Integer(state.fetch('metadata').fetch('positions')) == boundary
      record = { 'kind' => 'actual-chat24', 'tokens' => tokens, 'timing' => timing, 'received' => lines,
                 'state_wire' => hashes, 'state' => state, 'boundary_valid' => valid_boundary,
                 'generated_inputs_match_baseline' => mismatch.nil?, 'first_different_generated_input_index' => mismatch,
                 'valid' => valid_boundary && mismatch.nil? }
      StateContinuationCompare.identify(record, inputs, tokens[STEPS])
    end
  end

  def self.main
    raise 'run inside gpuq' if ENV.fetch('GPUQ_JOB_ID', '').empty?
    wide_engine, wide_cubin = ARGV.fetch(0), ARGV.fetch(1)
    reference_engine = ENV.fetch('REFERENCE_ENGINE', StateCompare::REFERENCE_ENGINE)
    reference_cubin = ENV.fetch('REFERENCE_CUBIN', StateCompare::REFERENCE_CUBIN)
    batch = Integer(ENV.fetch('STATE_CONTINUATION_BATCH', '16'))
    raise 'STATE_CONTINUATION_BATCH must be 16 or 32' unless [16, 32].include?(batch)
    fixture_id, fixture = StateCompare.document(ENV.fetch('BASELINE_JOB', StateCompare::BASELINE_JOB), 'goal-decode-0')
    prompt = fixture.fetch('wire').fetch('sent').find { |line| line.start_with?('chat ') }.split.drop(2).map { |token| Integer(token) }
    outputs = fixture.fetch('wire').fetch('received').filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
    raise 'expected the baseline 64-token prompt and 128 outputs' unless prompt.size == 64 && outputs.size == 128
    fixed_ids = prompt + outputs.first(STEPS)
    desired = outputs.fetch(STEPS)
    artifacts = [reference_engine, reference_cubin, wide_engine, wide_cubin].to_h { |path| [path, Digest::SHA256.file(path).hexdigest] }
    helper_paths = [File.expand_path(__FILE__), File.expand_path('state-compare.rb', __dir__)]
    StateCompare.put('continuation-compare-helper', helper_paths.map { |path| { path: path, sha256: Digest::SHA256.file(path).hexdigest, source: File.read(path) } })
    StateCompare.put('continuation-compare-fixture', { fixture_document: fixture_id, prompt_ids: prompt,
      known_reference_generated_inputs: outputs.first(STEPS), fixed_input_ids: fixed_ids,
      next_choice_index: STEPS, expected_next_token: desired, wide_batch: batch, artifacts: artifacts,
      note: 'chat24 returns 25 IDs while consuming 88 positions; only its first24 output IDs are inputs. Diagnostic engine requests, not chat-endpoint throughput measurements. FNV fingerprint equality is evidence, not a bytewise proof.' })
    core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
    before = core.single('config')
    original = StateCompare.fields(before)
    raise 'original engine must be selected before starting' unless original.fetch('cubin') == reference_cubin && original.fetch('prefill_batch', '0') == '0'
    raise 'expected six dies/full model/no GPU trace' unless original['dies'] == '6' && original['layers'] == '48/48' && original['trace_tokens'] == '0'
    chunk = 64
    records = {}
    comparisons = {}
    wide_name = "wide#{batch}"
    begin
      # Stage all88 known PLE rows with the patched engine's request ID1.
      # The old reference reload then consumes ID1 with a one-token breaker;
      # request ID2 must wait for newly staged rows rather than reuse ID1.
      config = core.reload(wide_engine, wide_cubin, 0, chunk, 1)
      raise 'wide control did not load' unless StateCompare.fields(config)['prefill_batch'] == '0'
      prime = core.request(fixed_ids, request_id: 1)
      StateCompare.put('continuation-compare-original-prime', { config: config, probe: prime })
      modes = [
        ['original', reference_engine, reference_cubin, 0, true],
        ['control', wide_engine, wide_cubin, 0, false],
        [wide_name, wide_engine, wide_cubin, batch, false],
        ['control-return', wide_engine, wide_cubin, 0, false]
      ]
      modes.each do |name, engine, cubin, width, old_reference|
        config = core.reload(engine, cubin, width, chunk, 1)
        metadata = StateCompare.fields(config)
        raise "wrong mode #{name}" unless metadata['cubin'] == cubin && metadata.fetch('prefill_batch', '0') == width.to_s
        CONFIG_KEYS.each { |key| raise "#{name} changed #{key}" unless metadata[key] == original[key] }
        core.breaker(prompt.first == 1 ? 2 : 1) if old_reference
        request_id = old_reference ? 2 : 1
        # A one-token probe determines the second stop ID if this mode chooses
        # a token other than the saved baseline. Stops affect only fixed88.
        probe = core.request(fixed_ids, request_id: request_id)
        request_id += 1
        stops = [desired, probe.fetch('tokens').first].uniq
        rows = { 'config' => config, 'probe' => probe, 'stop_ids' => stops, 'actual' => [], 'fixed' => [] }
        records[name] = rows
        2.times do |index|
          actual = core.continuation(prompt, outputs, request_id: request_id)
          request_id += 1
          rows['actual'] << actual
          validate_batch(actual, width, prompt.size, "#{name} actual#{index}")
          fixed = core.request(fixed_ids, request_id: request_id, stops: stops, capture: true)
          request_id += 1
          fixed = identify(fixed.merge('kind' => 'fixed88'), fixed_ids, fixed.fetch('tokens').first)
          rows['fixed'] << fixed
          validate_batch(fixed, width, fixed_ids.size, "#{name} fixed#{index}")
        end
        StateCompare.put("continuation-compare-#{name}", rows)
        %w[actual fixed].each { |kind| comparisons["#{name}-#{kind}-repeat"] = compare(rows[kind][0], rows[kind][1]) }
        2.times { |index| comparisons["#{name}-actual-fixed-#{index}"] = compare(rows['actual'][index], rows['fixed'][index]) }
      end
      %w[actual fixed].each do |kind|
        comparisons["original-control-#{kind}"] = compare(records['original'][kind][0], records['control'][kind][0])
        comparisons["control-#{wide_name}-#{kind}"] = compare(records['control'][kind][0], records[wide_name][kind][0])
        comparisons["control-return-#{kind}"] = compare(records['control'][kind][0], records['control-return'][kind][0])
      end
      StateCompare.put('continuation-compare-results', comparisons)
      stable = ->(key) { row = comparisons.fetch(key); row['comparable'] && row['fingerprints_equal'] }
      choice_anomalies = comparisons.select { |_, row| row['comparable'] && row['logits_equal'] && !row['next_token_equal'] }.keys
      StateCompare.put('continuation-compare-diagnosis', { wide_batch: batch, expected_next_token: desired,
        mode_summary: records.to_h { |name, rows| [name, {
          actual_stable: stable.call("#{name}-actual-repeat"), fixed_stable: stable.call("#{name}-fixed-repeat"),
          actual_inputs_match_baseline: rows['actual'].all? { |row| row['generated_inputs_match_baseline'] },
          actual_fixed_fingerprints_equal: 2.times.all? { |index| stable.call("#{name}-actual-fixed-#{index}") },
          actual_next_choices: rows['actual'].map { |row| row['next_token'] },
          fixed_next_choices: rows['fixed'].map { |row| row['next_token'] }
        }] }, matching_logits_different_choice: choice_anomalies })
      puts JSON.generate(comparisons.transform_values { |row| row.reject { |key, _| key == 'differences' }.merge('different_fingerprint_count' => row.fetch('differences', []).size) })
    rescue StandardError => error
      StateCompare.put('continuation-compare-error', { error: error.message, backtrace: error.backtrace, records: records, wire: core.history })
      raise
    ensure
      begin
        core.close
        core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
        after = core.reload(reference_engine, reference_cubin, original.fetch('prefill_batch', '0'), Integer(original.fetch('prefill_chunk')), Integer(original.fetch('two_position')))
        StateCompare.put('continuation-compare-restored', { before: before, after: after, wire: core.history })
        raise 'original configuration was not restored' unless after == before
      rescue StandardError => error
        StateCompare.put('continuation-compare-restore-error', { error: error.message })
        raise
      ensure
        core.close
      end
    end
  end
end

StateContinuationCompare.main if $PROGRAM_NAME == __FILE__
