#!/usr/bin/env ruby
# Finite diagnostic. The parent runs GPU work through gpuq.
# --plan reads saved GPUQ documents/artifact identities only; no engine access.
# Defaults reproduce failed full-model B32 settings while toggling only Q3.
# STATE_Q3_OUTPUT_PREFIX=0 checks the 8192-input boundary; default is the first
# saved output divergence (38), teacher-forcing the shared reference prefix.
require_relative 'state-compare'

module StateQ3Compare
  FAILED_JOB = 'bed446e0-7011-48a0-92f9-e1089f4bafac'
  BASELINE_JOB = StateCompare::BASELINE_JOB
  SETTINGS = { 'q2_batch' => 'FLASH_Q2_BATCH', 'hcup_batch' => 'FLASH_HCUP_BATCH',
    'hcr_batch' => 'FLASH_HCR_BATCH', 'dnstep_batch' => 'FLASH_DNSTEP_BATCH',
    'actq_batch' => 'FLASH_ACTQ_BATCH' }.freeze
  CONFIG_KEYS = %w[maxt layers nblk dies split attn_map rope_owner kv_host ple_disk ple_cache ple_readers img_rows attn_q_split attn_q_mask trace_tokens].freeze

  def self.ids(record)
    record.fetch('wire').fetch('received').filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
  end

  def self.fixture(argv)
    baseline_job = ENV.fetch('BASELINE_JOB', BASELINE_JOB)
    failed_job = ENV.fetch('STATE_Q3_FAILED_JOB', FAILED_JOB)
    baseline_id, baseline = StateCompare.document(baseline_job, 'goal-prefill-0')
    failed_id, failed = StateCompare.document(failed_job, 'goal-prefill-0')
    artifact_id, artifact = StateCompare.document(failed_job, 'candidate-artifacts')
    allocation_id, allocation = StateCompare.document(failed_job, 'candidate-allocated')
    command = ->(row) { row.fetch('wire').fetch('sent').find { |line| line.start_with?('chat ') } || raise('missing saved chat command') }
    raise 'saved prompt commands differ' unless command.call(baseline) == command.call(failed)
    prompt = command.call(baseline).split.drop(2).map { |token| Integer(token) }
    raise 'expected the original 8192-token prompt' unless prompt.size == 8192
    expected, observed = ids(baseline), ids(failed)
    first = (0...[expected.size, observed.size].max).find { |i| expected[i] != observed[i] }
    raise 'saved outputs do not diverge' unless first
    suffix = Integer(ENV.fetch('STATE_Q3_OUTPUT_PREFIX', first.to_s))
    raise 'teacher-forced outputs must be within the common saved prefix' unless suffix.between?(0, first)
    consumed = prompt + expected.first(suffix)
    options = StateCompare.fields(allocation.fetch('config'))
    batch = Integer(ENV.fetch('STATE_Q3_BATCH', options.fetch('prefill_batch')))
    raise 'STATE_Q3_BATCH must be 8,16 or32' unless [8, 16, 32].include?(batch)
    settings = SETTINGS.to_h do |key, name|
      value = ENV.fetch("STATE_Q3_#{key.delete_suffix('_batch').upcase}", options.fetch(key, '0'))
      raise "invalid #{name}" unless %w[0 1].include?(value)
      [name, value]
    end
    settings['FLASH_PROFILE_PREFILL'] = '1' # proves that the Q3 batch body ran
    repeats = Integer(ENV.fetch('STATE_Q3_REPEATS', '1'))
    raise 'STATE_Q3_REPEATS must be 1..3' unless repeats.between?(1, 3)
    engine = argv.fetch(0, artifact.fetch('candidate_engine'))
    cubin = argv.fetch(1, artifact.fetch('candidate'))
    raise 'expected at most library and cubin paths' if argv.size > 2
    reference_engine = ENV.fetch('REFERENCE_ENGINE', StateCompare::REFERENCE_ENGINE)
    reference_cubin = ENV.fetch('REFERENCE_CUBIN', StateCompare::REFERENCE_CUBIN)
    paths = [engine, cubin, reference_engine, reference_cubin]
    paths.each { |path| raise 'unsafe protocol path' unless path.match?(/\A[\w.\/-]+\z/) }
    artifacts = paths.uniq.to_h { |path| [path, Digest::SHA256.file(path).hexdigest] }
    {
      'baseline_document' => baseline_id, 'failed_document' => failed_id,
      'artifact_document' => artifact_id, 'allocation_document' => allocation_id,
      'baseline_count' => expected.size, 'failed_count' => observed.size,
      'first_output_difference' => first, 'baseline_first_different_token' => expected[first],
      'failed_first_different_token' => observed[first], 'reference_output_prefix' => suffix,
      'input_ids' => consumed, 'input_count' => consumed.size,
      'input_sha256' => Digest::SHA256.hexdigest(consumed.pack('V*')),
      'stop_ids' => [expected.fetch(suffix), observed.fetch(suffix)].uniq,
      'batch' => batch, 'repeats' => repeats, 'settings' => settings,
      'engine' => engine, 'cubin' => cubin, 'reference_engine' => reference_engine,
      'reference_cubin' => reference_cubin, 'artifacts' => artifacts,
      'note' => 'Equal-input state diagnostic, not a throughput measurement. Teacher-forced reference output IDs are known inputs. FNV equality is fingerprint evidence; differences can include documented attention padding.'
    }
  end

  class Core < StateCompare::Core
    def reload_mode(engine, cubin, batch, chunk, pair, settings)
      options = { 'FLASH_CUBIN' => cubin, 'FLASH_PREFILL_BATCH' => batch,
        'FLASH_PREFILL_CHUNK' => chunk, 'FLASH_TWO_POSITION' => pair, 'FLASH_TRACE' => 0 }.merge(settings)
      options.each do |key, value|
        raise 'unsafe reload option' unless key.match?(/\AFLASH_[A-Z0-9_]+\z/) && value.to_s.match?(/\A[\w.\/-]+\z/)
      end
      reply = single('reload ' + engine + ' ' + options.map { |key, value| "#{key}=#{value}" }.join(' '))
      raise reply unless reply.start_with?('ok:')
      close
      connect
      single('config')
    end

    def fixed(inputs, request_id:, stops:, batch:)
      prepare(stops)
      lines = exchange('chat 1 ' + inputs.join(' '), timeout: 900) { |line| line.start_with?('done ') }
      timing_line = lines.find { |line| line.start_with?('timings ') } or raise 'missing timing record'
      timing = JSON.parse(timing_line.delete_prefix('timings '))
      raise 'another request interleaved' unless Integer(timing.fetch('request_id')) == request_id
      raise 'device error/incomplete counters' unless timing['counts_final'] && timing['error_code'] == 0 && !timing['aborted']
      raise 'snapshot reused' unless timing.fetch('resumed_positions') == 0
      raise 'input count changed' unless timing.fetch('prompt_tokens') == inputs.size
      tokens = lines.filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
      hashes = exchange('statehash', timeout: 240) { |line| line == 'statehash done' }
      state = StateCompare.state(hashes, 6, 48)
      valid = timing['positions_completed'] == inputs.size && timing['stop_token_observed'] &&
        tokens.size == 1 && stops.include?(tokens.first) && Integer(state['metadata'].fetch('positions')) == inputs.size
      expected = ((inputs.size - 1) / batch) * batch
      raise 'wrong count of batched known positions' if valid && Integer(state['metadata'].fetch('batched_positions')) != expected
      { 'valid' => valid, 'tokens' => tokens, 'timing' => timing, 'received' => lines,
        'state_wire' => hashes, 'state' => state, 'stop_ids' => stops,
        'consumed_ids_sha256' => Digest::SHA256.hexdigest(inputs.pack('V*')) }
    end

    def prompt_profile(request_id:, batch:)
      entries = 6.times.map do |die|
        lines = exchange("prefillprof #{die}") { |line| line == 'prefillprof done' }
        metadata = StateCompare.fields(lines.first)
        raise 'missing or stale prompt profile' unless metadata['request'] == request_id.to_s && metadata['valid'] == '1' && metadata['width'] == batch.to_s
        { 'die' => die, 'lines' => lines, 'metadata' => metadata }
      end
      count = entries.sum do |entry|
        entry['lines'].sum { |line| (match = line.match(/^\s+q3kk(?:2560|6144)8\s+(\d+)\s+phases\b/)) ? Integer(match[1]) : 0 }
      end
      { 'q3_batch_phases' => count, 'dies' => entries }
    end
  end

  def self.compare(left, right)
    return { 'comparable' => false, 'reason' => 'known input hashes differ' } unless left['consumed_ids_sha256'] == right['consumed_ids_sha256']
    result = StateCompare.compare(left, right)
    result['logits_equal'] = left.fetch('state').fetch('logits') == right.fetch('state').fetch('logits') if result['comparable']
    result
  end

  def self.main(argv)
    plan = argv.delete('--plan')
    raise 'run GPU work inside gpuq' unless plan || !ENV.fetch('GPUQ_JOB_ID', '').empty?
    data = fixture(argv)
    if plan
      puts JSON.pretty_generate(data.reject { |key, _| key == 'input_ids' })
      return
    end
    StateCompare.put('q3-state-helper', [__FILE__, File.join(__dir__, 'state-compare.rb')].map { |path| { path: path, sha256: Digest::SHA256.file(path).hexdigest, source: File.read(path) } })
    StateCompare.put('q3-state-fixture', data)
    core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
    before = core.single('config')
    original = StateCompare.fields(before)
    raise 'start with the original scalar reference selected' unless original['cubin'] == data['reference_cubin'] && original.fetch('prefill_batch', '0') == '0'
    raise 'expected six dies/full model/no GPU trace' unless original['dies'] == '6' && original['layers'] == '48/48' && original['trace_tokens'] == '0'
    records, comparisons = {}, {}
    begin
      [['off', 0], ['on', 1], ['off-return', 0]].each do |name, selection|
        options = data['settings'].merge('FLASH_Q3_BATCH' => selection)
        config = core.reload_mode(data['engine'], data['cubin'], data['batch'], 64, 1, options)
        metadata = StateCompare.fields(config)
        CONFIG_KEYS.each { |key| raise "#{name} changed #{key}" unless metadata[key] == original[key] }
        raise 'wrong Q3/batch mode' unless metadata['q3_batch'] == selection.to_s && metadata['prefill_batch'] == data['batch'].to_s
        SETTINGS.each { |key, option| raise "#{name} changed #{key}" unless metadata.fetch(key, '0') == options[option].to_s }
        rows, attempts = [], []
        records[name] = { 'config' => config, 'samples' => rows, 'attempts' => attempts }
        stops = data['stop_ids']
        (data['repeats'] + 1).times do |attempt|
          row = core.fixed(data['input_ids'], request_id: attempt + 1, stops: stops, batch: data['batch'])
          row['prefill_profile'] = core.prompt_profile(request_id: attempt + 1, batch: data['batch'])
          phases = row['prefill_profile']['q3_batch_phases']
          raise 'Q3-on did not execute a Q3 batch body' if selection == 1 && phases.zero?
          raise 'Q3-off executed a Q3 batch body' if selection == 0 && phases != 0
          attempts << row
          StateCompare.put("q3-state-#{name}-#{attempt}", { config: config, sample: row })
          if row['valid']
            rows << row
            break if rows.size == data['repeats']
          else
            raise 'fixed-input request missed the exact boundary twice' if attempt > 0 || row['tokens'].empty?
            # Only the first choice must be stopped. Retrying a third observed
            # choice avoids treating an extra consumed token as equal input.
            stops = [data['stop_ids'].first, row['tokens'].first].uniq
          end
        end
        raise 'missing valid samples' unless rows.size == data['repeats']
        comparisons["#{name}-repeat"] = compare(rows[0], rows[1]) if rows.size > 1
      end
      comparisons['off-on'] = compare(records['off']['samples'][0], records['on']['samples'][0])
      comparisons['off-return'] = compare(records['off']['samples'][0], records['off-return']['samples'][0])
      StateCompare.put('q3-state-results', comparisons)
      stable = ->(key) { value = comparisons[key]; value && value['comparable'] && value['fingerprints_equal'] }
      primary = comparisons.fetch('off-on')
      repeated = data['repeats'] > 1
      StateCompare.put('q3-state-diagnosis', {
        repeats_per_mode: data['repeats'], repetition_stability_evaluated: repeated,
        off_stable: repeated ? stable.call('off-repeat') && stable.call('off-return-repeat') && stable.call('off-return') : nil,
        on_stable: repeated ? stable.call('on-repeat') : nil,
        off_return_equal: stable.call('off-return'), first_different_layer: primary['first_different_layer'],
        repeatable_q3_difference: repeated ? stable.call('off-repeat') && stable.call('off-return-repeat') && stable.call('off-return') && stable.call('on-repeat') && primary['comparable'] && !primary['fingerprints_equal'] : nil,
        matching_logits_different_choice: comparisons.select { |_, value| value['comparable'] && value['logits_equal'] && !value['output_tokens_equal'] }.keys,
        next_choices: records.transform_values { |value| value['samples'].map { |row| row['tokens'].first } }
      })
      puts JSON.generate(comparisons.transform_values { |value| value.reject { |key, _| key == 'differences' }.merge('different_fingerprint_count' => value.fetch('differences', []).size) })
    rescue StandardError => error
      StateCompare.put('q3-state-error', { error: error.message, backtrace: error.backtrace, records: records, wire: core.history })
      raise
    ensure
      begin
        core.close
        core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
        clear = SETTINGS.values.to_h { |option| [option, 0] }.merge('FLASH_Q3_BATCH' => 0, 'FLASH_PROFILE_PREFILL' => 0)
        # This patched library clears core-owned readiness IDs at initialization.
        # No inference follows this reset before the old reference is restored.
        reset = core.reload_mode(data['engine'], data['cubin'], 0, 64, 1, clear)
        after = core.reload_mode(data['reference_engine'], data['reference_cubin'], original.fetch('prefill_batch', '0'), original.fetch('prefill_chunk'), original.fetch('two_position'), clear)
        StateCompare.put('q3-state-restored', { before: before, readiness_reset_config: reset, after: after, wire: core.history })
        raise 'original configuration not restored' unless after == before
      rescue StandardError => error
        StateCompare.put('q3-state-restore-error', { error: error.message })
        raise
      ensure
        core.close
      end
    end
  end
end

StateQ3Compare.main(ARGV.dup) if $PROGRAM_NAME == __FILE__
