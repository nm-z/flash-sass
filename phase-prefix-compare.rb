#!/usr/bin/env ruby
# Parent-owned finite GPU diagnostic. --plan is read-only.
# Known-prefix K runs K complete two-layer positions, then scalar position K
# stops at a semantic boundary. No vocabulary head or generated IDs are used.
require_relative 'state-q3-compare'

module PhasePrefixCompare
  STAGES = %w[input read up quant conv dn norm outquant out outsend ffnread ffnup ffnquant router select gateup act down parts write].freeze
  class Core < StateQ3Compare::Core
    def probe(inputs, layer, stage)
      lines = exchange("phaseprobe #{layer} #{stage} " + inputs.join(' '), timeout: 900) { |line| line == 'phaseprobe done' }
      raise 'probe reported an invalid live range' if lines.any? { |line| line.start_with?('phaseprobe error=') }
      header = lines.find { |line| line.start_with?('phaseprobe request=') } or raise 'missing phaseprobe header'
      metadata = StateCompare.fields(header)
      raise 'wrong phaseprobe boundary' unless metadata['request'] == '1' && metadata['positions'] == inputs.size.to_s && metadata['layer'] == layer.to_s && metadata['stage'] == stage && metadata['layers'] == '2' && metadata['valid'] == '1'
      fingerprints = {}
      lines.each do |line|
        next unless (match = line.match(/\Aphaseprobe die=(\d+) name=(\w+) bytes=(\d+) fnv=([0-9a-f]{16})\z/))
        key = "#{match[1]}:#{match[2]}"
        raise 'duplicate phaseprobe fingerprint' if fingerprints.key?(key)
        fingerprints[key] = { 'bytes' => Integer(match[3]), 'fnv' => match[4] }
      end
      6.times do |die|
        %w[dn0 conv0 dn1 conv1 ple].each { |name| raise 'missing recurrent fingerprint' unless fingerprints.key?("#{die}:#{name}") }
        raise 'missing stream output' if %w[input write ple].include?(stage) && !fingerprints.key?("#{die}:S")
      end
      { 'metadata' => metadata, 'fingerprints' => fingerprints, 'wire' => lines,
        'inputs_sha256' => Digest::SHA256.hexdigest(inputs.pack('V*')) }
    end
  end

  def self.compare(left, right)
    raise 'different known inputs' unless left['inputs_sha256'] == right['inputs_sha256']
    a, b = left.fetch('fingerprints'), right.fetch('fingerprints')
    differences = (a.keys | b.keys).sort.filter_map do |key|
      { 'key' => key, 'left' => a[key], 'right' => b[key] } unless a[key] == b[key]
    end
    selected = differences.select { |row| row['key'].end_with?(':S') }
    live = differences.reject { |row| row['key'].match?(/:(?:dn0|conv0|dn1|conv1|ple)\z/) }
    { 'equal' => differences.empty?, 'stream_equal' => selected.empty?,
      'live_outputs_equal' => live.empty?, 'different_live_output_count' => live.size,
      'different_fingerprint_count' => differences.size, 'differences' => differences,
      'layer0_recurrence_equal' => differences.none? { |row| row['key'].match?(/:(?:dn0|conv0)\z/) } }
  end

  def self.main(argv)
    plan_only = argv.delete('--plan')
    raise 'run GPU work inside gpuq' unless plan_only || !ENV.fetch('GPUQ_JOB_ID', '').empty?
    data = StateQ3Compare.fixture([])
    engine = argv.fetch(0)
    cubin = argv.fetch(1, data['cubin'])
    raise 'expected diagnostic library and optional cubin' if argv.size > 2
    [engine,cubin].each { |path| raise 'unsafe artifact path' unless path.match?(/\A[\w.\/-]+\z/) }
    inputs = data.fetch('input_ids')
    cap = Integer(ENV.fetch('PHASE_PREFIX_MAX', (inputs.size - 1).to_s))
    raise 'invalid prefix cap' unless cap.between?(1, inputs.size - 1)
    budget = Integer(ENV.fetch('PHASE_PREFIX_PAIRS', '18'))
    raise 'pair budget must be 1..64' unless budget.between?(1,64)
    q3 = ENV.fetch('PHASE_PREFIX_Q3','0')
    raise 'invalid Q3 selection' unless %w[0 1].include?(q3)
    checkpoints = ENV.fetch('PHASE_PREFIX_CHECKPOINTS','32,64,128,256,512,1024,2048,4096').split(',').map { |value| Integer(value) }.select { |value| value <= cap }
    raise 'negative prefix checkpoint' if checkpoints.any?(&:negative?)
    checkpoints << cap
    checkpoints = checkpoints.uniq.sort
    settings = data.fetch('settings').merge('FLASH_Q3_BATCH' => q3, 'FLASH_ACTQ_BATCH' => '0',
      'FLASH_RMS_BATCH' => '0', 'FLASH_PROFILE_PREFILL' => '0', 'FLASH_LAYERS' => '2')
    identity = { 'engine' => engine, 'cubin' => cubin, 'engine_sha256' => Digest::SHA256.file(engine).hexdigest,
      'cubin_sha256' => Digest::SHA256.file(cubin).hexdigest, 'settings' => settings,
      'batch' => data['batch'], 'known_prefix_checkpoints' => checkpoints, 'maximum_known_prefix' => cap,
      'maximum_pairs' => budget, 'input_source' => data.reject { |key, _| key == 'input_ids' },
      'note' => 'Diagnostic only: two layers, identical known IDs, headless final semantic cut. Prefix bisection reports an observed bracket, not monotonicity or globally earliest nondeterminism. Every pair reloads the same artifacts/settings.' }
    if plan_only
      puts JSON.pretty_generate(identity)
      return
    end
    paths = [__FILE__,File.join(__dir__,'state-q3-compare.rb'),File.join(__dir__,'state-compare.rb')]
    StateCompare.put('phase-prefix-helper',paths.map { |path| { path: path, sha256: Digest::SHA256.file(path).hexdigest, source: File.read(path) } })
    StateCompare.put('phase-prefix-plan',identity.merge('fixed_input_ids' => inputs))
    core = Core.new(ENV.fetch('STATE_SOCKET','/tmp/flash.sock'))
    before = core.single('config');original = StateCompare.fields(before)
    raise 'original reference must be selected' unless original['cubin'] == data['reference_cubin'] && original.fetch('prefill_batch','0') == '0' && original['layers'] == '48/48'
    pairs = 0;records = [];observed_low = nil;observed_high = nil;phase_results = []
    capture_pair = lambda do |known,layer,stage,label|
      raise 'diagnostic pair budget exhausted' if pairs >= budget
      rows = 2.times.map do |attempt|
        config = core.reload_mode(engine,cubin,data['batch'],64,1,settings)
        metadata = StateCompare.fields(config)
        raise 'diagnostic layer selection failed' unless metadata['layers'] == '2/48'
        (StateQ3Compare::CONFIG_KEYS - ['layers']).each { |key| raise "diagnostic changed #{key}" unless metadata[key] == original[key] }
        raise 'wrong Q3 mode' unless metadata.fetch('q3_batch','0') == q3
        StateQ3Compare::SETTINGS.each { |key, option| raise "diagnostic changed #{key}" unless metadata.fetch(key,'0') == settings[option].to_s }
        row = core.probe(inputs.first(known + 1),layer,stage)
        expected = (known / data['batch']) * data['batch']
        raise 'wrong known batch count' unless Integer(row['metadata'].fetch('batched_positions')) == expected
        StateCompare.put("phase-prefix-#{pairs}-#{attempt}",{ known_prefix: known, final_position: known, layer: layer, stage: stage, label: label, config: config, capture: row })
        row
      end
      result = compare(*rows).merge('known_prefix' => known, 'layer' => layer, 'stage' => stage, 'label' => label)
      records << result;pairs += 1
      StateCompare.put("phase-prefix-comparison-#{pairs}",result)
      puts JSON.generate(result.reject { |key, _| key == 'differences' });STDOUT.flush
      result
    end
    begin
      checkpoints.each do |known|
        break if pairs >= budget
        row = capture_pair.call(known,1,'input','exponential-prefix')
        if row['stream_equal']
          observed_low = known
        else
          observed_high = known
          break
        end
      end
      # All comparisons use the same fixed prefix. Nondeterminism need not be
      # monotone; retain every result and describe this as an observed bracket.
      if observed_high && observed_low.nil? && pairs<budget
        row=capture_pair.call(0,1,'input','zero-prefix-boundary')
        row['stream_equal'] ? observed_low=0 : observed_high=0
      end
      if observed_high && observed_low
        while observed_high - observed_low > 1 && pairs < budget
          middle = (observed_high + observed_low) / 2
          row = capture_pair.call(middle,1,'input','binary-prefix')
          row['stream_equal'] ? observed_low = middle : observed_high = middle
        end
      end
      if observed_high && ENV.fetch('PHASE_PREFIX_PHASES','0') == '1'
        STAGES.each do |stage|
          break if pairs >= budget
          phase_results << capture_pair.call(observed_high,0,stage,'layer0-producer')
        end
      end
      StateCompare.put('phase-prefix-result',{ pairs: pairs, pair_budget: budget, equal_observed_prefix: observed_low,
        different_observed_prefix: observed_high, bracket_width: observed_low && observed_high ? observed_high-observed_low : nil,
        layer0_recurrence_equal_at_different_stream: records.select { |row| !row['stream_equal'] }.all? { |row| row['layer0_recurrence_equal'] },
        first_observed_different_producer: phase_results.find { |row| !row['live_outputs_equal'] }&.fetch('stage'), comparisons: records,
        note: 'Observed trial comparisons only. A passing prefix does not prove all smaller prefixes stable; independently repeat a bracket before attributing a producer.' })
    rescue StandardError => error
      StateCompare.put('phase-prefix-error',{ error: error.message, backtrace: error.backtrace, comparisons: records, wire: core.history })
      raise
    ensure
      begin
        core.close;core = Core.new(ENV.fetch('STATE_SOCKET','/tmp/flash.sock'))
        clear = StateQ3Compare::SETTINGS.values.to_h { |key| [key,0] }.merge('FLASH_Q3_BATCH' => 0,
          'FLASH_RMS_BATCH' => 0, 'FLASH_PROFILE_PREFILL' => 0, 'FLASH_LAYERS' => 48)
        reset = core.reload_mode(engine,cubin,0,64,1,clear)
        after = core.reload_mode(data['reference_engine'],data['reference_cubin'],0,original.fetch('prefill_chunk'),original.fetch('two_position'),clear)
        StateCompare.put('phase-prefix-restored',{ before: before, readiness_reset_config: reset, after: after })
        raise 'original configuration not restored' unless after == before
      rescue StandardError => error
        StateCompare.put('phase-prefix-restore-error',{ error: error.message });raise
      ensure
        core.close
      end
    end
  end
end

PhasePrefixCompare.main(ARGV.dup) if $PROGRAM_NAME == __FILE__
