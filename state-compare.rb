#!/usr/bin/env ruby
# Finite diagnostic only. Parent runs this inside gpuq; no throughput claims.
# Usage: ruby state-compare.rb WIDE_LIBENGINE WIDE_CUBIN
# STATE_OUTPUT_PREFIX=24 also teacher-forces the first 24 reference output IDs.
require 'socket'
require 'timeout'
require 'json'
require 'open3'
require 'digest'

module StateCompare
  REFERENCE_ENGINE = '/home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so'
  REFERENCE_CUBIN = '/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
  BASELINE_JOB = '63668679-0f08-49f2-b283-22969e66b4bf'

  def self.put(label, value)
    _, error, status = Open3.capture3('gpuq', 'put', label, stdin_data: JSON.generate(value))
    raise error unless status.success?
  end

  def self.document(job, source)
    raise 'invalid baseline job ID' unless job.match?(/\A[0-9a-f-]{36}\z/)
    raise 'invalid source' unless source.match?(/\A[a-z0-9-]+\z/)
    sql = "SELECT id FROM documents WHERE job_id='#{job}' AND kind='supplied' AND source='#{source}' ORDER BY id DESC LIMIT 1"
    text, error, status = Open3.capture3('gpuq', 'query', sql)
    raise error unless status.success?
    id = JSON.parse(text).fetch(0).fetch('id')
    text, error, status = Open3.capture3('gpuq', 'document', id.to_s, '--raw')
    raise error unless status.success?
    [id, JSON.parse(text)]
  end

  def self.fields(line)
    line.split.drop(1).to_h { |word| word.split('=', 2) }
  end

  def self.state(lines, dies, layers)
    header = lines.find { |line| line.start_with?('statehash positions=') } or raise 'missing state header'
    values = { 'metadata' => fields(header), 'layers' => {}, 'ple' => {}, 'logits' => {} }
    lines.each do |line|
      case line
      when /\Astatehash die=(\d+) layer=(\d+) state=([0-9a-f]{16})\z/
        key = "#{Integer($2)}:#{Integer($1)}"
        raise 'duplicate layer fingerprint' if values['layers'].key?(key)
        values['layers'][key] = $3
      when /\Astatehash die=(\d+) ple=([0-9a-f]{16}) logits=([0-9a-f]{16})\z/
        die = Integer($1).to_s
        raise 'duplicate PLE fingerprint' if values['ple'].key?(die)
        values['ple'][die] = $2
        values['logits'][die] = $3
      end
    end
    expected = []
    layers.times do |layer|
      keys = values['layers'].keys.select { |key| key.start_with?("#{layer}:") }
      required = layer % 4 == 3 ? 1 : dies
      raise "layer #{layer} has #{keys.size} fingerprints, expected #{required}" unless keys.size == required
      keys.each do |key|
        die = Integer(key.split(':').last)
        raise 'invalid layer owner' unless (0...dies).cover?(die)
        expected << key
      end
    end
    raise 'unexpected layer fingerprint' unless expected.sort == values['layers'].keys.sort
    raise 'missing PLE/logit fingerprints' unless values['ple'].keys.sort == (0...dies).map(&:to_s).sort
    values
  end

  def self.compare(left, right)
    return { 'comparable' => false, 'reason' => 'request did not stop at the exact known-input boundary' } unless left['valid'] && right['valid']
    differences = []
    %w[layers ple logits].each do |kind|
      a = left.fetch('state').fetch(kind)
      b = right.fetch('state').fetch(kind)
      raise 'different fingerprint coverage' unless a.keys.sort == b.keys.sort
      a.keys.each { |key| differences << { 'kind' => kind, 'key' => key, 'left' => a[key], 'right' => b[key] } if a[key] != b[key] }
    end
    first_layer = differences.select { |entry| entry['kind'] == 'layers' }.map { |entry| Integer(entry['key'].split(':').first) }.min
    { 'comparable' => true, 'fingerprints_equal' => differences.empty?,
      'first_different_layer' => first_layer, 'differences' => differences,
      'output_tokens_equal' => left['tokens'] == right['tokens'] }
  end

  class Core
    attr_reader :history

    def initialize(path)
      @path = path
      @history = []
      connect
    end

    def connect
      @socket = UNIXSocket.new(@path)
    end

    def close
      @socket&.close
    rescue IOError
    end

    def exchange(command, timeout: 180)
      received = []
      record = { 'command' => command, 'received' => received }
      @history << record
      Timeout.timeout(timeout) do
        @socket.puts(command)
        loop do
          line = @socket.gets or raise IOError, 'engine EOF'
          line = line.chomp
          received << line
          raise IOError, line if line.start_with?('error')
          break if yield(line)
        end
      end
      received
    end

    def single(command)
      exchange(command) { true }.first
    end

    def reload(engine, cubin, batch, chunk, pair)
      [engine, cubin].each { |path| raise 'unsafe protocol path' unless path.match?(/\A[\w.\/-]+\z/) }
      reply = single("reload #{engine} FLASH_CUBIN=#{cubin} FLASH_PREFILL_BATCH=#{batch} FLASH_TRACE=0 FLASH_PREFILL_CHUNK=#{chunk} FLASH_TWO_POSITION=#{pair}")
      raise reply unless reply.start_with?('ok:')
      close
      connect
      single('config')
    end

    def prepare(stops = [])
      single('timings 1')
      single('sample 7 0 32 1')
      single('snap 0')
      single('stop' + (stops.empty? ? '' : ' ' + stops.uniq.first(2).join(' ')))
    end

    def request(ids, request_id:, stops: [], capture: false, dies: 6, layers: 48)
      prepare(stops)
      lines = exchange('chat 1 ' + ids.join(' ')) { |line| line.start_with?('done ') }
      timing_line = lines.find { |line| line.start_with?('timings ') } or raise 'missing timing record'
      timing = JSON.parse(timing_line.delete_prefix('timings '))
      raise 'another request interleaved with this diagnostic' unless Integer(timing.fetch('request_id')) == request_id
      raise 'device error or incomplete counters' unless timing['counts_final'] && timing['error_code'] == 0 && !timing['aborted']
      raise 'snapshot was reused' unless timing.fetch('resumed_positions') == 0
      raise 'prompt changed' unless timing.fetch('prompt_tokens') == ids.size
      tokens = lines.filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
      record = { 'tokens' => tokens, 'timing' => timing, 'received' => lines }
      if capture
        hashes = exchange('statehash', timeout: 240) { |line| line == 'statehash done' }
        record['state_wire'] = hashes
        record['state'] = StateCompare.state(hashes, dies, layers)
        record['valid'] = timing['positions_completed'] == ids.size && timing['stop_token_observed'] &&
                          tokens.size == 1 && stops.include?(tokens.first) &&
                          Integer(record['state']['metadata'].fetch('positions')) == ids.size
      else
        raise 'probe failed to process all known inputs' unless timing['positions_completed'] >= ids.size && !tokens.empty?
      end
      record
    end

    def breaker(token)
      prepare
      lines = exchange("gen 0 #{token}") { |line| line.start_with?('time:') }
      timing_line = lines.find { |line| line.start_with?('timings ') } or raise 'missing breaker timings'
      timing = JSON.parse(timing_line.delete_prefix('timings '))
      raise 'unexpected request before original reference' unless Integer(timing.fetch('request_id')) == 1
      raise 'breaker did not clear the old prefix' unless timing['resumed_positions'] == 0 && timing['positions_completed'] == 1
      lines
    end
  end

  def self.main
    raise 'run inside gpuq' if ENV.fetch('GPUQ_JOB_ID', '').empty?
    wide_engine, wide_cubin = ARGV.fetch(0), ARGV.fetch(1)
    reference_engine = ENV.fetch('REFERENCE_ENGINE', REFERENCE_ENGINE)
    reference_cubin = ENV.fetch('REFERENCE_CUBIN', REFERENCE_CUBIN)
    suffix = Integer(ENV.fetch('STATE_OUTPUT_PREFIX', '0'))
    raise 'STATE_OUTPUT_PREFIX must be a multiple of four in 0..124' unless suffix.between?(0, 124) && suffix % 4 == 0
    fixture_id, fixture = document(ENV.fetch('BASELINE_JOB', BASELINE_JOB), 'goal-decode-0')
    prompt = fixture.fetch('wire').fetch('sent').find { |line| line.start_with?('chat ') }.split.drop(2).map { |token| Integer(token) }
    outputs = fixture.fetch('wire').fetch('received').filter_map { |line| Integer(line.split[1]) if line.start_with?('tok ') }
    raise 'expected the baseline 64-token prompt and 128 outputs' unless prompt.size == 64 && outputs.size == 128
    ids = prompt + outputs.first(suffix)
    desired = outputs.fetch(suffix)
    artifacts = [reference_engine, reference_cubin, wide_engine, wide_cubin].to_h { |path| [path, Digest::SHA256.file(path).hexdigest] }
    put('state-compare-helper', { path: File.expand_path(__FILE__), sha256: Digest::SHA256.file(__FILE__).hexdigest, source: File.read(__FILE__) })
    put('state-compare-fixture', { fixture_document: fixture_id, prompt_ids: ids, known_reference_outputs: suffix,
      expected_next_token: desired, prompt_sha256: Digest::SHA256.hexdigest(ids.pack('V*')), artifacts: artifacts,
      note: 'Diagnostic engine requests, not chat-endpoint throughput measurements. Equality is FNV fingerprint evidence; mismatches prove byte differences.' })
    core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
    before = core.single('config')
    original = fields(before)
    raise 'original engine must be selected before starting' unless original.fetch('cubin') == reference_cubin && original.fetch('prefill_batch', '0') == '0'
    raise 'expected six dies/full model/no GPU trace' unless original['dies'] == '6' && original['layers'] == '48/48' && original['trace_tokens'] == '0'
    chunk = 64
    records = {}
    comparisons = {}
    begin
      # A patched fresh B0 engine stages every known row with request ID 1.
      # Keep this connection until original reload: no client can overwrite
      # those readiness flags between the prime and reload. The old original
      # lacks generation reset, so its one-token breaker consumes ID 1; its
      # subsequent full probe uses ID 2 and must wait on every staged row.
      config = core.reload(wide_engine, wide_cubin, 0, chunk, 1)
      raise 'wide control did not load' unless fields(config)['prefill_batch'] == '0'
      prime = core.request(ids, request_id: 1)
      put('state-compare-original-prime', { config: config, probe: prime })
      modes = [
        ['original', reference_engine, reference_cubin, 0, true],
        ['control', wide_engine, wide_cubin, 0, false],
        ['wide16', wide_engine, wide_cubin, 16, false],
        ['control-return', wide_engine, wide_cubin, 0, false]
      ]
      modes.each do |name, engine, cubin, batch, old_reference|
        config = core.reload(engine, cubin, batch, chunk, 1)
        metadata = fields(config)
        raise "wrong mode #{name}" unless metadata['cubin'] == cubin && metadata.fetch('prefill_batch', '0') == batch.to_s
        %w[maxt layers nblk dies split attn_map rope_owner kv_host ple_disk ple_cache ple_readers img_rows attn_q_split attn_q_mask trace_tokens].each do |key|
          raise "#{name} changed #{key}" unless metadata[key] == original[key]
        end
        core.breaker(ids.first == 1 ? 2 : 1) if old_reference
        first_id = old_reference ? 2 : 1
        probe = core.request(ids, request_id: first_id)
        stops = [desired, probe.fetch('tokens').first].uniq
        rows = 2.times.map do |i|
          row = core.request(ids, request_id: first_id + i + 1, stops: stops, capture: true)
          executed = Integer(row.fetch('state').fetch('metadata').fetch('batched_positions', '0'))
          expected = batch == 16 ? ((ids.size - 1) / 16) * 16 : 0
          raise "#{name} executed #{executed} B16 positions, expected #{expected}" unless executed == expected
          row
        end
        records[name] = rows
        put("state-compare-#{name}", { config: config, probe: probe, stop_ids: stops, samples: rows })
        comparisons["#{name}-repeat"] = compare(rows[0], rows[1])
      end
      comparisons['original-control'] = compare(records['original'][0], records['control'][0])
      comparisons['control-wide16'] = compare(records['control'][0], records['wide16'][0])
      comparisons['control-return'] = compare(records['control'][0], records['control-return'][0])
      put('state-compare-results', comparisons)
      stable = ->(key) { value = comparisons.fetch(key); value['comparable'] && value['fingerprints_equal'] }
      primary = comparisons.fetch('control-wide16')
      put('state-compare-diagnosis', { original_stable: stable.call('original-repeat'),
        control_stable: stable.call('control-repeat') && stable.call('control-return-repeat') && stable.call('control-return'),
        wide_stable: stable.call('wide16-repeat'), first_different_layer: primary['first_different_layer'],
        repeatable_wide_difference: stable.call('control-repeat') && stable.call('control-return-repeat') &&
          stable.call('control-return') && stable.call('wide16-repeat') && primary['comparable'] && !primary['fingerprints_equal'] })
      puts JSON.generate(comparisons.transform_values { |value| value.reject { |key, _| key == 'differences' }.merge('different_fingerprint_count' => value.fetch('differences', []).size) })
    rescue StandardError => error
      put('state-compare-error', { error: error.message, backtrace: error.backtrace, records: records, wire: core.history })
      raise
    ensure
      begin
        core.close
        core = Core.new(ENV.fetch('STATE_SOCKET', '/tmp/flash.sock'))
        after = core.reload(reference_engine, reference_cubin, original.fetch('prefill_batch', '0'), Integer(original.fetch('prefill_chunk')), Integer(original.fetch('two_position')))
        put('state-compare-restored', { before: before, after: after, wire: core.history })
        raise 'original configuration was not restored' unless after == before
      rescue StandardError => error
        put('state-compare-restore-error', { error: error.message })
        raise
      ensure
        core.close
      end
    end
  end
end

StateCompare.main if $PROGRAM_NAME == __FILE__
