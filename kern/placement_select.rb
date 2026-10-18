#!/usr/bin/env ruby
# Offline comparison of recorded placements. Never invokes an engine or GPU.
require 'json'
require 'zlib'

module PlacementSelect
  class EvidenceError < StandardError; end
  PROTOCOL = {
    'requests_per_round' => 5, 'rounds' => 2, 'warmups_per_round' => 1,
    'request' => {
      'messages' => [{'role' => 'user', 'content' => 'What is the capital of France? Answer in one word.'}],
      'max_tokens' => 40, 'temperature' => 0
    }
  }.freeze
  PLACEMENT = %w[split attn_map attn_dies].freeze
  CONFIG_REQUIRED = %w[
    rev kv_host ple_disk maxt layers nblk dies attn_dies split attn_map
    img_rows two_position prefill_chunk cubin cubin_fnv
  ].freeze
  IDENTITY_REQUIRED = %w[
    hostname boot_id engine_config frontend_sha256 model device_uuids device_indices
    resident_processes frontend_props frontend_environment collector_sha256
  ].freeze
  module_function

  def check(condition, message)
    raise EvidenceError, message unless condition
  end

  def object(value, name)
    check(value.is_a?(Hash), "#{name}: expected object")
    value
  end

  def array(value, name)
    check(value.is_a?(Array), "#{name}: expected array")
    value
  end

  def string(value, name)
    check(value.is_a?(String) && !value.strip.empty?, "#{name}: empty/non-string")
    value
  end

  def integer(value, name, minimum = 0, maximum = nil)
    check(value.is_a?(Integer) && value >= minimum && (!maximum || value <= maximum),
          "#{name}: invalid integer")
    value
  end

  def numeric(value, name, minimum = 0, positive = false)
    check(value.is_a?(Numeric) && value.to_f.finite? &&
          (positive ? value > minimum : value >= minimum), "#{name}: invalid number")
    value.to_f
  end

  def sensor(value, name)
    result = Float(string(value, name))
    check(result.finite?, "#{name}: non-finite value")
    result
  rescue ArgumentError
    raise EvidenceError, "#{name}: invalid sensor value"
  end

  def integers(text, name, size, minimum, maximum = nil)
    values = string(text, name).split(',', -1)
    check(values.size == size && values.all? { |v| /\A\d+\z/.match?(v) }, "#{name}: invalid list")
    values.map { |v| integer(v.to_i, name, minimum, maximum) }
  end

  def config(text)
    words = string(text, 'engine_config').strip.split
    check(words.shift == 'config', 'engine_config: missing config prefix')
    fields = {}
    words.each do |word|
      match = /\A([a-z_]+)=([^\s=]+)\z/.match(word)
      check(match && !fields.key?(match[1]), 'engine_config: malformed/duplicate field')
      fields[match[1]] = match[2]
    end
    CONFIG_REQUIRED.each { |key| check(fields.key?(key), "engine_config: missing #{key}") }
    check(/\A[0-9a-f]{7,40}\z/.match?(fields['rev']), 'engine_config: invalid revision')
    check(/\A[0-9a-f]{16}\z/.match?(fields['cubin_fnv']), 'engine_config: invalid cubin hash')
    check(fields['dies'] == '6' && fields['layers'] == '48/48' && fields['nblk'] == '512',
          'engine_config: require six dies, full model and sparse budget 512')
    %w[kv_host ple_disk two_position].each do |key|
      check(%w[0 1].include?(fields[key]), "engine_config: invalid #{key}")
    end
    %w[maxt img_rows prefill_chunk].each do |key|
      check(/\A\d+\z/.match?(fields[key]), "engine_config: invalid #{key}")
    end
    check(fields['maxt'].to_i > 0, 'engine_config: zero capacity')
    integers(fields['split'], 'split', 6, 1)
    owners = integers(fields['attn_map'], 'attn_map', 12, 0, 5)
    check(fields['attn_dies'] == owners.uniq.sort.join(','), 'engine_config: attention owners disagree')
    fields
  end

  def profile(entry)
    entry = object(entry, 'profile')
    die = integer(entry.fetch('die'), 'profile die', 0, 5)
    lines = string(entry.fetch('text'), "die #{die} profile").lines.map(&:strip).reject(&:empty?)
    header = /\Aprof: (\d+) phases, (\d+\.\d{2}) ms between the first and last timestamp\z/.match(lines.shift || '')
    check(header, "die #{die}: missing/malformed profile header")
    phases = integer(header[1].to_i, 'profile phases', 2)
    span = numeric(header[2].to_f, 'profile span', 0, true) * 1000
    check(lines.last&.start_with?('longest:'), "die #{die}: missing profile terminator")
    lines.pop
    rows = {}
    lines.each do |line|
      match = /\A([a-z][a-z0-9_]*)\s+(\d+) phases\s+(\d+\.\d+) us \(\s*(\d+\.\d+) us each\)\z/.match(line)
      check(match, "die #{die}: malformed profile row")
      name, count, total, each = match[1], match[2].to_i, match[3].to_f, match[4].to_f
      check(!rows.key?(name), "die #{die}: duplicate #{name}")
      integer(count, 'phase count', 1)
      numeric(total, 'phase total', 0, true)
      numeric(each, 'phase average')
      check((each - total / count).abs <= 0.051 + 0.051 / count,
            "die #{die}: phase average disagrees")
      rows[name] = [count, total]
    end
    check(rows.fetch('actq256', [0])[0] == 146, "die #{die}: require all 146 actq256 phases")
    check(rows.values.sum(&:first) == phases - 1, "die #{die}: incomplete phase coverage")
    # The header rounds to 0.01 ms; each row rounds to 0.1 us.
    check((rows.values.sum(&:last) - span).abs <= 5.001 + 0.051 * rows.size,
          "die #{die}: totals disagree with rounded span")
    {'die' => die, 'span_us' => span, 'actq256_us' => rows['actq256'][1]}
  end

  def snapshot(raw, uuids, indices)
    raw = object(raw, 'host snapshot')
    %w[foreign_gpu_processes resident_identity_errors].each do |key|
      check(array(raw.fetch(key), key).empty?, "host snapshot: #{key} contaminated capture")
    end
    gpus = array(raw.fetch('gpus'), 'GPU telemetry')
    ids = gpus.map { |g| string(object(g, 'GPU').fetch('uuid'), 'GPU UUID') }
    check(ids == uuids, 'GPU telemetry order differs from engine-die order')
    selected = gpus.map.with_index do |gpu, die|
      %w[index clocks.sm pcie.link.gen.current pcie.link.width.current memory.total].each do |key|
        check(sensor(gpu.fetch(key), key) >= 0, "negative #{key}")
      end
      check(gpu['index'] == indices[die].to_s, 'NVML index/UUID mapping changed')
      check(sensor(gpu.fetch('utilization.gpu'), 'GPU utilization').between?(0, 100),
            'invalid GPU utilization')
      [gpu['uuid'], gpu['index'], gpu['clocks.sm'], gpu['pcie.link.gen.current'],
       gpu['pcie.link.width.current'], gpu['memory.total']]
    end
    temperatures = gpus.map { |g| sensor(g.fetch('temperature.gpu'), 'temperature') }
    links = array(raw.fetch('root_links'), 'root links').map do |link|
      link = object(link, 'root link')
      %w[pci_address speed width].map { |key| string(link.fetch(key), "root link #{key}") }
    end.sort
    check(links.size == 3 && links.map(&:first).uniq.size == 3, 'require three unique root links')
    [{'gpus' => selected, 'root_links' => links}, temperatures]
  end

  def native_commands(value)
    commands = array(value, 'native_commands').map { |line| string(line, 'native command').strip }
    verbs = commands.map { |line| line.split.first }
    check([%w[stop sample chat], %w[stop sample snap chat]].include?(verbs),
          'unexpected native command sequence')
    check(/\Astop \d+ \d+\z/.match?(commands[0]), 'invalid native stop command')
    sample = /\Asample (\d+) (\S+) (\d+) (\S+)\z/.match(commands[1])
    check(sample && sensor(sample[2], 'native temperature').zero?, 'native sampling is not greedy')
    check(sample[3].to_i > 0 && sensor(sample[4], 'native top_p').between?(0, 1),
          'invalid native sampler')
    # flashd creates a fresh positional seed for every request. Temperature zero
    # selects argmax, so ignore only this field, retaining the full prompt and controls.
    commands[1] = "sample <greedy> #{sample[2]} #{sample[3]} #{sample[4]}"
    check(verbs.size == 3 || /\Asnap \d+\z/.match?(commands[2]), 'invalid native snapshot')
    chat = commands.last.split
    check(chat.size > 2 && chat[1] == '39' && chat.drop(2).all? { |v| /\A\d+\z/.match?(v) },
          'native chat must request 39 additional tokens and contain prompt IDs')
    [commands, chat.size - 2]
  end

  def response(sample)
    ids = array(sample.fetch('token_ids'), 'token_ids')
    integer(ids.size, 'generated token count', 2, 40)
    ids.each { |id| integer(id, 'token ID', 0, 0xffffffff) }
    body = object(sample.fetch('response'), 'HTTP response')
    check(!body.key?('error'), 'HTTP response contains error')
    choices = array(body.fetch('choices'), 'choices')
    check(choices.size == 1, 'require exactly one choice')
    choice = object(choices.first, 'choice')
    message = object(choice.fetch('message'), 'message')
    check(message.fetch('role') == 'assistant', 'unexpected message role')
    %w[content reasoning_content].each do |key|
      check(message[key].nil? || message[key].is_a?(String), "invalid #{key}")
    end
    finish = string(choice.fetch('finish_reason'), 'finish reason')
    timings = object(body.fetch('timings'), 'HTTP timings')
    generated = integer(timings.fetch('predicted_n'), 'predicted_n', 2, 40)
    check(generated == ids.size, 'HTTP count differs from native token IDs')
    ms = numeric(timings.fetch('predicted_ms'), 'predicted_ms', 0, true)
    rate = numeric(timings.fetch('predicted_per_second'), 'predicted_per_second', 0, true)
    check((rate - (generated - 1) * 1000.0 / ms).abs <= 0.011,
          'HTTP timing violates G-1 decode convention')
    check(timings.fetch('decode_intervals') == generated - 1 &&
          (numeric(timings.fetch('decode_ms'), 'decode_ms') - ms).abs <= 0.051,
          'HTTP decode aliases disagree')
    usage = object(body.fetch('usage'), 'HTTP usage')
    prompt = integer(usage.fetch('prompt_tokens'), 'prompt tokens', 1)
    check(usage.fetch('completion_tokens') == generated && usage.fetch('total_tokens') == prompt + generated,
          'HTTP usage counts disagree')
    check(integer(timings.fetch('prompt_n'), 'prompt_n', 1) == prompt, 'HTTP prompt counts disagree')
    first = numeric(timings.fetch('prompt_ms'), 'prompt_ms', 0, true)
    cached = integer(timings.fetch('cache_n'), 'cache_n', 0, prompt - 1)
    done = string(sample.fetch('engine_done'), 'engine_done').strip
    match = /\Adone (\d+) ([01]) (\d+) (\d+\.\d+) (\d+\.\d+) (\d+) (\d+)\z/.match(done)
    check(match, 'invalid native done record')
    check(match[1].to_i == generated && match[3].to_i == prompt && match[6].to_i == cached,
          'native done counts disagree with HTTP')
    check(match[7].to_i < prompt, 'native snapshot exceeds prompt')
    check((match[5].to_f - first).abs <= 0.051 &&
          (match[4].to_f - match[5].to_f - ms).abs <= 0.11,
          'native done timing disagrees with HTTP')
    check(finish == (match[2] == '1' ? 'stop' : 'length'), 'native/HTTP finish reasons disagree')
    check(match[2] == '1' || generated == 40, 'incomplete token budget without stop')
    commands, native_prompt = native_commands(sample.fetch('native_commands'))
    check(native_prompt == prompt, 'native prompt length differs from HTTP')
    stops = commands.first.split.drop(1).map(&:to_i).reject { |id| id == 0xffffffff }
    check(match[2] != '1' || stops.include?(ids.last), 'native stop lacks a final stop token')
    raw = array(sample.fetch('native_responses'), 'native_responses').map { |l| string(l, 'native reply').strip }
    check(raw.none? { |l| l.start_with?('error') }, 'native transcript contains an error')
    check(raw.grep(/\Adone /) == [done], 'native transcript done disagrees')
    tokens = raw.grep(/\Atok /).map do |line|
      check(/\Atok \d+\z/.match?(line), 'invalid native token reply')
      line.split[1].to_i
    end
    check(tokens == ids, 'native transcript token IDs disagree')
    check(raw.grep(/\Aconfig /) == [sample.fetch('config')], 'native transcript config disagrees')
    profile_lines = sample.fetch('profiles').flat_map { |p| p.fetch('text').lines.map(&:strip) }
    check(raw.drop(raw.index(sample.fetch('config')) + 1) == profile_lines,
          'native transcript profiles disagree')
    {
      'tokens' => ids, 'output' => [message['content'], message['reasoning_content'], finish],
      'native_commands' => commands, 'cache_n' => cached, 'rate' => rate
    }
  end

  def record(raw)
    raw = object(raw, 'capture')
    check(raw.fetch('schema') == 'flash-sass-placement-v1', 'unknown capture schema')
    check(raw.fetch('status') == 'complete' && array(raw.fetch('failures'), 'failures').empty?,
          'incomplete/failed capture')
    check(raw.fetch('protocol') == PROTOCOL, 'capture does not use the required 5x2 protocol')
    label = string(raw.fetch('label'), 'label')
    identity = object(raw.fetch('identity'), 'identity')
    IDENTITY_REQUIRED.each { |key| check(identity.key?(key), "identity: missing #{key}") }
    %w[hostname boot_id].each { |key| string(identity[key], key) }
    %w[frontend_sha256 collector_sha256].each do |key|
      check(identity[key].is_a?(String) && /\A[0-9a-f]{64}\z/.match?(identity[key]),
            "identity: invalid #{key}")
    end
    object(identity['frontend_props'], 'frontend_props')
    object(identity['frontend_environment'], 'frontend_environment')
    model = object(identity['model'], 'model identity')
    string(model.fetch('path'), 'model path')
    integer(model.fetch('size'), 'model size', 1)
    string(model.fetch('mtime'), 'model mtime')
    residents = array(identity['resident_processes'], 'resident_processes')
    check(!residents.empty?, 'missing resident identities')
    residents.each do |r|
      r = object(r, 'resident process')
      integer(r.fetch('pid'), 'resident PID', 1)
      check(/\A\d+\z/.match?(string(r.fetch('starttime_ticks'), 'resident start time')),
            'invalid resident start time')
      string(r.fetch('comm'), 'resident command')
    end
    check(residents.map { |r| r['pid'] }.uniq.size == residents.size, 'duplicate resident PID')
    uuids = array(identity['device_uuids'], 'device UUIDs')
    uuids.each { |uuid| string(uuid, 'device UUID') }
    check(uuids.size == 6 && uuids.uniq.size == 6, 'require six unique device UUIDs')
    indices = array(identity['device_indices'], 'device indices')
    indices.each { |i| integer(i, 'NVML index') }
    check(indices.size == 6 && indices.uniq.size == 6, 'require six unique NVML indices')
    cfg = config(identity['engine_config'])
    samples = array(raw.fetch('samples'), 'samples')
    check(samples.size == 12, 'require two warmups and ten measured samples')
    indexed, hardware, temperatures = {}, nil, []
    observe = lambda do |data|
      signature, temps = snapshot(data, uuids, indices)
      hardware ||= signature
      check(signature == hardware, 'hardware/SM clocks changed within capture')
      temperatures << temps
    end
    # Earlier preflight snapshots may show work that the collector waited for.
    %w[preflight postflight].each do |key|
      entries = array(raw.fetch(key), key)
      check(entries.size >= 2 && entries.last(2).all? { |s| object(s, key)['idle'] == true },
            "#{key}: missing final two idle checks")
      entries.last(2).each { |s| observe.call(s) }
    end
    telemetry = array(raw.fetch('telemetry'), 'telemetry')
    check(!telemetry.empty?, 'missing telemetry observations')
    telemetry.each { |entry| observe.call(entry) }
    samples.each do |sample|
      sample = object(sample, 'sample')
      check(sample.fetch('status') == 'complete', 'incomplete sample')
      round = integer(sample.fetch('round'), 'round', 1, 2)
      request = integer(sample.fetch('request'), 'request', 0, 5)
      check(sample.fetch('warmup') == request.zero?, 'warmup flag/request index disagree')
      key = [round, request]
      check(!indexed.key?(key), 'duplicate sample')
      check(config(sample.fetch('config')) == cfg, 'engine config changed within capture')
      observe.call(sample.fetch('host_before'))
      observe.call(sample.fetch('host_after'))
      profiles = array(sample.fetch('profiles'), 'profiles').map { |p| profile(p) }.sort_by { |p| p['die'] }
      check(profiles.map { |p| p['die'] } == (0..5).to_a, 'need six unique die profiles')
      waits = profiles.map { |p| p['actq256_us'] }
      indexed[key] = response(sample).merge(
        'warmup' => request.zero?, 'span_us' => profiles.map { |p| p['span_us'] }.max,
        'spread_us' => waits.max - waits.min, 'actq256_us' => waits
      )
    end
    check(indexed.keys.sort == (1..2).flat_map { |r| (0..5).map { |n| [r, n] } }, 'missing protocol sample')
    check(indexed.values.map { |s| s['native_commands'] }.uniq.size == 1,
          'native prompt/controls changed within capture')
    measured = indexed.values.reject { |s| s['warmup'] }
    {
      'label' => label,
      # A placement requires new holders/core; PIDs must stay fixed within a run,
      # but necessarily may differ between a baseline and an authorized reload.
      'identity' => identity.reject { |key, _| %w[engine_config resident_processes].include?(key) },
      'config' => cfg, 'hardware' => hardware, 'samples' => indexed,
      'mean_max_die_span_us' => measured.sum { |s| s['span_us'] } / measured.size,
      'max_actq256_spread_us' => measured.map { |s| s['spread_us'] }.max,
      'mean_actq256_us_by_die' => (0..5).map { |d| measured.sum { |s| s['actq256_us'][d] } / measured.size },
      'mean_decode_tok_s' => measured.sum { |s| s['rate'] } / measured.size,
      'decode_range_tok_s' => measured.map { |s| s['rate'] }.minmax,
      'temperature_range_c_by_die' => (0..5).map { |d| temperatures.map { |t| t[d] }.minmax },
      'measured_output_variants' => measured.map { |s| [s['tokens'], s['output']] }.uniq.size,
      'measured_token_counts' => measured.map { |s| s['tokens'].size }.uniq.sort,
      'measured_requests' => indexed.reject { |_, s| s['warmup'] }.map do |key, s|
        {'round' => key[0], 'request' => key[1], 'actq256_spread_us' => s['spread_us'],
         'max_die_span_us' => s['span_us'], 'decode_tok_s' => s['rate']}
      end
    }
  end

  def analyze(raw_records)
    check(raw_records.is_a?(Array) && !raw_records.empty?, 'supply baseline first')
    records = raw_records.map { |raw| record(raw) }
    check(records.map { |r| r['label'] }.uniq.size == records.size, 'duplicate labels')
    baseline = records.first
    check(baseline['measured_output_variants'] == 1, 'baseline greedy output is not repeatable')
    stable = ->(cfg) { cfg.reject { |key, _| PLACEMENT.include?(key) } }
    reports = records.map.with_index do |candidate, index|
      check(candidate['identity'] == baseline['identity'], 'incomparable capture identity')
      check(candidate['hardware'] == baseline['hardware'], 'incomparable hardware/clocks')
      check(stable.call(candidate['config']) == stable.call(baseline['config']), 'incomparable engine config')
      parity = candidate['samples'].map do |key, sample|
        reference = baseline['samples'].fetch(key)
        check(sample['native_commands'] == reference['native_commands'], 'native prompt/controls differ')
        check(sample['warmup'] || sample['cache_n'] == reference['cache_n'], 'measured cache reuse differs')
        sample['tokens'] == reference['tokens'] && sample['output'] == reference['output']
      end.all?
      changed = PLACEMENT.any? { |key| candidate['config'][key] != baseline['config'][key] }
      failures = []
      failures << 'baseline or unchanged placement is not a new candidate' if index.zero? || !changed
      failures << 'output tokens/content/reasoning/finish changed' unless parity
      failures << 'actq256 spread exceeds 500 us' if candidate['max_actq256_spread_us'] > 500.0 + 1e-8
      failures << 'mean decode is below 35.9 tok/s' if candidate['mean_decode_tok_s'] < 35.9
      candidate.reject { |key, _| %w[identity config hardware samples].include?(key) }.merge(
        'placement' => candidate['config'].select { |key, _| PLACEMENT.include?(key) },
        'output_parity' => parity, 'eligible' => failures.empty?, 'failures' => failures
      )
    end
    ranked = reports.drop(1).sort_by { |r| [r['mean_max_die_span_us'], r['label']] }
    chosen = ranked.find { |r| r['eligible'] }
    [{
      'schema' => 'flash-sass-placement-selection-v1',
      'status' => chosen ? 'measured_gates_pass' : 'partial_no_acceptance',
      'measurement_gates_passed' => !chosen.nil?, 'selected_label' => chosen && chosen['label'],
      'objective' => 'mean of the maximum per-die last-token phase span; includes waits',
      'scope' => 'Ranks supplied measurements only. Model provenance, SPEC recording and PR readiness require review.',
      'baseline' => reports.first, 'candidates_ranked' => ranked
    }, chosen ? 0 : 1]
  rescue EvidenceError, KeyError, TypeError, ArgumentError => error
    [{'schema' => 'flash-sass-placement-selection-v1', 'status' => 'invalid_evidence',
      'measurement_gates_passed' => false, 'selected_label' => nil, 'error' => error.message}, 2]
  end

  def read_capture(path)
    text = path.end_with?('.gz') ? Zlib::GzipReader.open(path, &:read) : File.read(path)
    JSON.parse(text)
  end

  def cli(paths)
    analyze(paths.map { |path| read_capture(path) })
  rescue JSON::ParserError, SystemCallError, Zlib::Error, EOFError => error
    [{'status' => 'invalid_evidence', 'measurement_gates_passed' => false, 'error' => error.message}, 2]
  end
end

if $PROGRAM_NAME == __FILE__
  report, status = PlacementSelect.cli(ARGV)
  puts JSON.pretty_generate(report)
  exit status
end
