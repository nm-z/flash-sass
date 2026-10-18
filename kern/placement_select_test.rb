#!/usr/bin/env ruby
# CPU-only synthetic evidence tests. No model, CUDA, engine or GPU is used.
require_relative 'placement_select'
require 'tmpdir'

module PlacementSelectTest
  module_function

  def copy(value)
    Marshal.load(Marshal.dump(value))
  end

  def assert(value, message = 'assertion failed')
    raise message unless value
  end

  def profile(die, span_us, actq_us)
    rest_count = 1007
    rest_us = span_us - actq_us
    {
      'die' => die,
      'text' => format(
        "prof: 1154 phases, %.2f ms between the first and last timestamp\n" \
        "  actq256 146 phases %.1f us (%6.1f us each)\n" \
        "  gemv %d phases %.1f us (%6.1f us each)\n" \
        "  longest:\n",
        span_us / 1000.0, actq_us, actq_us / 146.0, rest_count, rest_us, rest_us / rest_count.to_f
      )
    }
  end

  def sync_transcript(sample)
    sample['native_responses'] = ["ok stop\n", "ok sample\n", "ok snap\n"] +
      sample['token_ids'].map { |id| "tok #{id}\n" } +
      [sample['engine_done'] + "\n", sample['config'] + "\n"] +
      sample['profiles'].flat_map { |p| p['text'].lines }
  end

  def fixture(label = 'baseline', split = '112,97,81,116,97,97',
              span_us = 28_000.0, spread_us = 2300.0, rate = 35.9)
    uuids = (0..5).map { |n| "GPU-#{n}" }
    cfg = 'config rev=edc66d9 kv_host=0 ple_disk=1 maxt=230000 ' \
          'layers=48/48 nblk=512 dies=6 attn_dies=1,4,5 ' \
          "split=#{split} attn_map=1,1,1,1,4,4,4,4,5,5,5,5 " \
          'img_rows=4096 two_position=0 prefill_chunk=0 cubin=die.cubin ' \
          'cubin_fnv=61f38fc0784534a2'
    host = {
      'gpus' => uuids.map.with_index do |uuid, n|
        {
          'index' => n.to_s, 'uuid' => uuid, 'memory.used' => '7000',
          'memory.total' => '7618', 'utilization.gpu' => '0',
          'temperature.gpu' => (60 + n).to_s, 'clocks.sm' => '1177',
          'pcie.link.gen.current' => '3', 'pcie.link.width.current' => '16'
        }
      end,
      'root_links' => (1..3).map do |n|
        {'pci_address' => "0000:00:0#{n}.0",
         'speed' => n == 3 ? '5.0 GT/s' : '8.0 GT/s', 'width' => '16'}
      end,
      'gpu_processes_csv' => '', 'ps' => '', 'foreign_gpu_processes' => [],
      'resident_identity_errors' => [], 'idle' => true
    }
    ids = (100..137).to_a
    ms = (37_000.0 / rate).round(1)
    samples = (1..2).flat_map do |round|
      (0..5).map do |request|
        sample = {
          'status' => 'complete', 'round' => round, 'request' => request, 'warmup' => request.zero?,
          'config' => cfg, 'token_ids' => ids.dup,
          'engine_done' => format('done 38 1 20 %.1f 15.0 19 0', ms + 15),
          'native_commands' => [
            'stop 137 4294967295', "sample #{round * 100 + request} 0 20 0.95",
            'snap 19', 'chat 39 ' + (1..20).to_a.join(' ')
          ],
          'response' => {
            'choices' => [{
              'index' => 0, 'finish_reason' => 'stop',
              'message' => {'role' => 'assistant', 'content' => 'Paris',
                            'reasoning_content' => 'France has capital Paris.'}
            }],
            'usage' => {'prompt_tokens' => 20, 'completion_tokens' => 38, 'total_tokens' => 58},
            'timings' => {
              'prompt_n' => 20, 'cache_n' => 19, 'prompt_ms' => 15.0, 'predicted_n' => 38,
              'predicted_ms' => ms, 'predicted_per_second' => rate,
              'decode_intervals' => 37, 'decode_ms' => ms
            }
          },
          'profiles' => (0..5).map { |die| profile(die, span_us - die * 10, 3500 + spread_us * die / 5) },
          'host_before' => copy(host), 'host_after' => copy(host)
        }
        sync_transcript(sample)
        sample
      end
    end
    {
      'schema' => 'flash-sass-placement-v1', 'label' => label, 'status' => 'complete', 'failures' => [],
      'protocol' => copy(PlacementSelect::PROTOCOL),
      'identity' => {
        'hostname' => 'archy', 'boot_id' => 'same-boot', 'engine_config' => cfg,
        'frontend_sha256' => 'a' * 64, 'collector_sha256' => 'b' * 64,
        'frontend_props' => {'model' => 'model.gguf'}, 'frontend_environment' => {},
        'model' => {'path' => '/model.gguf', 'size' => 100, 'mtime' => '2026-10-18T00:00:00Z'},
        'device_uuids' => uuids, 'device_indices' => (0..5).to_a,
        'resident_processes' => [{'pid' => 10, 'starttime_ticks' => '123', 'comm' => 'flashcore'}]
      },
      'samples' => samples, 'telemetry' => [copy(host)],
      'preflight' => [copy(host), copy(host)], 'postflight' => [copy(host), copy(host)]
    }
  end

  def candidate
    fixture('candidate', '110,99,81,116,97,97', 27_000, 400, 36)
  end

  def run
    count = 0
    test = lambda do |name, &block|
      block.call
      count += 1
      puts "ok #{count} - #{name}"
    rescue StandardError => error
      warn "not ok #{count + 1} - #{name}: #{error.message}"
      exit 1
    end
    invalid = lambda do |name, &mutation|
      test.call(name) do
        raw = fixture
        mutation.call(raw)
        report, status = PlacementSelect.analyze([raw])
        assert(status == 2 && report['status'] == 'invalid_evidence', report.inspect)
      end
    end
    puts 'TAP version 13'
    test.call('baseline only reports measurements without acceptance') do
      report, status = PlacementSelect.analyze([fixture])
      assert(status == 1 && !report['measurement_gates_passed'], report.inspect)
      assert(report['baseline']['max_actq256_spread_us'] == 2300.0)
    end
    test.call('changed placement passes exact boundary gates with new holders and random seeds') do
      c = fixture('candidate', '110,99,81,116,97,97', 27_000, 500, 35.9)
      c['identity']['resident_processes'][0]['pid'] = 20
      c['samples'].each { |s| s['native_commands'][1] = 'sample 9999999 0 20 0.95' }
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status.zero? && report['selected_label'] == 'candidate', report.inspect)
    end
    test.call('ranking uses busiest die span and skips faster failing candidate') do
      fast = fixture('fast-but-uneven', '110,99,81,116,97,97', 26_000, 501, 36)
      good = fixture('winner', '111,98,81,116,97,97', 27_000, 400, 36)
      other = fixture('other', '113,96,81,116,97,97', 27_500, 300, 36)
      report, status = PlacementSelect.analyze([fixture, other, fast, good])
      assert(status.zero? && report['selected_label'] == 'winner', report.inspect)
      assert(report['candidates_ranked'].map { |r| r['label'] } == %w[fast-but-uneven winner other])
    end
    test.call('one request above spread limit cannot be hidden by averaging') do
      c = candidate
      s = c['samples'][1]
      s['profiles'] = (0..5).map { |d| profile(d, 27_000 - d * 10, 3500 + 501.0 * d / 5) }
      sync_transcript(s)
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status == 1 && report['candidates_ranked'][0]['max_actq256_spread_us'] == 501)
    end
    test.call('warmup speed and spread are excluded from measured gates') do
      c = candidate
      slow = fixture('warmups', '110,99,81,116,97,97', 28_000, 2300, 30)
      [0, 6].each { |i| c['samples'][i] = slow['samples'][i] }
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status.zero? && report['candidates_ranked'][0]['mean_decode_tok_s'] == 36, report.inspect)
    end
    test.call('same text with different hidden token IDs fails parity') do
      c = candidate
      c['samples'][1]['token_ids'][0] = 999
      sync_transcript(c['samples'][1])
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status == 1 && !report['candidates_ranked'][0]['output_parity'], report.inspect)
    end
    test.call('changed reasoning with same IDs fails parity') do
      c = candidate
      c['samples'][1]['response']['choices'][0]['message']['reasoning_content'] = 'Changed'
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status == 1 && !report['candidates_ranked'][0]['output_parity'], report.inspect)
    end
    test.call('slow candidate fails despite level waits') do
      c = fixture('candidate', '110,99,81,116,97,97', 27_000, 400, 35.89)
      report, status = PlacementSelect.analyze([fixture, c])
      assert(status == 1 && !report['measurement_gates_passed'])
    end
    test.call('unchanged placement cannot resolve issue') do
      _, status = PlacementSelect.analyze([fixture, fixture('candidate', '112,97,81,116,97,97', 27_000, 400, 36)])
      assert(status == 1)
    end
    invalid.call('missing measured sample') { |r| r['samples'].delete_at(1) }
    invalid.call('duplicate sample') { |r| r['samples'][2] = copy(r['samples'][1]) }
    invalid.call('warmup flag cannot hide a measured request') { |r| r['samples'][1]['warmup'] = true }
    invalid.call('missing die') { |r| r['samples'][1]['profiles'].pop }
    invalid.call('duplicate die') { |r| r['samples'][1]['profiles'][5]['die'] = 4 }
    invalid.call('truncated profile rows') { |r| r['samples'][1]['profiles'][0]['text'].sub!(/^  gemv.*\n/, '') }
    invalid.call('missing profile terminator') { |r| r['samples'][1]['profiles'][0]['text'].sub!(/^  longest:.*\n/, '') }
    invalid.call('wrong actq count') { |r| r['samples'][1]['profiles'][0]['text'].sub!('actq256 146', 'actq256 145') }
    invalid.call('bad rounded profile total') { |r| r['samples'][1]['profiles'][0]['text'].sub!('28.00 ms', '29.00 ms') }
    invalid.call('wrong HTTP G count') { |r| r['samples'][1]['response']['timings']['predicted_n'] = 37 }
    invalid.call('G rather than G-1 rate') do |r|
      r['samples'][1]['response']['timings']['predicted_ms'] = 38_000.0 / 35.9
    end
    invalid.call('nonfinite timing') { |r| r['samples'][1]['response']['timings']['predicted_ms'] = Float::NAN }
    invalid.call('HTTP JSON error') { |r| r['samples'][1]['response']['error'] = {'message' => 'failed'} }
    invalid.call('native done count differs') { |r| r['samples'][1]['engine_done'].sub!('done 38', 'done 37') }
    invalid.call('native done timing differs') { |r| r['samples'][1]['engine_done'].sub!('15.0', '16.0') }
    invalid.call('native done is truncated') { |r| r['samples'][1]['engine_done'] = 'done 38' }
    invalid.call('HTTP finish reason differs from native') do |r|
      r['samples'][1]['response']['choices'][0]['finish_reason'] = 'length'
    end
    invalid.call('native stop requires matching final stop token') do |r|
      r['samples'][1]['token_ids'][-1] = 777
      sync_transcript(r['samples'][1])
    end
    invalid.call('native transcript tokens cannot disagree') { |r| r['samples'][1]['native_responses'][3] = "tok 999\n" }
    invalid.call('profile summary must match native transcript') { |r| r['samples'][1]['native_responses'].pop }
    invalid.call('sample marked incomplete') { |r| r['samples'][1]['status'] = 'failed' }
    invalid.call('failure list is not ignored') { |r| r['failures'] << 'foreign process' }
    invalid.call('foreign GPU work during telemetry') { |r| r['telemetry'][0]['foreign_gpu_processes'] << {'pid' => 100} }
    invalid.call('missing resident identity checks') { |r| r['telemetry'][0].delete('resident_identity_errors') }
    invalid.call('resident identity changed') { |r| r['postflight'][1]['resident_identity_errors'] << 'PID reused' }
    invalid.call('missing postflight completion') { |r| r['postflight'].pop }
    invalid.call('clock changed after sample') { |r| r['samples'][1]['host_after']['gpus'][0]['clocks.sm'] = '531' }
    invalid.call('link changed during capture') { |r| r['telemetry'][0]['root_links'][0]['speed'] = '2.5 GT/s' }
    invalid.call('non-greedy sampler cannot normalize its seed away') do |r|
      r['samples'][1]['native_commands'][1] = 'sample 123 0.8 20 0.95'
    end
    test.call('revision, model, prompt, cache and frontend mismatches are incomparable') do
      %w[revision model prompt cache frontend].each do |kind|
        c = candidate
        case kind
        when 'revision'
          c['identity']['engine_config'].sub!('edc66d9', '1234567')
          c['samples'].each { |s| s['config'] = c['identity']['engine_config']; sync_transcript(s) }
        when 'model'
          c['identity']['model']['size'] += 1
        when 'prompt'
          c['samples'].each { |s| s['native_commands'][-1].sub!(' 20', ' 21') }
        when 'cache'
          c['samples'][1]['response']['timings']['cache_n'] = 18
          c['samples'][1]['engine_done'].sub!(' 19 0', ' 18 0')
          sync_transcript(c['samples'][1])
        when 'frontend'
          c['identity']['frontend_sha256'] = 'c' * 64
        end
        report, status = PlacementSelect.analyze([fixture, c])
        assert(status == 2, "#{kind}: #{report.inspect}")
      end
    end
    test.call('temperature is reported by die without arbitrary threshold') do
      raw = fixture
      raw['samples'][1]['host_after']['gpus'][0]['temperature.gpu'] = '86'
      report, status = PlacementSelect.analyze([raw])
      assert(status == 1 && report['baseline']['temperature_range_c_by_die'][0] == [60.0, 86.0])
    end
    test.call('plain and gzip files produce the same report and malformed JSON fails') do
      Dir.mktmpdir('placement-selector-test-') do |dir|
        plain, zipped = File.join(dir, 'capture.json'), File.join(dir, 'capture.json.gz')
        File.write(plain, JSON.generate(fixture))
        Zlib::GzipWriter.open(zipped) { |gz| gz.write(File.read(plain)) }
        assert(PlacementSelect.cli([plain]) == PlacementSelect.cli([zipped]))
        File.write(plain, '{broken')
        assert(PlacementSelect.cli([plain])[1] == 2)
      end
    end
    puts "1..#{count}"
  end
end

PlacementSelectTest.run if $PROGRAM_NAME == __FILE__
