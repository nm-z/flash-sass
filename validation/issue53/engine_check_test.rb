#!/usr/bin/env ruby
# CPU-only protocol and rollback checks; never connects to a real engine.
require 'tmpdir'
require_relative 'engine_check'

def check(condition, message)
  raise message unless condition
end

def rejects(message)
  begin
    yield
  rescue StandardError
    return
  end
  raise "Accepted invalid input: #{message}"
end

raw = (0..5).map do |die|
  "== die #{die}\nprof: 1800 phases, 25.00 ms between the first and last timestamp\n" +
    "  bf16hcup 97 phases 1000.0 us (10.3 us each)\n" +
    "  hcwsum 48 phases 1000.0 us (20.8 us each)\n" +
    "  bf16hcrwx 48 phases 1000.0 us (20.8 us each)\n" +
    "  bf16hcrx 49 phases 1000.0 us (20.4 us each)\n"
end.join
check(HCCheck.profile(raw).values.all? { |d| d[:hc_ms] == 4.0 }, 'Wrong per-die HC sum')
rejects('missing die') { HCCheck.profile(raw.sub('== die 5', '== die 4')) }
rejects('wrong phase count') { HCCheck.profile(raw.sub('hcwsum 48', 'hcwsum 47')) }
rejects('missing kind') { HCCheck.profile(raw.sub('bf16hcrwx', 'unrelated')) }
extra = raw.gsub("  bf16hcup", "  bf16hcx 1 phases 500.0 us (500.0 us each)\n  bf16hcup")
check(HCCheck.profile(extra).values.all? { |d| d[:hc_ms] == 4.5 }, 'Additional HC kind was omitted')
fixture = {'ids' => [1, 2], 'stop' => [99], 'max_tokens' => 4}
response = "ok: stop 99 -1\ntok 3\ntok 99\ndone 2 1 2 8.6 1.2 0 0\n"
check(HCCheck.tokens(response, fixture)[:ids] == [3, 99], 'Wrong generated IDs')
rejects('truncated token list') { HCCheck.tokens(response.sub('tok 3' + "\n", ''), fixture) }
rejects('no terminal done') { HCCheck.tokens("tok 3\n", fixture) }
rejects('wrong prompt size') { HCCheck.tokens(response.sub('done 2 1 2', 'done 2 1 3'), fixture) }
rejects('interrupted response') { HCCheck.tokens(response.sub('done 2 1 2', 'done 2 0 2'), fixture) }
rejects('prefix reuse') { HCCheck.tokens(response.sub('8.6 1.2 0 0', '8.6 1.2 1 0'), fixture) }
puts 'PASS profile and token protocol checks'

# Exercise Runner#run's actual ensure/rescue paths using fake engine operations.
# /proc is read only for this Ruby process; no CUDA, shell command or socket runs.
class MockRunner < HCCheck::Runner
  attr_reader :restore_calls
  def initialize(opt, original, library, failure)
    @mock_original, @mock_library, @failure = original, library, failure
    @restore_calls = 0
    super(opt)
  end
  def log(_message); end
  def identity; 'mock-core'; end
  def lock_file; File.open(File.join(@o[:out], 'mock.lock'), 'a'); end
  def idle(_name); end
  def loaded_library; @mock_library; end
  def config(name)
    c = @mock_original.dup
    c['rev'] = 'wrong' if name == 'restored-config' && @failure == 'restore'
    c
  end
  def reload(name, library, cubin)
    check(File.file?(library) && File.file?(cubin), 'Missing reload inputs')
    @touched = true
    @restore_calls += 1 if name == 'restore-engine'
  end
  def run_arm(name, directory, _revision, _fixtures)
    reload(name, File.join(directory, 'libengine.so'), File.join(directory, 'die.cubin'))
    raise 'injected arm failure' if name == @failure || (name == 'candidate' && @failure == 'restore')
  end
end

abort 'Set FLASH_SOCK=/tmp/issue53-cpu-mock before starting this test' unless ENV['FLASH_SOCK'] == '/tmp/issue53-cpu-mock'
Dir.mktmpdir('issue53-engine-check-test-') do |dir|
  library = File.join(dir, 'libengine.so')
  cubin = File.join(dir, 'die.cubin')
  File.write(library, 'fake library')
  File.write(cubin, 'a')
  check(HCCheck.fnv(cubin) == 'af63dc4c8601ec8c', 'Wrong FNV identity')
  original = {'rev' => 'mockrev', 'cubin' => cubin, 'cubin_fnv' => HCCheck.fnv(cubin), 'dies' => '6', 'layers' => '48/48'}
  HCCheck::MATCH.each { |key| original[key] ||= '0' }
  fixtures = File.join(dir, 'fixtures.json')
  File.write(fixtures, JSON.generate({'cases' => [fixture.merge('name' => 'test')]}))
  %w[baseline candidate restore].each do |failure|
    out = File.join(dir, failure)
    runner = MockRunner.new({out: out, socket: ENV.fetch('FLASH_SOCK'), core_pid: Process.pid,
                             fixtures: fixtures, baseline_dir: dir, candidate_dir: dir,
                             baseline_rev: 'base', candidate_rev: 'candidate', wait_seconds: 1}, original, library, failure)
    runner.run
    result = JSON.parse(File.read(File.join(out, 'summary.json')))
    check(runner.restore_calls == 1, "Restoration skipped after #{failure} failure")
    check(!runner.success? && result['status'] == 'ERROR', 'Failure reported as success')
    if failure == 'restore'
      check(result['restoration'] != 'PASS' && result['error'].include?('Restoration mismatch'), 'Restoration failure hidden')
    else
      check(result['restoration'] == 'PASS' && result['error'].include?('injected arm failure'), 'Original was not restored after failure')
    end
  end
end
puts 'PASS baseline/candidate failure restoration and restoration-mismatch rejection'
