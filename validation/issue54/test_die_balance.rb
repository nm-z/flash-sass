# frozen_string_literal: true
# CPU-only rejection/selection tests. Fixture numbers are never performance evidence.
require "minitest/autorun"
require "tmpdir"
require "fileutils"
require "stringio"
require_relative "../../kern/die_balance"
require_relative "fixtures"

class TestDieBalance < Minitest::Test
  include DieBalanceFixtures

  def setup
    @directory = Dir.mktmpdir("synthetic-die-balance-")
    @serial = 0
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def save(data)
    @serial += 1
    path = File.join(@directory, "#{@serial}.json")
    File.write(path, JSON.generate(data))
    path
  end

  def analyze(candidate, baseline = fixture_capture)
    DieBalance.analyze(save(baseline), [save(candidate)])
  end

  def rejected(data)
    result = analyze(data)
    assert_equal "rejected", result[:status]
    assert_nil result[:selected]
    refute_empty result[:candidates].first[:failures]
    result
  end

  def replace_config(data)
    new_config = yield data["manifest"]["config_before"]
    data["manifest"]["config_before"] = data["manifest"]["config_after"] = new_config
    data["runs"].each { |run| run["config"] = new_config }
  end

  def test_inclusive_rate_and_spread_boundaries
    report = analyze(fixture_capture)
    assert_equal "accepted", report[:status], report.inspect
    assert_equal 0.5, report[:candidates].first[:worst_spread_ms]
    assert_in_delta 35.9, report[:candidates].first[:mean_decode_tok_s], 1e-10
  end

  def test_collector_revision_does_not_impersonate_runtime_revision
    assert DieBalance.load_capture(save(fixture_capture))[:valid]
    data = fixture_capture
    replace_config(data) { |raw| raw.sub("rev=abcdef1234567", "rev=abcdef1234567-dirty") }
    assert DieBalance.load_capture(save(data))[:valid]
  end

  def test_worst_spread_checks_each_measured_run
    data = fixture_capture
    data["runs"][2]["profiles"][5] = fixture_profile(5, 27.5, 5500.1)
    rejected(data)
  end

  def test_rate_uses_arithmetic_mean_and_excludes_warmup
    data = fixture_capture
    [1.0, 35.0, 35.9, 36.8].each_with_index { |rate, index| fixture_rate(data["runs"][index], rate) }
    assert_equal "accepted", analyze(data)[:status]
    rejected(fixture_capture(rate: 35.89))
  end

  def test_bad_rates_fail_closed
    [nil, "NaN", "35.9", -1, 0].each do |value|
      data = fixture_capture
      data["runs"][1]["response"]["timings"]["predicted_per_second"] = value
      rejected(data)
    end
    path = save(fixture_capture)
    File.write(path, File.read(path).sub('"predicted_per_second":35.9', '"predicted_per_second":NaN'))
    refute DieBalance.load_capture(path)[:valid]
  end

  def test_http_and_raw_timing_reconciliation
    data = fixture_capture
    data["runs"][1]["response"]["timings"]["decode_ms"] = 1000.0
    rejected(data)
    data = fixture_capture
    data["runs"][1]["done"] = "done 3 1 3 1000.0 20.0 0 0\n"
    rejected(data)
  end

  def test_exact_tokens_not_rendered_text
    %w[input_tokens output_tokens].each do |key|
      data = fixture_capture
      data["runs"].each { |run| run[key][-1] = 999 }
      rejected(data)
    end
    data = fixture_capture
    data["runs"][2]["output_tokens"][0] = 999
    rejected(data)
  end

  def test_output_counts_match_raw_ids
    data = fixture_capture
    data["runs"][1]["response"]["usage"]["completion_tokens"] = 2
    rejected(data)
    data = fixture_capture
    data["runs"][1]["done"].sub!("done 3", "done 2")
    rejected(data)
  end

  def test_all_six_dies_are_required_once
    data = fixture_capture
    data["runs"][1]["profiles"].pop
    rejected(data)
    data = fixture_capture
    data["runs"][1]["profiles"][5]["die"] = 4
    rejected(data)
  end

  def test_profile_completeness_and_reconciliation
    mutations = [
      ->(raw) { raw.sub(/.*actq256.*\n/, "") },
      ->(raw) { raw.sub(/.*longest:.*\n/, "") },
      ->(raw) { raw + "  actq256 6 phases 1.0 us (0.2 us each)\n" },
      ->(raw) { raw.sub("5000.0 us", "NaN us") },
      ->(raw) { raw.sub("actq256      6 phases", "actq256      7 phases") },
      ->(raw) { raw.sub("28.00 ms", "29.00 ms") }
    ]
    mutations.each do |change|
      data = fixture_capture
      data["runs"][1]["profiles"][0]["raw"] = change.call(data["runs"][1]["profiles"][0]["raw"])
      rejected(data)
    end
  end

  def test_warmup_and_three_repetitions_required
    data = fixture_capture
    data["runs"].shift
    rejected(data)
    data = fixture_capture
    data["runs"].pop
    rejected(data)
    data = fixture_capture
    data["warmup_requests"] = 0
    rejected(data)
  end

  def test_config_must_not_change_during_capture
    data = fixture_capture
    data["runs"][2]["config"] += " bad=1"
    rejected(data)
  end

  def test_cross_capture_runtime_and_placement_validation
    [["kv_host=0", "kv_host=1"], ["0123456789abcdef", "fedcba9876543210"],
     ["dies=6", "dies=5"], ["112,97,81,116,97,97", "112,97,81,116,194"],
     ["attn_map=1", "attn_map=6"]].each do |old, new_value|
      data = fixture_capture
      replace_config(data) { |raw| raw.sub(old, new_value) }
      rejected(data)
    end
  end

  def test_manifest_identity_must_match
    %w[frontend_sha256 core_exe_sha256 engine_sha256].each do |key|
      data = fixture_capture
      data["manifest"][key] = "e" * 64
      rejected(data)
    end
    data = fixture_capture
    data["manifest"]["model"].delete("inode")
    rejected(data)
  end

  def test_exact_request_protocol
    data = fixture_capture
    data["request"]["temperature"] = 0.1
    rejected(data)
  end

  def test_minimax_ranks_worst_run_and_never_sums_parallel_dies
    baseline = save(fixture_capture(total: 30.0))
    better = save(fixture_capture(total: 27.0, split: "100,100,100,100,100,100"))
    worse = fixture_capture(total: 20.0)
    worse["runs"][3]["profiles"][0] = fixture_profile(0, 28.0)
    report = DieBalance.analyze(baseline, [save(worse), better])
    assert_equal "accepted", report[:status], report.inspect
    assert_equal better, report[:selected][:path]
    assert_equal 27.0, report[:selected][:objective_ms]
    assert_equal 6, report[:candidates].first[:per_die_max_total_ms].length
  end

  def test_baseline_alone_never_claims_selection
    report = DieBalance.analyze(save(fixture_capture), [])
    assert_equal "pending", report[:status]
    assert_nil report[:selected]
    output = StringIO.new
    DieBalance.print_report(report, output)
    assert_includes output.string, "no acceptance claim"
    refute_includes output.string, "SPLIT_SHARES="
  end

  def test_invalid_baseline_and_duplicate_json_keys
    data = fixture_capture
    data["runs"].pop
    assert_equal "rejected", analyze(fixture_capture, data)[:status]
    path = save(fixture_capture)
    File.write(path, File.read(path).sub('"warmup_requests":1', '"warmup_requests":1,"warmup_requests":1'))
    refute DieBalance.load_capture(path)[:valid]
  end

  def test_json_report_and_exit_status_protect_existing_files
    baseline, candidate = save(fixture_capture), save(fixture_capture)
    output = File.join(@directory, "report.json")
    capture_io { assert_equal 0, DieBalance.cli(["--baseline", baseline, "--json", output, candidate]) }
    assert_equal "accepted", JSON.parse(File.read(output))["status"]
    before = File.read(baseline)
    capture_io { assert_equal 2, DieBalance.cli(["--baseline", baseline, "--json", baseline, candidate]) }
    assert_equal before, File.read(baseline)
    capture_io { assert_equal 1, DieBalance.cli(["--baseline", baseline, save(fixture_capture(rate: 35.0))]) }
  end
end
