# frozen_string_literal: true
# CPU-only UNIX socket tests; no NVIDIA commands or model are executed.
require "minitest/autorun"
require "tmpdir"
require "fileutils"
require "timeout"
require "open3"
require "rbconfig"
require_relative "../../kern/die_balance_capture"
require_relative "fixtures"

class TestDieBalanceCapture < Minitest::Test
  include DieBalanceFixtures

  def setup
    @directory = Dir.mktmpdir("synthetic-balance-relay-")
  end

  def teardown
    FileUtils.remove_entry(@directory)
  end

  def with_relay(handler, peer_pid: Process.pid)
    upstream_path = File.join(@directory, "upstream.sock")
    relay_path = File.join(@directory, "relay.sock")
    server = UNIXServer.new(upstream_path)
    worker = Thread.new do
      core = server.accept
      handler.call(core)
    ensure
      core&.close unless core&.closed?
    end
    worker.report_on_exception = false
    relay = DieBalanceCapture::Relay.new(relay_path, upstream_path, peer_pid, File.join(@directory, "protocol.jsonl"))
    client = UNIXSocket.new(relay_path)
    Timeout.timeout(5) { yield relay, client }
  ensure
    client&.close unless client&.closed?
    relay&.close
    server&.close unless server&.closed?
    if worker
      worker.join(1)
      worker.kill if worker.alive?
      worker.value
    end
  end

  def test_startup_info_finishes_without_chat
    commands = []
    handler = lambda do |core|
      commands << core.gets
      core.write("info 8192 0\n")
      commands << core.gets
    end
    with_relay(handler) do |relay, client|
      client.write("info\n")
      assert_equal "info 8192 0\n", client.gets
      client.close
      relay.check!
    end
    assert_equal ["info\n", nil], commands
  end

  def test_pipelined_commands_and_tokens_use_one_connection_for_profiles
    observed = []
    handler = lambda do |core|
      [["stop 1 2\n", "ok: stop 1 2\n"], ["sample 7 0 32 1\n", "ok: sample seed 7 temp 0 top_k 32 top_p 1 (greedy)\n"],
       ["snap 3\n", "ok: snap 3\n"]].each do |_command, reply|
        observed << core.gets
        core.write(reply)
      end
      observed << core.gets
      # Deliberately one write: catches select/gets buffering deadlocks.
      core.write("tok 101\ntok 102\ntok 103\ndone 3 1 3 75.7 20.0 0 0\n")
      observed << core.gets
      core.write(fixture_config)
      6.times do |die|
        observed << core.gets
        core.write(fixture_profile(die)["raw"])
      end
      observed << :all_profiles_sent
      core.gets # frontend closes only after consuming done
    end
    with_relay(handler) do |relay, client|
      client.write("stop 1 2\nsample 7 0 32 1\nsnap 3\nchat 39 1 2 3\n")
      replies = []
      replies << client.gets until replies.last&.start_with?("done ")
      assert_equal 7, replies.length
      assert_includes observed, :all_profiles_sent
      evidence = relay.pop
      assert_equal [1, 2, 3], evidence["input_tokens"]
      assert_equal [101, 102, 103], evidence["output_tokens"]
      assert_equal (0...6).to_a, evidence["profiles"].map { |profile| profile["die"] }
      assert_equal fixture_config, evidence["config"]
      client.close
    end
    assert_equal ["stop 1 2\n", "sample 7 0 32 1\n", "snap 3\n", "chat 39 1 2 3\n", "config\n"] +
      (0...6).map { |die| "prof #{die} all\n" } + [:all_profiles_sent], observed
  end

  def test_dangerous_command_never_reaches_core
    observed = []
    with_relay(->(core) { observed << core.gets }) do |relay, client|
      client.write("quit\n")
      error = assert_raises(DieBalanceCapture::CaptureError) { relay.pop }
      assert_match(/refusing frontend command/, error.message)
    end
    assert_equal [nil], observed
  end

  def test_wrong_peer_pid_is_rejected_before_command
    observed = []
    with_relay(->(core) { observed << core.gets }, peer_pid: Process.pid + 100_000) do |relay, _client|
      error = assert_raises(DieBalanceCapture::CaptureError) { relay.pop }
      assert_match(/core peer PID/, error.message)
    end
    assert_equal [nil], observed
  end

  def test_engine_error_and_disconnect_do_not_return_evidence
    ["error: synthetic failure\n", "", "done malformed\n"].each do |reply|
      FileUtils.rm_f(File.join(@directory, "upstream.sock"))
      FileUtils.rm_f(File.join(@directory, "protocol.jsonl"))
      handler = ->(core) { core.gets; core.write(reply) unless reply.empty? }
      with_relay(handler) do |relay, client|
        client.write("chat 39 1 2 3\n")
        assert_raises(DieBalanceCapture::CaptureError) { relay.pop }
      end
    end
  end

  def test_missing_profile_footer_fails
    handler = lambda do |core|
      core.gets
      core.write("tok 101\ntok 102\ntok 103\ndone 3 1 3 75.7 20.0 0 0\n")
      core.gets
      core.write(fixture_config)
      core.gets
      core.write(fixture_profile(0)["raw"].sub(/.*longest:.*\n/, ""))
    end
    with_relay(handler) do |relay, client|
      client.write("chat 39 1 2 3\n")
      assert_raises(DieBalanceCapture::CaptureError) { relay.pop }
    end
  end

  def test_busy_standalone_lock_waits_for_its_owner
    lock_path = File.join(@directory, "engine.lock")
    worker = nil
    File.open(lock_path, "w") do |lock|
      assert lock.flock(File::LOCK_EX | File::LOCK_NB)
      entered = Queue.new
      acquired = Queue.new
      worker = Thread.new do
        entered << true
        lease = DieBalanceCapture.validation_lock(lock_path, {})
        acquired << true
      ensure
        lease&.close
      end
      Timeout.timeout(2) { entered.pop }
      refute worker.join(0.05), "another owner must retain exclusive access"
      assert acquired.empty?
      lock.close
      Timeout.timeout(2) { worker.value }
      assert acquired.pop
    end
  ensure
    worker&.kill&.join if worker&.alive?
  end

  def test_queued_collector_reuses_and_preserves_the_inherited_lease
    lock_path = File.join(@directory, "engine.lock")
    script = <<~RUBY
      require #{File.expand_path('../../kern/die_balance_capture', __dir__).inspect}
      lease = DieBalanceCapture.validation_lock(ARGV.fetch(0))
      lease.close
      puts "inherited lease accepted"
    RUBY
    File.open(lock_path, "w") do |parent|
      assert parent.flock(File::LOCK_EX | File::LOCK_NB)
      out, err, status = Open3.capture3({"GPUQ_JOB_ID" => "synthetic-parent", "GPUQ_LOCK_FD" => "9"},
        RbConfig.ruby, "-e", script, lock_path, 9 => parent)
      assert status.success?, err
      assert_equal "inherited lease accepted\n", out
      File.open(lock_path, "r+") do |probe|
        refute probe.flock(File::LOCK_EX | File::LOCK_NB), "collector cleanup must not unlock its parent"
      end
    end
  end

  def test_inherited_descriptor_must_refer_to_the_requested_lock
    path = File.join(@directory, "engine.lock")
    wrong = File.join(@directory, "wrong.lock")
    File.write(path, "")
    File.open(wrong, "w") do |file|
      error = assert_raises(DieBalanceCapture::CaptureError) do
        DieBalanceCapture.validation_lock(path, {"GPUQ_JOB_ID" => "synthetic", "GPUQ_LOCK_FD" => file.fileno.to_s})
      end
      assert_match(/does not match/, error.message)
      refute file.closed?
    end
  end

  def test_independent_descriptor_cannot_impersonate_the_owners_lease
    path = File.join(@directory, "engine.lock")
    File.open(path, "w") do |owner|
      assert owner.flock(File::LOCK_EX | File::LOCK_NB)
      File.open(path, "r+") do |other|
        assert_raises(DieBalanceCapture::CaptureError) do
          DieBalanceCapture.validation_lock(path, {"GPUQ_JOB_ID" => "synthetic", "GPUQ_LOCK_FD" => other.fileno.to_s})
        end
        refute other.closed?
      end
    end
  end

  def test_incomplete_queue_environment_is_rejected
    [{"GPUQ_JOB_ID" => "synthetic"}, {"GPUQ_LOCK_FD" => "9"},
     {"GPUQ_JOB_ID" => "synthetic", "GPUQ_LOCK_FD" => "invalid"}].each do |environment|
      assert_raises(DieBalanceCapture::CaptureError) do
        DieBalanceCapture.validation_lock(File.join(@directory, "uncreated.lock"), environment)
      end
    end
    refute File.exist?(File.join(@directory, "uncreated.lock"))
  end

  def test_existing_evidence_is_never_overwritten_on_error
    sentinel = File.join(@directory, "sentinel")
    File.write(sentinel, "retain this")
    capture_io do
      assert_equal 1, DieBalanceCapture.run(["--out", @directory, "--label", "synthetic",
        "--core-pid", "123", "--flashd-pid", "124", "--lock", File.join(@directory, "engine.lock")])
    end
    assert_equal "retain this", File.read(sentinel)
    refute File.exist?(File.join(@directory, "error.txt"))
  end

  def test_engine_identity_finds_the_requested_cores_loaded_generation
    maps = <<~MAPS
      1000-2000 r-xp 00001000 08:13 100 /usr/lib/libc.so.6
      2000-3000 r--p 00001000 00:2b 200 /tmp/flashcore-123-gen21.so
      3000-4000 r-xp 00002000 00:2b 200 /tmp/flashcore-123-gen21.so
      4000-5000 r-xp 00002000 00:2b 300 /tmp/flashcore-124-gen1.so
    MAPS
    row = DieBalanceCapture.engine_mapping(123, maps)
    assert_equal "/tmp/flashcore-123-gen21.so", row[5]
    assert_equal "200", row[4]
    assert_raises(DieBalanceCapture::CaptureError) { DieBalanceCapture.engine_mapping(125, maps) }
    duplicate = maps + "5000-6000 r-xp 00002000 00:2b 400 /tmp/flashcore-123-gen22.so\n"
    assert_raises(DieBalanceCapture::CaptureError) { DieBalanceCapture.engine_mapping(123, duplicate) }
    direct = "3000-4000 r-xp 00002000 00:2b 200 /private/libengine.so\n"
    assert_equal "/private/libengine.so", DieBalanceCapture.engine_mapping(123, direct)[5]
    deleted = maps.sub("3000-4000 r-xp 00002000 00:2b 200 /tmp/flashcore-123-gen21.so",
      "3000-4000 r-xp 00002000 00:2b 200 /tmp/flashcore-123-gen21.so (deleted)")
    assert_equal "/tmp/flashcore-123-gen21.so (deleted)", DieBalanceCapture.engine_mapping(123, deleted)[5]
  end

  def test_numeric_nvidia_queries_need_no_csv_gem
    uuid = "GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f"
    patterns = [/\A\d+\z/, DieBalanceCapture::GPU_UUID, /\A(?:100|[1-9]?\d)\z/]
    assert_equal [["0", uuid, "0"], ["5", uuid, "100"]],
      DieBalanceCapture.nvidia_rows("0, #{uuid}, 0\n5, #{uuid}, 100\n", patterns)
    assert_empty DieBalanceCapture.nvidia_rows("", [/\A[1-9]\d*\z/, DieBalanceCapture::GPU_UUID])
    ["0, #{uuid}, N/A\n", "0, #{uuid}, 101\n", "0, #{uuid}\n", "0, #{uuid}, 0, extra\n",
     "0, invalid-uuid, 0\n", "\"0\", #{uuid}, 0\n", "\n"].each do |raw|
      assert_raises(DieBalanceCapture::CaptureError) { DieBalanceCapture.nvidia_rows(raw, patterns) }
    end
  end

  def test_child_cleanup_does_not_signal_nonchildren
    DieBalanceCapture.stop_owned_child(Process.pid)
    assert Process.kill(0, Process.pid)
  end
end
