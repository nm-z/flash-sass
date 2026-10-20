# frozen_string_literal: true
# All data in this file is SYNTHETIC. It is not a hardware measurement.
module DieBalanceFixtures
  def fixture_config(split = "112,97,81,116,97,97")
    "config rev=abcdef1234567 kv_host=0 ple_disk=1 maxt=8192 layers=48/48 " \
      "nblk=512 dies=6 attn_dies=1,4,5 split=#{split} " \
      "attn_map=1,1,1,1,4,4,4,4,5,5,5,5 img_rows=0 two_position=0 " \
      "prefill_chunk=1 cubin=/synthetic/die.cubin cubin_fnv=0123456789abcdef\n"
  end

  def fixture_profile(die, total = 28.0, actq = 5000.0)
    sync = total * 1000.0 - actq
    {"die" => die, "raw" =>
      format("prof: 11 phases, %.2f ms between the first and last timestamp\n", total) +
      format("  actq256      6 phases %9.1f us (%6.1f us each)\n", actq, actq / 6) +
      format("  sync         4 phases %9.1f us (%6.1f us each)\n", sync, sync / 4) +
      "  longest: SYNTHETIC CPU fixture\n"}
  end

  def fixture_rate(run, rate)
    decode_ms = (2000.0 / rate).round(1)
    run["response"]["timings"] = {
      "predicted_n" => 3, "predicted_per_second" => rate,
      "decode_intervals" => 2, "decode_ms" => decode_ms
    }
    run["done"] = format("done 3 1 3 %.1f 20.0 0 0\n", decode_ms + 20.0)
  end

  def fixture_capture(rate: 35.9, spread: 0.5, total: 28.0, split: "112,97,81,116,97,97")
    config = fixture_config(split)
    runs = (0..3).map do |index|
      run = {"index" => index, "warmup" => index.zero?, "config" => config,
        "input_tokens" => [1, 2, 3], "output_tokens" => [101, 102, 103],
        "response" => {"choices" => [{"finish_reason" => "stop", "message" => {"content" => "synthetic"}}],
          "usage" => {"prompt_tokens" => 3, "completion_tokens" => 3}},
        "profiles" => (0...6).map { |die| fixture_profile(die, total - die * 0.1, 5000.0 + (die == 5 ? spread * 1000.0 : 0.0)) }}
      fixture_rate(run, rate)
      run
    end
    {"schema" => DieBalance::SCHEMA, "label" => "SYNTHETIC CPU fixture",
      "created_at" => "2026-01-01T00:00:00Z", "hostname" => "synthetic",
      "request" => Marshal.load(Marshal.dump(DieBalance::REQUEST)), "warmup_requests" => 1,
      "manifest" => {"config_before" => config, "config_after" => config,
        "frontend_sha256" => "a" * 64, "frontend_env" => {}, "core_exe_sha256" => "b" * 64,
        "engine_sha256" => "c" * 64, "source_revision" => "d" * 40,
        "model" => {"identity_kind" => "stat", "path" => "/synthetic/model.gguf", "realpath" => "/synthetic/model.gguf",
          "device" => 1, "inode" => 2, "size" => 3, "mtime_ns" => 4},
        "hardware" => {"identity" => "SYNTHETIC GPU identities", "pci_link_snapshot" => "SYNTHETIC PCI links"}},
      "runs" => runs}
  end
end
