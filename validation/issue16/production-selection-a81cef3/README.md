# Production selection capture for issue #16

GPUQ job `44e46c4d-ab5f-40c2-bdfb-3b127665222f` ran the issue-review prompt through the actual Qwen3.8-Flash-Next Q2 model on Archy and captured the live QSA `OP_IDXSEL` output. The job exited 0. Its code source was `a81cef3645ae8fa422d65de43b8b1d1410eac56b`, the PR branch head used to stage and run the launcher.

The model ran all 48 blocks with `MODEL_ATTN=qsa`, the live holder manifests, and the local PLE shard. It processed the 2,697-token prompt and one greedy output token (2,698 positions total). The captured selection came from attention block 47, die 5, at position 2052: a 512-block budget selected 512 sorted IDs from 513 candidate blocks. The row's FNV-1a 64-bit signature is `7d9cf74cdc279b06`; the saved 512-entry ID file has SHA-256 `4f48b12f469f7e6672312ecb031ee922aec30aa3c0128ac325a73962bcfdf467`. The prompt file is [`../production-prompt.txt`](../production-prompt.txt); [`../production-prompt-tokenizer.md`](../production-prompt-tokenizer.md) records the pinned tokenizer source, range, hashes, and rendering settings. The rendered prompt and token IDs are also preserved in the raw archive.

The launcher and benchmark were built through DSH on Benji from that source revision (build `build-3919491-1792578277376915060`). Key artifact hashes were:

- `model_test`: `05260ae3f1a1d90a2b61d76035fea4032c9fa66918b0f74f7c3d76e744238249`
- `qsa_gather_bench`: `246caa8f78d65579b7d6cdc8005a69357972b5b911702d5a67f0243df03af892`
- `issue16_production_prompt_ids`: `6b241772ef1471c0bdeed285301b8786358b24eab766af099236db7f5e6aa543`
- `issue16_production_launcher`: `df2a02689ff87d2413d9c740daa025d365b46d886643c5b989e58a9c13c59bbd`
- `die.cubin`: `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`

## Paired gather validation

The captured row was used unchanged for paired `host_kv=0` and `host_kv=1` arms at `nv=513`, `tail=1`, and position range 2052–2052. The block budget was 512 and the tail residue was 0. Each arm had 10 warmup launches and 8 measured iterations. Both report 2,049 logical keys, 2,052 physically gathered rows, 2,232,576 source unique-address bytes, and 8,404,992 destination unique-address bytes. The realized placements are device VRAM and pinned system memory, respectively.

Every gathered output passed the exact C reference preflight (`preflight_verified_all=true`). The pair check is [`true`](gather-pair-check.json); the arms also have the same cache signature (`e3a8f87d8c355d72`), selection-table signature (`d9531977b3ef11f5`), captured row signature (`7d9cf74cdc279b06`), and verified output signature (`a80de7c117200fb4`). Fixture allocation and upload are outside the timed kernel interval. The reported CUDA-event p50s are:

| Placement | `host_kv` | p50 per gather launch |
| --- | ---: | ---: |
| Device VRAM | 0 | 204.671993852 µs |
| Pinned system memory | 1 | 3523.551940918 µs |

The standalone details are in [`production-manifest.json`](production-manifest.json), [`gather-host0.json`](gather-host0.json), [`gather-host1.json`](gather-host1.json), [`gather-host0.csv`](gather-host0.csv), [`gather-host1.csv`](gather-host1.csv), and [`production-selection.ids.u32`](production-selection.ids.u32). The complete captured GPUQ document is [`raw-results.tar.gz`](raw-results.tar.gz), document 81336, SHA-256 `eff19224de038d9465c55c3859f4cc200bd13cba2b1353677f1387bfddca2051`.

The selection IDs are from a real full-model prompt execution. The gather benchmark's position/channel-signature INT8 K/V and FP16 scale arrays are finite synthetic data; this validates gather correctness and placement costs for the production selection layout, not whole-model K/V traffic or end-to-end decode performance. The run did not have an independent scalar full-model twin (`model_test_scalar_twin=false`), and these measurements do not establish that gather caused the historical decode slowdown.
