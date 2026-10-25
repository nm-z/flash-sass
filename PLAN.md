- [ ] 8-way exchange rounds
- [x] Port B4 to 8-way (B8; slower than plain here)
- [ ] Copy-one-item trials
- [x] DeltaNet CTA split (DNSPLIT, bit-identical)
- [x] B8 bit-identical to B1 (`--b8diff 48` PASS)
- [ ] B8 speed (verify 73.6 ms a round of 4; plain token 24 ms)
- [ ] juliac AOT runtime
- [ ] Measure JIT path
- [ ] Diff vs llama.cpp
- [ ] Test merged GGUF
- [ ] Verify 580 reboot
- [ ] Cut 180s build time
---
# Flash-Next in one Julia file: what is left

Constraints that still apply: one authored Julia source (`flashnext.jl`); at most one runtime `.ptx` and one `.sass`;
GPU work through `ssh archy gpuq --db /home/nate/.local/state/gpuq/flashnext.sqlite3 run --name flashnext-<what> -- …`;
every test under 60 s.

## Incomplete or partial
- **Where a block's time goes (`--prof FLASHNEXT_CTAPROF=1`, block 4 on die 0, relaxed counters):** about 155 µs of
  the ~650 µs is CTAs waiting for their x. The largest waits are proj and the router for Z, the hc mix output (25 and
  22 µs median), the routed experts for the top 10 (31 µs), and the hc downs for S (15-16 µs).
  `FLASHNEXT_CTADUMP=proj` shows the Z wait is data-bound (x ready about 138-147 µs whatever a CTA's own finish, so
  Z completes about 12 µs after die 0's last hc mix CTA: die 1's half is the late one), plus three effects:
  - a phase's CTA ends spread by about one row group of one warp (hc_up+mix 103-125 µs);
  - CTAs 0, 4, ..., 28 run about 8 µs late in both phases;
  - CTAs that start together get the router's x up to 12 µs apart (contention on the polled words).
  Not measured yet: the pair's skew on one clock (each die's globaltimer has its own offset).
- **Trials of 2026-10-25 on one DFASTBOTH holder (ABBA x2 at the plateau, against 40.62-40.80 tok/s):**
  FLASHNEXT_DFAST (the MoE down's rows on dies 0-3 for their channels and dies 4-7's, which then take no h; an AOT
  share from the links): 41.36-41.44, kept; FLASHNEXT_DNSPLIT (a value head's 128 state rows over 4 CTAs,
  bit-identical): 40.91-40.99, kept; FLASHNEXT_POLLDELAY 400: 41.56-41.62, kept (1600: 40.78-40.82); FLASHNEXT_W2
  with t (the hc down's partials and t as W2): 40.72-40.76, off. A fresh holder or server warms over about 10
  requests (29 to 40.7 tok/s), so A/Bs start at the plateau.
- **GEMV phases are short, not slow code (2026-10-25, dpar):** `--bench phase` on whole matrices streams 84 GB/s
  (Q2_0 gate_exps, 4.61 MB) and 113 GB/s (IQ4_XS qkv, 6.96 MB); in the kernel a die's 8-way share (0.2-2.5 MB a
  phase) runs at 31-54 GB/s after its x (`--prof`: proj 51 us for 2.47 MB, out_proj 35 us for 1.17 MB, hc_down(ffn)
  27 us for 0.83 MB), and a block is about 300 us of such work plus about 190 us of waits. Tried, off (ABBA against
  41.57-42.09 tok/s): FLASHNEXT_PFALL (each CTA's 1/grid of the phase's dense matrices to L2 while x is awaited, the
  first 1.5 MB: 40.56-40.74; 768 KB: 41.13-41.17); FLASHNEXT_PFNEXT (the formats without a bload split prefetch
  their next block, a line a lane: 41.69-41.73 against 41.98-42.05); FLASHNEXT_POLLEXP (doubling poll delays to 800: 40.69-40.73; to
  3200: 39.59-39.61) and POLLDELAY 800 (41.51-41.52) against 400 (41.67-41.76). More warps, fewer registers
  (FLASHNEXT_B1_THREADS / B1_MAXREGS; shared memory keeps 2 CTAs an SM): 512 / 64: 23.79-23.80; 384 / 80:
  31.39-31.40; 256 / 128: 42.02-42.03 (a server now takes each module's CTA size from its k_token!). Clocks
  during decode: 1177 MHz (the maximum) on every die, 84-94 W of 150, no throttle reasons. An isolated phase's
  fixed cost is 6-7 us (the x load 3.8). FLASHNEXT_ROUTERPAIR (the router split over each board's 2 dies, logits
  over the on-board link; bit-identical to the copy): 41.78-41.90 against the copy's 41.82-42.00, off: halving
  the router's 2.6 MB a die does not shorten the token.
- **Where an 8-way block goes (2026-10-25, `--prof`, fresh clock offsets):** die-to-die round trips 1-2 us (k_ping);
  isolated GEMV phases at a die's 8-way shares 70-120 GB/s (`--bench phase` in the server, FLASHNEXT_BENCH_DIV=4: qkv
  IQ4_XS 92, gate Q3_K 70, hc up BF16 120; the gathered experts 50), in the kernel about 50 % slower (proj 47 us after
  its x against 31 isolated); die 6 (ECC) 1.5 % slower isolated, last in proj and the router from its board's late
  input. Wide (10-lane) proj rows: slower (68 vs 92 GB/s). Per block a slow board takes about 55 KB in (z 15, h 21,
  out-proj partials 8-15, hc 10) at 0.6 GB/s and sends about 50 KB. Kept: DNPF (off: 40.32 vs 40.59 tok/s);
  POLLDELAY 400: +0.3-1 % in 3 pairs (always run second: not separated from warm-up yet). After a server start, decode
  warms over about 3 requests (36.8, 38.5, 38.8 tok/s).
- **Plain decode just past flash-sass:** 41.05-41.20 tok/s generated (warm, ABBA) on the 8-way layout with the
  router copy, vs its 40.3 (target: beat it, heading toward 140; practical estimate 192). About 500 µs a block, of
  which the exchange rounds in series and the slow boards' uplinks are most. Steps today, each ABBA on one server:
  W2 floats of 27 bits (18 mantissa bits, 10-bit flag) with the out-proj partials as W2 through RHIER: 40.91 vs
  40.63; the lane norms 4 words a die (were 32): 41.11-41.20 vs 41.00-41.02. Tried and off: the router split again
  with each die's top 10 to every die (FLASHNEXT_RTOP with ROUTERCOPY=0): 36.3 vs 41.05 (the candidates reach die 0
  about 60 µs after its own copy's top 10). Teacher forcing on the oracle (`--teacher --expect`): 1-2 of 128 argmax
  differ, all at near ties (margins 0.02-0.05: outputs 22 and 34), for every module tried.
- **8-way tensor split (the default layout since 29ee5a8, no layer split):** every block line split 1/8 over dies 0-7
  (`splitcheck.jl`: 32 lines equal 1/8; ssm_norm, 128 floats a block, is on every die: each die's value heads are
  scaled by all of it), the final collapse in k_token!, then copies one item at a time, kept when faster:
  - kept: the KV cache on the 4 dies of its KV head (50ab937): 33.85 tok/s vs 33.51 on the fixture, 34.15 vs 26.60
    at 640-768 positions;
  - the exchange: 14.35 tok/s with one 8-byte store a row output a die; 26.62 forwarded in 32-word chunks; 33.51
    with board relays and lane-summed hc partials; 4-byte words with a flag a chunk after a system fence: 30.20
    (slower, reverted; `scratchpad/ch-exchange.patch`).
  Links (`--bench sweep`, 2026-10-25, slots 1/2/4/5): on-board pairs 10.4 GB/s; boards 0-1 3.4-4.0; into boards 2
  and 3 (dies 4-7) 0.60-0.63, out of them 0.71-1.15; the 8-die LL all-gather of 384 floats 77 µs. A block is about
  585 µs (28 ms a token), of which about 230 µs is the slow boards' uplink bytes (about 140 KB in a block); the rest is
  9 exchange rounds in series a block and compute. Open: fewer rounds (copy the router: the logits round; the owners'
  t round), AOT weighted shares (the slow boards' egress), 2 values a word (24-bit floats, 16-bit seq: no fence).
- **PCIe uplinks (lspci, 2026-10-24):** board 0 (dies 0+1) is on CPU port 00:01.0 at Gen3 x8 (slot max x16); board 1
  (2+3) on 00:02.0 at Gen3 x16; board 2 (4+5) on 00:03.0 at Gen2 x16 (trained down from Gen3); board 3 (6+7) on the
  C610 PCH's root port 00:1c.4 at Gen2 x2 (port max x4), then DMI 2.0 (about 2 GB/s, shared with SATA, USB and the
  NIC). The E5-2620 v3's 40 lanes go to the other three boards. Board 3 is the pipeline's last stage and the draft's,
  so its incoming S, the final S to die 0 and the draft's taps all cross the x2 link. Open: BIOS (the PCH port at x4,
  board 2 at Gen3, board 0's slot at x16?), and which work belongs on the PCH board.
  Since the card swap (2026-10-24, Dell T5810, BIOS A34): slots 1 (00:01.0) and 2 (00:02.0) detect no receiver on any
  lane (00:01.0 x0 after a forced retrain; 00:02.0 hidden by the BIOS), with cards 234925 and 120440 that link in slot
  4, and with power cables moved; an M60 without its power cable still links (slot 5), so it is not the cables. Card
  0323117120440 (old dies 6+7) links x2 in slot 4's x16 as it did in the PCH's x4: the x2 was the card. 227101 links x4
  on the PCH. Only 4 dies are up, too few for the 37.6 GB of weights. archy reset once with no log at 9:09 PM, about
  10 s into a 4-die CUDA check on driver 580.
- **DeltaNet op (~29 µs of the critical path per DeltaNet block, about 3 % of a token):** the state update takes about
  21 µs per head per CTA, with 24 heads on 32 CTAs (8 idle). The state reads mostly hit L2 (DNPF prefetch; 3 MB in
  21 µs would be 146 GB/s, above DRAM), so the op is latency-bound, not bandwidth-bound: split heads across CTAs and
  issue the 4 rows' loads before their dot products.
- **Speculative output equal to greedy by construction (2026-10-25):** `--b8diff K` (B1 and B8 at one position over
  blocks 0..K-1, every LL buffer and the MoE down's terms compared bit for bit) passes at K = 1, 2, 4 and 48, and the
  fixture's 128 speculative ids equal greedy's (25.46 tok/s, 50 rounds, mean 2.600 accepted; greedy 33.57 on the
  same cold server). Before, B8 differed from B1 at 1e-7 to 2e-5 relative after block 0 (B4's 4-position GEMV, the
  hc down's passes, the MoE down in union order, LS / PLE / attention sums over 512 threads) and the ids differed at
  about output 20 (the oracle's near tie). B8 now: each position's x in B1's layout and every block through B1's
  blkdot / bcomp (positions in runs of 3 where 4 x slots do not fit, K = 2560), B1's hc down row groups, the down in
  each position's rank order, B1's thread partitions for LS, PLE sums and attention keys. The last difference was
  the top 10's weights: `unsafe_store!` on a raw shared pointer (B8's union tables in dynamic shared memory) was
  split into byte stores, and the low byte of each f32 came out 0 (the weights' low 8 mantissa bits); every such
  access now has `Val(4)`, and `--build-kernels` reports any `st.shared.b8` left.
- **Block 8 (2026-10-25): the draft at its trained block (`dflash.block_size` 8; target_layers [6, 16, 26, 36, 46]
  = S after blocks 5 .. 45; sliding window 2048 on all 5 layers), B8 rounds of up to P8MAX = 8 positions:**
  recurrent state copies, conv and indexer rings SRN = 16 deep (B1 too: a round's rejected positions overwrote slots
  the next round reads), union tables for 80 experts, the MoE down's (chunk, run) units, 256 counters a block.
  `--b8diff 48` PASS; the fixture's speculative ids equal greedy's at SPECP 4 and 8. SPECP 8: 17.30 tok/s (48
  rounds, accepted 1..8: 20/9/5/5/4/1/2/2, mean 2.688, 154.1 ms a round: verify 140.9, draft 13.15 of which the
  head 11.49); SPECP 4: 26.65 (2.453, 90.6 ms); plain 41.45. The verify costs about 16 ms a position; positions 5-8
  are rarely accepted on this (2-bit) target.
- **Multi-group placements fixed (2026-10-25, c73cc2c):** the output head now lives on the last group only (the
  group that runs the collapse; earlier groups' slices waited for a Z that never reached them), split over its
  dies; the draft's head uses those slices (checked to cover all 248320 rows); B8 only with one group; the
  one-token-at-a-time verify waits for every die after each position; the draft waits on every die before
  appending its taps. `--groups 0+1,2+3,4+5,6+7` (12 blocks a group): greedy 26.89 tok/s with ids equal to the
  one-group run; speculative ids equal to greedy at SPECP 4 (14.24 tok/s) and 8 (8.28). One group with
  FLASHNEXT_B8=0: equal to greedy at SPECP 8 (12.37 tok/s). The default (one group, B8) is unchanged.
- **Equal split of every block line (2026-10-25):** FLASHNEXT_DFAST off by default, `ffn_down_exps` and
  `ffn_down_shexp` 1/8 on every die (4,633 MiB of weights each): plain 41.53 tok/s (41.78 with DFAST), speculative
  28.06, `--b8diff 48` PASS.
- **Default install (2026-10-25, 5769888):** archy's `flashnext.sass` is HEAD's module (the old one kept as
  `flashnext.sass.bak-20261025`), the holder and server on the default layout (no FLASHNEXT_* overrides). On it:
  plain 41.78 tok/s on the fixture after 10 warm-up requests, speculative 29.21 (ids equal to greedy), `--b8diff 48`
  PASS. The holder's PLE table: 26.80 of 28.80 GB resident after locking.
- **Speculative decode still slower than plain (2026-10-25, 1339512, warm server, fixture, ids equal to greedy):**
  SPECP 2: 31.64 tok/s (73 rounds, mean 1.767 accepted, 55.4 ms a round: verify 48.0, draft 7.42); 3: 32.36 (2.224,
  68.2 ms, verify 60.8); 4: 31.61 (2.600, 81.0 ms, verify 73.6); plain 41.16-41.71 (ABBA against c1's 40.73-41.75).
  To beat plain at SPECP 4 a round must take under 2.6 x 24 = 62 ms. The draft's 7.3 ms is mostly its head (6.1 ms:
  output.weight's 3 positions on the draft board's 2 dies, 218 MB each, about 36 GB/s). `FLASHNEXT_B8PROF=1` (die
  0, ms a round of 4, before the down change): experts_down 15.5 (B1 1.0 a token), experts 15.4 (2.8), proj 7.4
  (2.9), out_proj 6.2 (2.3), router 6.0 (2.6), hc_up 5.3-5.4 (2.6-2.8), hc_down 3.4-5.6 (1.5-1.7), start/ple 4.0
  (0.4), deltanet 4.2 (0.7), hc_write(attn) 3.9 (0.6). Done: the MoE down's chunks side by side instead of one phase
  per 3 union experts (15.5 -> 12.2), the gate/up's second x run (position 3 alone at K = 2560) on its own 10
  experts only (15.4 -> 13.9). Tried and reverted: every slot's GEMV call unconditional for 3-4 slots so a block's
  decode is shared (k_token8! had 4.0 times k_token!'s PRMTs): slower (verify 96.9 vs 87.5 ms), and at 1 position
  no longer bit-identical. Open: a 4-slot bcomp per format in B1's per-slot order (decode once, FMAs interleaved,
  not 64 decoded values live at once); 4 x slots at K = 2560 (B1's padded layout needs 49,280 bytes, the limit is
  49,152 with 1,280 static: a swizzled B8 layout, the arithmetic unchanged); the draft head on more dies.
- **Speculative decode slower than plain on the 8-way layout (before B1's arithmetic):** B8 (`k_token8!`, 0c486b3),
  output identical to greedy on the fixture. FLASHNEXT_SPECP (positions a round), 128 tokens, two alternating runs each:
  SPECP 2: 22.4/23.6 tok/s (mean 1.767 accepted, 74-78 ms a round); 3: 26.3/28.0 (2.246, 80-85 ms); 4: 27.7/29.2
  (2.688, 91-96 ms); plain 38.8/38.9. A round's exchange bytes scale with its positions over the 0.6 GB/s uplinks of
  dies 4-7 (experts 22.7 ms, down 9.6 ms a round at 4), and bdot4 computes 4 positions whatever SPECP is (70.8 ms
  verify at 2). Open: P-aware GEMV, fewer position-scaled bytes, and the union down's reduction order (depends on
  the round's other positions; a Red ord table summing in each position's own rank order would make it
  P-independent). On the old pair layout it was 51.6 tok/s.
- **juliac executable rebuilt from ba0ed38 (2026-10-25, 255 s on engi, `/home/nate/work/flashnext/exe`, the
  previous one in `exe.bak-20261024`):** the 128-token speculative fixture in 32.3 s from start (8 dies attached
  from the holder in about 1 s each), 27.05 tok/s generated, ids equal to greedy (rc 1: `--expect`'s near tie at
  output 22). archy's root disk had 97 MB free (the install failed until an older bundle was removed).
- **juliac executable (built with FLASHNEXT_EXE=1: the B1 request path precompiled, its cache reads typed):** runs
  the 128-token speculative fixture in 54 s from start (attach about 15 s). Its one cold request generates at 24.3
  tok/s (18.7 before the typing) against the server's warm 52.6. Not yet ahead of time:
  - the draft's graph captures and the head-slice copies (dies 0+1, 6+7) at the first request;
  - the dynamic sites off the request path (`juliac --trim=safe`: 1730, mostly Stage A, benches, the JIT kernel
    path, hold and serve).
- **Merged GGUF:** the file equals the split one (metadata, the 1224 tensors in order, 366 MB of sampled bytes), but
  the engine has not run on it. It lives on engi, mounted on archy by sshfs (63 MB/s sequential). The first holder
  and server on it stalled from the fd 4 bug (fixed in f24dc44), not from sshfs.
- **JIT vs ptx vs sass:** the in-process JIT path (CUDA.jl compile) is not measured on the GPU (its compile alone is
  over 60 s).
- **Longer prompts against llama.cpp's greedy output:** compared only with flash-sass's oracle (33 of 128 leading ids,
  first difference at a near-tie).
- **gpuq observation:** sync time per token is not separated from the op times in `--prof`'s metrics.
  A job that exits non-zero is recorded `lost` with no logs (nm-z/gpuq#83): wrap failing checks in
  `bash -c "...; echo rc=$?"` until that is fixed.
- **Driver 580 (queued for archy's next reboot):** `nvidia-swap-580.service` swaps 470.256.02 for 580.196.07 in
  systemd's offline-update mode and reboots; log in `/var/lib/nvidia-swap/swap.log`, 470 rollback packages in
  `/var/lib/nvidia-swap/470/`. After it: P2P on every pair (`nvidia-smi topo -p2p r`; the `ForceP2P=0x11;
  PeerMappingOverride=1` registry keys are untested on 580), plain and speculative tok/s re-measured, `--check`,
  unified memory (`cuMemAllocManaged`, error 999 under 470's UVM on Linux >= 6.15). `compat/12-8` becomes links to the
  system 580 libraries (the 570 compat libcuda kept as `12-8.570-bak`).
- **Build time:** `--build-kernels` takes 113 s on engi (from 152 s): Julia start and loading `flashnext.jl` 28 s
  (CUDA.jl alone 2.3 s), PTX 69 s (`k_token4!` 56 s of it, compiled in the main process while 3 workers do the rest),
  ptxas 16 s. The bench kernels (`k_phasebench!`, `k_phase4bench!`, `k_gemv4bench!`) are in the module only with
  `FLASHNEXT_BENCHK=1`; otherwise getkernel compiles them with CUDA.jl at first launch (not yet run on a GPU). The
  new module equals the old one up to compiler-numbered names. Open: the 26 s of loading the file, and ptxas in one
  process (11.8's ptxas has no split compile; nvlink of per-group relocatable cubins would allow it).
- **Measurement:** tok/s are single runs (no repeated samples with min and median), so 1-4 % steps are not separated
  from noise. `CUDA.@profile` works on the M60s (CUPTI 11.8) and is unused: host launch gaps and stream waits in the
  verify and pipelined launch paths are not measured.
