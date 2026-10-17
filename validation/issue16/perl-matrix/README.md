# Perl gather runner validation

The runner exited 0 after all 40 matrix pairs: complete-block counts 128 and 2000, tails 0 through 3, and five layouts. Each pair includes VRAM and pinned-host KV arms, ten warm-up launches per arm, eight timed launches per arm, and comparison of every gathered value before timing is accepted. This is 80 arms and 640 recorded timed launches.

`report.json` records exact commands, source and executable/cubin hashes, selected-ID identities and all per-arm results. The tested C benchmark reports source revision `7c6f34d4372567fd97f1cef9b609d3dc0c81b0d3`; this runner-only change does not rebuild that binary. The captured-format generator is explicitly named in the report and creates fixtures, not model captures. The process helper separately passed ten CPU checks in `helper.tap`.

Reproduce from `kern` on an exclusively reserved idle GPU:

```sh
CUDA_VISIBLE_DEVICES=GPU-... perl run_qsa_gather_pair.pl \
  --out /new/output/directory --nv 2000 --iterations 8 --matrix
perl QsaEvidence_test.pl
```

The hardware-counter gate remains unrun. These measurements do not establish true DRAM or PCIe transaction counts, whole-model throughput, or the cause of the historical context-length anomaly. Arm order is fixed, so the timings are a verified correctness pairing rather than a randomized placement-speed experiment.
