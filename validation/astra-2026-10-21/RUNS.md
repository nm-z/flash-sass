# Recorded job inventory

All 72 recorded attempts are retained, including rejected submissions, cancelled jobs, recorder failures, and recovery. The database group is the original location, not a current recommendation to split history. See [the report](README.md) and [exact commands](jobs.json).

| Original database | Job UUID | Label | Status | Exit code |
| --- | --- | --- | --- | --- |
| early-runs | `a7fdcdc1-d903-4eb7-b7a4-c2e7821cc585` | fused-moe-component-v1-isolated-db | failed | 139 |
| early-runs | `7b6a99e6-819b-4229-ad87-a390c9602fe6` | fused-moe-component-v2 | failed | 139 |
| early-runs | `894d54c8-7f70-4949-b2d7-6a2a7296f231` | fused-moe-component-v3 | failed | 127 |
| early-runs | `97e1d335-6c4c-44b8-b52e-93c62a7066b4` | fused-moe-component-v4 | succeeded | 0 |
| early-runs | `24c9adba-50c1-44ce-9faa-2618ce386af1` | fused-moe-fullmodel-v3 | failed | 1 |
| early-runs | `335ca2a7-3082-4767-8880-2dc719608f2b` | fused-moe-resident-ab-v4 | failed | 1 |
| early-runs | `98abaeca-5873-4e33-81a9-1d72b991bcbf` | restore-reference-local-ple | succeeded | 0 |
| early-runs | `2100ef03-7191-4fed-8991-0213b7c9ddbd` | paired-full-head-cost-256 | failed | 1 |
| early-runs | `7dc1f170-4b3a-4553-956c-dea933f4166d` | paired-full-head-cost-64 | succeeded | 0 |
| early-runs | `c0aad286-7035-485f-8364-d2b0cf1a297d` | gpuq command rejected | rejected | 2 |
| early-runs | `57c8ba81-37ff-4848-9af7-1a22a79d878f` | gpuq command rejected | rejected | 2 |
| early-runs | `40d763e4-a58b-4ac1-8a83-aa8561f6106e` | gpuq command rejected | rejected | 2 |
| early-runs | `7fdce9d0-8959-48cf-b215-bce7b3d5c27f` | chunk-resident-ops-v1 | failed | 1 |
| early-runs | `527e9128-0825-457b-9b23-6c173141cef4` | chunk-resident-ops-v2 | failed | 1 |
| early-runs | `04d5a179-8b09-452d-b6e3-a619b1fab68e` | chunk-resident-ops-v3 | succeeded | 0 |
| early-runs | `8faa705b-5b77-4915-8542-9021b55ba239` | chunk-resident-ops-v4 | failed | 1 |
| early-runs | `f56b3050-0e6e-49ac-aa74-0654aee7a827` | native-chunk-resident-ab-v1 | failed | 2 |
| early-runs | `064bcbb4-e30f-4e6b-b7bb-abc5f35fad1a` | chunk-resident-ops-v5 | succeeded | 0 |
| early-runs | `430703d5-bdc9-4ac3-a1cf-fea9a3aad447` | native-chunk-resident-ab-v2 | failed | 2 |
| early-runs | `52e7b787-5332-474f-bf25-7aa1003d2d2b` | native-chunk-resident-ab-v3-diagnostic | failed | 2 |
| early-runs | `e7e618b0-b3d4-4069-a0cc-44640c15308a` | native-chunk-stream-audit | failed | 1 |
| early-runs | `05c5ea52-7685-43f6-83bd-6a6f6e47a5dc` | native-chunk-fenced-ab | failed | 1 |
| early-runs | `6cd83ffe-8a71-4fef-8105-40350f703dd0` | activation-cache-repro-ci | succeeded | 0 |
| early-runs | `6be1fc97-0cfc-4136-b2a9-d87242a2b8e0` | activation-cache-coherent-validation | succeeded | 0 |
| early-runs | `92375122-5304-4ead-a594-56a64fbf6532` | native-chunk-coherent-ab | failed | 2 |
| early-runs | `ccbfa7ed-548c-4191-9ea0-a36fa9b85c40` | native-prefix-state-audit | timed_out | 1 |
| recovery | `bdd56aa4-ad7c-4e05-8625-fd9d18b2e655` | restore-reference-io-fallback | succeeded | 0 |
| early-runs | `7b203f6c-88d0-4246-bc3b-833c02dd38d5` | expert-order-uniform-validation | capture_failed | 0 |
| kernel-audit | `fa0787c1-1bea-42de-8adc-9efdc3f80647` | expert-order-repro-original | failed | 1 |
| kernel-audit | `47937aa8-550c-4baa-a04c-ad0da8ee0568` | expert-order-uniform-validation | succeeded | 0 |
| kernel-audit | `7d5fae0a-9ed1-4fe3-bc37-71303f68971c` | uniform-prefix-state-audit | failed | 1 |
| kernel-audit | `e322e12d-391d-4a18-94dc-e341fcebcdee` | uniform-prefix-repeatability-only | succeeded | 0 |
| kernel-audit | `6fdd6eb2-7255-47c5-98ef-b2c6ac6cea48` | uniform-native-decode-frozen-reference | failed | 1 |
| kernel-audit | `f1f6929b-e40c-46d3-ac37-2757b975f060` | production-scalar-native-prefill-reference | failed | 1 |
| kernel-audit | `dcb94c6a-4c19-43dd-b33f-d1d684f0186b` | production-kernel-matched-native-ab | failed | 1 |
| kernel-audit | `3acbbd6f-5a97-4566-bb3b-2d19c079a7fd` | exact-b8-bf16-parity | succeeded | 0 |
| kernel-audit | `add0e87a-fbbd-4111-9fe5-04d88504f738` | exact-b8-q2-parity | failed | 70 |
| kernel-audit | `c7e8fd87-1474-4083-a5fa-084416860fe1` | exact-b8-q2-parity-r2 | succeeded | 0 |
| kernel-audit | `96bedeb1-dad0-439c-bcad-6e3ff8c2e27a` | mtp-q8-all-widths-parity | cancelled | 143 |
| kernel-audit | `fb7686bb-f84f-4a82-a075-b27ff3aedce6` | exact-b8-recurrent-state-parity | succeeded | 0 |
| kernel-audit | `7bbbf4ff-fbed-4475-80a2-0b9ae2199be6` | mtp-persistent-phase-parity | cancelled | 143 |
| kernel-audit | `739c1ceb-378c-4deb-920e-43d822fd225f` | mtp-q8-wide-prefix-parity-v2 | succeeded | 0 |
| kernel-audit | `ed45aa5e-f550-45d1-b3d3-345697e9266b` | mtp-persistent-wide-prefix-parity-v2 | failed | 2 |
| kernel-audit | `50c218af-c84d-476a-a6ec-bd80ee6c5163` | mtp-persistent-phase-memcheck | failed | 1 |
| kernel-audit | `6706ab8c-899d-46ac-9bef-ded6f7f0988c` | exact-b8-full48-matched-ab-v2 | failed | 1 |
| kernel-audit | `c531e873-c597-4c78-8fcb-d670486f8af8` | mtp-phase-unpatched-fault-location | failed | 99 |
| kernel-audit | `2721a2a5-d6f0-436c-9fb3-e57a64f887fd` | mtp-phase-single-launch-diagnostic | failed | 2 |
| kernel-audit | `df22cfbe-adcd-4cdd-833e-d5f7826ea5e4` | mtp-predicate-schedule-parity-v3 | succeeded | 0 |
| kernel-audit | `e83e7f59-745b-436e-96cd-b7eadc5c0b75` | mtp-shadow-full48-audit-v1 | failed | 1 |
| kernel-audit | `51e44bf6-ea27-4caa-90dd-a65692d0a57b` | mtp-shadow-full48-memory-partition-v2 | failed | 1 |
| kernel-audit | `d7a794f8-6c9a-478e-950f-edda90e168c9` | mtp-shadow-prefix-stage-audit-v3 | failed | 1 |
| kernel-audit | `8dc627e7-6687-43ea-aa04-4da68da7e9cb` | verify-b8-intermediate-state-capture | succeeded | 0 |
| kernel-audit | `b71dc2e8-5306-4629-bc93-f38329b6eb25` | mtp-shadow-completion-before-copy-v4 | failed | 1 |
| kernel-audit | `336f8bd3-9284-43b5-a1e4-ceb66824fd32` | mtp-target-position15-cuda-debug | failed | 1 |
| kernel-audit | `b1147221-640d-42be-90ea-28ebd0a95c38` | mtp-completed-counter-invariants-v5 | failed | 1 |
| kernel-audit | `cddffb63-8abd-4c0e-9678-31e1ea242438` | verify-full48-retain1-poison7 | failed | 1 |
| kernel-audit | `90afe422-647f-45b2-83dd-f4ab89d3eda8` | mtp-early-finished-peer-invariants-v6 | failed | 1 |
| kernel-audit | `c384337c-9280-4b53-acdd-6bda795d472c` | verify-full48-retain1-compact-arenas | succeeded | 0 |
| kernel-audit | `37d9d487-1975-4486-ae52-23e129d0a630` | mtp-frame-schedule-component-parity | succeeded | 0 |
| kernel-audit | `a2bc5c42-a0e3-425b-8699-bb26c08f6353` | production-schedule-exchange-regression | succeeded | 0 |
| kernel-audit | `84c2d15d-924a-4257-8923-1fcb7bed3e55` | mtp-full128-safe-dispatch-schedule | failed | 1 |
| kernel-audit | `d8037316-de4e-411d-ae0b-0b6b5c9f488e` | verify-full48-all8-rollback-boundaries | failed | 1 |
| kernel-audit | `3d6874ff-5d5b-468d-ac4f-6894a82c97a6` | coherent-controller-exchange-regression | succeeded | 0 |
| kernel-audit | `bf84b970-14d6-481b-bd6f-f1109e69172f` | mtp-coherent-controller-component-parity | succeeded | 0 |
| kernel-audit | `68037e60-3e43-455d-a7a1-828410d4829f` | mtp-full128-coherent-controller-start | failed | 1 |
| kernel-audit | `340e5cf1-4912-4d70-a770-fb242a3f0be3` | collective-publication-counter-regression | succeeded | 0 |
| kernel-audit | `4d8e377b-8463-4003-baad-b359cd66c339` | independent-prefetch-counter-regression | succeeded | 0 |
| kernel-audit | `18def9d1-2de3-425c-bcde-91bb2e0e99d5` | mtp-independent-prefetch-component-parity | succeeded | 0 |
| kernel-audit | `a4ee0d93-ff67-4bf9-99c6-a5942e40ab60` | collective-independent-prefetch-regression | succeeded | 0 |
| kernel-audit | `2e850c9e-2acb-44f9-8e5d-79513ff79dbd` | mtp-full128-independent-prefetch | failed | 1 |
| kernel-audit | `bb1ac914-e2a4-487d-990a-9adc45ae7558` | collective-publication-full48-ab | failed | 1 |
| kernel-audit | `9d19f0e1-e980-416d-8923-7aeabbd8d6a0` | mtp-compact-layout-program-audit-v8 | succeeded | 0 |
