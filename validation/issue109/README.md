# Generic pair-body grid-barrier diagnostic

Refs [issue #109](https://github.com/nm-z/flash-sass/issues/109) and
[issue #9](https://github.com/nm-z/flash-sass/issues/9).
This is an experimental discriminator for the intermittent paired engine
failure. No root cause or successful fix is established by this source alone.

The unmodified generator places a CTA barrier and device memory fence between
the two bodies of a generic paired phase. The C postprocessor adds the existing
device-wide generation barrier at that boundary. Fused pair bodies and scalar
phases bypass the insertion. The exchange counter and peer flags remain
published once after both bodies; the insertion does not publish them.

The transform requires a unique generic-pair branch, its nearby mode test and
the following fence. It refuses existing output, aliases, altered anchors and
already patched input. The build verifies the original cubin against the
pinned baseline before applying the transform. The final kernel receipt binds
both generated SASS files, the C transform, op sources and private MaxAs files.
The inserted barrier uses the framework's generation storage, polling and
launch-expiry logic, including predicate-delay padding before its exit branch.

Builds and CPU structural checks run remotely through DSH:

```console
dsh build --host archy --require-tool cargo --require-tool cc -- make -f validation/issue109/Makefile pair-build pair-check --artifact kern/die.cubin --artifact kern/die.sass --artifact kern/issue9-kernel.sha256 --artifact kern/libengine.so --artifact kern/flashcore --artifact kern/model_test --artifact validation/issue9/bin/engine_check --artifact validation/issue109/bin
```

GPU request controls use the inherited `engine-matrix` recipe through Archy's
FIFO, with the diagnostic checkout's full `ENGINE_REVISION` and six verified
live holder PIDs. The private test children import the holders' existing
allocations. Keep the diagnostic cubin in its separate checkout, retain the
unchanged baseline controls and record passing and failing runs. A bounded
series of passing repeats would provide evidence about this discriminator;
it would not establish the original failure's instruction-level cause.

## Measured build checks

| Routed Archy build | Result |
| --- | --- |
| `build-660111-1792511465775211408` | Original baseline hash matched; transformed SASS assembled with 249 registers. Engine/client builds completed. The final structural-test compilation then rejected an unused test constant under `-Werror`, so the overall request exited 2. |
| `build-674345-1792511587574712206` | After removing that unused test constant, strict C builds and the real generated-source structural checks passed, including malformed/missing/duplicate anchor rejection, already patched input, existing output and alias preservation. Exit 0. |

The transformed cubin SHA-256 is
`c2d8fe082adf54dfb35fcd6ef522baa20cf32f9660026ef7a98e40eab6f7e0e8`.
The engine SHA-256 is
`aa9ebd387020f1382e5805e4fc8d790807d0067b941b9af15c26f25a447d7cfd`;
it reports source revision `11ac712913234331088808284524a11effca7208`.
The test-only constant removal does not change either production artifact.
The final receipt also passed verification after copying the artifacts into the
separate Archy checkout. Build excerpts retain both the initial failure and
the corrected successful check. GPU results remain pending.
