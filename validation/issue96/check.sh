#!/bin/bash
# Build phase runs through dsh build; GPU phase runs through Archy's gpuq.
set -euo pipefail
llama=${LLAMA_CPP:-$HOME/claude/llama.cpp-cuda}
build=${LLAMA_BUILD:-$llama/build}
if [[ ${1-} == --build ]]; then
  incs=(-I/opt/cuda/include -I"$llama/include" -I"$llama/ggml/include" -I"$llama/tools/mtmd")
  libs=(-L"$build/bin" -Wl,-rpath,"$build/bin")
  make -C kern flashvis
  cc -std=gnu11 -O2 -Wall "${incs[@]}" -DVIS_FA_NO_GGML -DVIS_FA_PTX_PATH="\"$PWD/kern/vis_fa.cubin\"" \
    validation/issue96/fa_kernel_test.c kern/vis_fa.c "${libs[@]}" -lggml-base -lcuda -lm -o kern/fa_kernel_test
  cc -std=gnu11 -O2 -Wall "${incs[@]}" validation/issue96/fa_native_test.c "${libs[@]}" \
    -lggml -lggml-base -lm -o kern/fa_native_test
  cc -std=gnu99 -O2 -Wall -Wl,--export-dynamic "${incs[@]}" validation/issue96/gpu_reference.c "${libs[@]}" \
    -lmtmd -lllama -lggml -lggml-base -lm -o kern/vis_gpu_reference
  cc -std=gnu99 -Wall -shared -fPIC "${incs[@]}" validation/issue96/fault.c -ldl -o kern/flashvis_fault.so
  ruby validation/issue96/supervisor_test.rb
  exit
fi
: "${GPUQ_JOB_ID:?run GPU checks through gpuq}"
: "${GPUQ_RUN_DIR:?run GPU checks through gpuq}"
projector=${1:?projector GGUF}
model=${2:?text GGUF}
image=${3:?image file}
./kern/fa_kernel_test
./kern/fa_native_test "$build/bin"
ruby validation/issue96/recovery.rb kern/flashvis "$projector" "$model" "$image" kern/flashvis_fault.so
./kern/vis_gpu_reference "$projector" "$model" "$image" "$GPUQ_RUN_DIR/reference.bin" "$build/bin"
ruby validation/issue96/encode.rb kern/flashvis "$projector" "$model" "$image"
ruby validation/issue96/compare.rb "$GPUQ_RUN_DIR/reference.bin" "$GPUQ_RUN_DIR/embeddings-0.bin"
bash validation/issue96/identity.sh
