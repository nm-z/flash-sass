#!/bin/bash
# chunk-test.sh BLK NPOS LDX [CUBIN]: the prefill chunk program's DeltaNet half-block of block BLK on GPUs 6 and 7 (kern/chunk_test.c):
# a full tile cache of the block (written once, by a holder without split on GPU 6), the two split holders, the test, the holders stopped.
# Run inside gpuq.
set -u
K=/home/nate/codex/flash-goal-140-1200/chain/kern
G=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
B=$1; N=$2; L=$3; CUBIN=${4:-die_slab.cubin}
C=/tmp/ck-cache-b$B
unset SPLIT_SHARES ATTN_MAP ATTN_DIE
export CUDA_DEVICE_ORDER=PCI_BUS_ID HOLD_VISIBLE=1
cd $K || exit 1
waitready() { local m=$1 pid=$2; until [ -f $m.ready ]; do kill -0 $pid 2>/dev/null || { echo "holder for $m died"; return 1; }; sleep 2; done; }
if [ ! -f $C/.done ]; then
	mkdir -p $C; rm -f /tmp/ck-full.manifest.ready
	CUDA_VISIBLE_DEVICES=6 ./hold $G 6 $B $((B + 1)) /tmp/ck-full.manifest cache=$C > /tmp/ck-full.log 2>&1 & HP=$!
	waitready /tmp/ck-full.manifest $HP || { tail -5 /tmp/ck-full.log; exit 1; }
	kill $HP; wait $HP 2>/dev/null; touch $C/.done
fi
rm -f /tmp/ck-h0.manifest.ready /tmp/ck-h1.manifest.ready
CUDA_VISIBLE_DEVICES=6 ./hold $G 6 $B $((B + 1)) /tmp/ck-h0.manifest split=0/2 > /tmp/ck-h0.log 2>&1 & H0=$!
CUDA_VISIBLE_DEVICES=7 ./hold $G 7 $B $((B + 1)) /tmp/ck-h1.manifest split=1/2 > /tmp/ck-h1.log 2>&1 & H1=$!
trap 'kill $H0 $H1 2>/dev/null' EXIT
waitready /tmp/ck-h0.manifest $H0 && waitready /tmp/ck-h1.manifest $H1 || { tail -3 /tmp/ck-h0.log /tmp/ck-h1.log; exit 1; }
CUDA_VISIBLE_DEVICES=6,7 DIE_CUBIN=$CUBIN ./chunk_test $C $B $N $L /tmp/ck-h0.manifest /tmp/ck-h1.manifest
