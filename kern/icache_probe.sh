#!/bin/bash
# icache_probe.sh: build icp_<n>.cubin for the sizes and the host program (run in kern/ on archy), then print the run line.
#   ICP_SIZES="64 128 ..." ./icache_probe.sh ; ICP_THREADS=32 ICP_CTAS=1 ./icache_probe 64:60000 128:30000 ...
set -e
cd "$(dirname "$0")"
export PATH=/opt/cuda/bin:$PATH
CUDA=${CUDA:-/opt/cuda}
gcc -std=gnu99 -O2 -Wall -o icache_probe icache_probe.c -I$CUDA/include -lcuda
${PTXAS11:-/opt/cuda-11.4/bin/ptxas} -arch=sm_52 -o skel_icp.cubin icache_probe.ptx   # the driver-matching ptxas, as the Makefile's skeletons
run=""
for n in ${ICP_SIZES:-64 128 256 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 65536}; do
	r=$((4000000 / n)); [ $r -lt 20 ] && r=20
	cp skel_icp.cubin icp_$n.cubin
	ICP_N=$n ICP_R=$r perl -I../maxas/lib ../maxas/bin/maxas.pl -i -k icache_probe icache_probe.sass icp_$n.cubin > /dev/null
	run="$run $n:$r"
done
echo "./icache_probe$run"
