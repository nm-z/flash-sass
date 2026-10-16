#!/bin/bash
# serve.sh <all|holders|core|vis|flashd|reload [lib]|status|stop>: run the Flash-Next engine on archy.
#   holders   the six weight holders (dies 0-5; ~40 min from NFS; restart only when the split changes)
#   vis       flashvis: the vision encoder (libmtmd on the CPU; the projector is loaded once, seconds per image)
#   core      flashcore: contexts, arenas and the 28.8 GB PLE table (~7 min), then the engine library
#   flashd    the HTTP front-end on 127.0.0.1:18090 (the chat page's gateway finds it through NMZ_CHAT_EXTRA)
#   reload    swap libengine.so (default: the same path, rebuilt) in the running core: seconds, nothing reloads
# Change host code: `make libengine.so && ./serve.sh reload`; kernel code: `make die.cubin` then
# `echo "cubin $PWD/die.cubin" | socat - UNIX-CONNECT:/tmp/flash.sock`. Neither touches the weights or the table.
cd "$(dirname "$0")" || exit 1
GGUF1=${GGUF1:-/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
GGUF2=${GGUF2:-/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf}
SOCK=${FLASH_SOCK:-/tmp/flash.sock}
# the split of the weights over the six dies and the die of each attention layer (its projections, indexer pool and RoPE tables live there;
# the int8 KV caches are in pinned host memory). Die 0 and die 2 each have a failing band of VRAM (die 2 above 5.657 GiB, die 0 above 5.76 GiB)
# and get the small shares; every die keeps >= 300 MB free (see SPEC, "Long context")
export SPLIT_SHARES=${SPLIT_SHARES:-82,110,77,106,112,113} ATTN_MAP=${ATTN_MAP:-1,1,1,1,4,4,4,4,5,5,5,5} ATTN_DIE=${ATTN_DIE:-1}
case $1 in
all)   # everything at once: the holders (35 min of repacking) and the core (14 min of PLE table) load side by side, the core waits for the holders
	$0 holders; $0 core; $0 vis
	for i in $(seq 7200); do [ -S "$SOCK" ] && break; read -t 1 < /dev/zero 2>/dev/null; done
	$0 flashd ;;
holders)
	for i in 0 1 2 3 4 5; do
		HOLD_VISIBLE=1 CUDA_VISIBLE_DEVICES=$i setsid -f ./hold "$GGUF1" $i 0 48 /tmp/hold-h$i.manifest token_embd output split=$i/6 > /tmp/hold-h$i.out 2> /tmp/hold-h$i.err < /dev/null
	done ;;
core)
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 FLASH_MAXT=${FLASH_MAXT:-262144} FLASH_MAXIMG=${FLASH_MAXIMG:-4096} FLASH_PLE_OFF=${FLASH_PLE_OFF:-192} FLASH_SOCK=$SOCK \
		setsid -f ./flashcore "$GGUF2" "$PWD/libengine.so" /tmp/hold-h{0,1,2,3,4,5}.manifest > /tmp/flashcore.out 2> /tmp/flashcore.err < /dev/null ;;
vis)
	setsid -f ./flashvis "${MMPROJ:-$HOME/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf}" "$GGUF1" "${FLASHVIS_SOCK:-/tmp/flashvis.sock}" "${VIS_THREADS:-10}" > /tmp/flashvis.out 2> /tmp/flashvis.err < /dev/null ;;
flashd)
	FLASHVIS_SOCK=${FLASHVIS_SOCK:-/tmp/flashvis.sock} FLASHD_NAME=${FLASHD_NAME:-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-sass} setsid -f ./flashd "$GGUF1" "$SOCK" 127.0.0.1:18090 > /tmp/flashd.out 2> /tmp/flashd.err < /dev/null ;;
reload)
	echo "reload ${2:-}" | socat -t 120 - "UNIX-CONNECT:$SOCK" ;;
status)
	pgrep -xa hold | cut -c1-60; pgrep -xa flashcore | cut -c1-60; pgrep -xa flashd | cut -c1-60
	tail -n 3 /tmp/flashcore.err /tmp/flashd.err ;;
stop)
	pkill -x flashd; pkill -x flashcore; pkill -x flashvis ;;
*)
	echo "usage: serve.sh all|holders|core|vis|flashd|reload [lib]|status|stop" >&2; exit 2 ;;
esac
