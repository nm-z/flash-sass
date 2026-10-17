#!/bin/bash
# serve.sh <all|holders|core|vis|flashd|reload [lib]|status|stop|unload>: run the Flash-Next engine on archy.
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
# the int8 KV caches are in VRAM, FLASH_KV_HOST=1 puts them in pinned host memory). ECC is on for all dies (7.25 GiB allocatable each; memtest 2026-10-16:
# dies 0,1,3,4,5 clean, die 2 fails from 5.375 GiB and is fenced there, so it gets the small share). The shares were solved with the measured per-die overhead, see SPEC "Open work"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA numbers the dies the way nvidia-smi does, so a die index names the same card in the fence lookup, the holders and the tools
export SPLIT_SHARES=${SPLIT_SHARES:-112,99,79,112,99,99} ATTN_MAP=${ATTN_MAP:-1,1,1,1,4,4,4,4,5,5,5,5} ATTN_DIE=${ATTN_DIE:-1}
case $1 in
all)   # everything at once: the holders (35 min of repacking) and the core (14 min of PLE table) load side by side, the core waits for the holders
	$0 holders; $0 core; $0 vis
	for i in $(seq 7200); do [ -S "$SOCK" ] && break; read -t 1 < /dev/zero 2>/dev/null; done
	$0 flashd ;;
holders)   # DIES="3 4 5" ./serve.sh holders starts just those
	# every holder (and fence) keeps a shared flock on its die's reservation file (the convention of the embedding services in ~/codex/asm-port: they
	# take the same lock with `flock -x -n` and refuse to start while anyone holds it), so nothing else lands on a die the engine is loaded on.
	# A die with a fence file (fences/<gpu uuid>.txt, so the fence follows the physical card when cards change slots: "<block MiB> <bad block ranges>", found with memtest/blockmap) first gets a `fence` process that keeps its
	# bad blocks allocated, then a memtest of the free memory (FENCE_VERIFY=0 skips it) proves the list still matches the die, then the holder starts.
	LOCKDIR=${GPU_LOCKDIR:-/home/nate/codex/asm-port}
	# token_embd (273 MB) lives on one die only (EMBED_DIE, default 0): that die embeds the token and sends the stream to the others (prog_embed_bcast)
	for i in ${DIES:-0 1 2 3 4 5}; do
		uuid=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader); lock="$LOCKDIR/$uuid.lock"
		if [ -s "fences/$uuid.txt" ]; then
			read -r fmib franges < "fences/$uuid.txt"; rm -f /tmp/fence-d$i.ready
			setsid -f flock -s -n "$lock" ./fence $i $fmib $franges > /tmp/fence-d$i.err 2>&1 < /dev/null
			for w in $(seq 120); do [ -e /tmp/fence-d$i.ready ] && break; read -t 1 < /dev/zero 2>/dev/null; done
			[ -e /tmp/fence-d$i.ready ] || { echo "serve.sh: the fence of die $i did not come up: $(cat /tmp/fence-d$i.err)" >&2; exit 1; }
			if [ "${FENCE_VERIFY:-1}" = 1 ]; then
				./memtest $i 1 reserve=128 > /tmp/fence-d$i.memtest 2>&1 || { echo "serve.sh: memtest of the free memory of die $i failed (/tmp/fence-d$i.memtest): the fence list does not match the die" >&2; exit 1; }
			fi
		fi
		HOLD_VISIBLE=1 CUDA_VISIBLE_DEVICES=$i setsid -f flock -s -n "$lock" ./hold "$GGUF1" $i 0 48 /tmp/hold-h$i.manifest $([ $i = ${EMBED_DIE:-0} ] && echo token_embd) output split=$i/6 > /tmp/hold-h$i.out 2> /tmp/hold-h$i.err < /dev/null
	done ;;
core)
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 FLASH_MAXT=${FLASH_MAXT:-262144} FLASH_ENGINE_MAXT=${FLASH_ENGINE_MAXT:-150000} FLASH_KV_HOST=${FLASH_KV_HOST:-0} FLASH_MAXIMG=${FLASH_MAXIMG:-4096} FLASH_PLE_OFF=${FLASH_PLE_OFF:-192} FLASH_SOCK=$SOCK \
		setsid -f ./flashcore "$GGUF2" "$PWD/libengine.so" /tmp/hold-h{0,1,2,3,4,5}.manifest > /tmp/flashcore.out 2> /tmp/flashcore.err < /dev/null ;;
vis)
	setsid -f ./flashvis "${MMPROJ:-$HOME/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf}" "$GGUF1" "${FLASHVIS_SOCK:-/tmp/flashvis.sock}" "${VIS_THREADS:-10}" > /tmp/flashvis.out 2> /tmp/flashvis.err < /dev/null ;;
flashd)
	FLASHVIS_SOCK=${FLASHVIS_SOCK:-/tmp/flashvis.sock} FLASHD_NAME=${FLASHD_NAME:-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-sass} setsid -f ./flashd "$GGUF1" "$SOCK" 127.0.0.1:18090 > /tmp/flashd.out 2> /tmp/flashd.err < /dev/null ;;
reload)   # reload [<lib>] [FLASH_ENGINE_MAXT=n FLASH_KV_HOST=0|1 ...]
	shift; echo "reload $*" | socat -t 120 - "UNIX-CONNECT:$SOCK" ;;
status)
	pgrep -xa hold | cut -c1-60; pgrep -xa flashcore | cut -c1-60; pgrep -xa flashd | cut -c1-60
	tail -n 3 /tmp/flashcore.err /tmp/flashd.err ;;
stop)   # the engine; the holders (weights) and fences stay
	pkill -x flashd; pkill -x flashcore; pkill -x flashvis ;;
unload)   # everything, the fences included: the dies are free again (a fence must start on an empty die)
	$0 stop; pkill -x hold; pkill -x fence ;;
*)
	echo "usage: serve.sh all|holders|core|vis|flashd|reload [lib]|status|stop|unload" >&2; exit 2 ;;
esac
