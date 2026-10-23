#!/bin/bash
# bandmap.sh <die>: find the bad 128 MiB chunks above the verified bottom of a die. memtest reports the lowest failing chunk among the
# chunks it touches, so each round touches only the chunks whose state is unknown; a failure marks that chunk bad (and retires a page),
# a clean round proves the rest good. Stops at 64 retired pages.
die=${1:-2}; cd ~/claude/flash-sass-dev/kern
NCH=58
bad="43 44 52"; unknown="45 46 47 48 49 50 51 53 54 55 56 57"; good_until=42
mk_skip() {   # skip every chunk that is not in $1
	local keep=" $1 " s=""
	for ((c = 0; c < NCH; c++)); do [[ "$keep" == *" $c "* ]] || s+="$c,"; done
	echo "skip=${s%,}"
}
retired() { nvidia-smi --query-gpu=retired_pages.double_bit.count --format=csv,noheader -i $die; }
round=0
while [ -n "$unknown" ]; do
	round=$((round + 1))
	if [ "$(retired)" -ge 64 ]; then echo "bandmap: $(retired) pages retired, stopping with unknown: $unknown"; break; fi
	./memtest $die 1 $(mk_skip "$unknown") > /tmp/bm-run.log 2>&1; rc=$?
	echo "bandmap: round $round tested [$unknown] rc=$rc: $(tail -n1 /tmp/bm-run.log | cut -c1-120)"
	if [ $rc -eq 0 ]; then echo "bandmap: chunks [$unknown] are good"; unknown=""; break; fi
	[ $rc -eq 3 ] || { echo "bandmap: memtest failed without an ECC report"; break; }
	x=$(sed -n 's/.*while on chunk \([0-9]*\) =.*/\1/p' /tmp/bm-run.log)
	[ -n "$x" ] || { echo "bandmap: no chunk in the report"; break; }
	bad="$bad $x"; unknown=$(for c in $unknown; do [ "$c" = "$x" ] || echo -n "$c "; done); unknown=${unknown% }
	for ((w = 0; w < 600; w++)); do [ "$(nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader -i $die)" = "Yes" ] && break; done
	bash /tmp/reset-die.sh $die | tail -n1
done
echo "bandmap: bad chunks: $(echo $bad | tr ' ' '\n' | sort -n | tr '\n' ' ')"
echo "bandmap: unknown: ${unknown:-none}; retired pages: $(retired)"
