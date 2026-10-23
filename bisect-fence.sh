#!/bin/bash
# bisect-fence.sh <die> <passes> <lo-hi> [<lo-hi> ...]: bisect.sh in the real configuration. The fence process (fences/die<N>.txt, kept in
# /home/nate/fences-die<N>.txt on archy) holds the known-bad blocks, a second process (memtest) takes whatever is free, and the guilty
# blocks are named by that second process's own allocation order. A failing run kills every context on the die (the fence too), so the
# die is reset and the fence rebuilt before the next probe. Prints "bad <ordinal>" per guilty ordinal.
die=$1; passes=$2; shift 2
cd ~/claude/flash-sass-dev/kern
bad=""; probes=0; fails=0
up=0
mkfence() {
	pkill -x fence; timeout 2 tail -f /dev/null
	bash /tmp/reset-die.sh $die > /dev/null
	rm -f /tmp/fence-d$die.ready
	read -r mib ranges < /home/nate/fences-die$die.txt
	setsid -f ./fence $die $mib $ranges ${FENCE_TAIL:-2} > /tmp/fence-d$die.err 2>&1 < /dev/null
	for w in $(seq 120); do [ -e /tmp/fence-d$die.ready ] && break; timeout 1 tail -f /dev/null; done
	[ -e /tmp/fence-d$die.ready ] || { echo "bisect-fence: fence did not come up: $(cat /tmp/fence-d$die.err)"; exit 1; }
	up=1
}
probe() {   # 0 clean, 1 failing
	local lo=$1 hi=$2
	[ $up -eq 1 ] || mkfence
	probes=$((probes + 1))
	./memtest-new $die $passes chunk=2 only=$lo-$hi > /tmp/bisect-fence-run.log 2>&1; local rc=$?
	if [ $rc -eq 0 ]; then return 0; fi
	if [ $rc -ne 3 ]; then echo "bisect-fence: rc=$rc on $lo-$hi: $(tail -n1 /tmp/bisect-fence-run.log | cut -c1-120)"; exit 1; fi
	fails=$((fails + 1)); up=0; return 1
}
split() {
	local lo=$1 hi=$2
	if [ $lo -eq $hi ]; then echo "bisect-fence: bad ordinal $lo"; bad+="$lo "; return; fi
	local mid=$(((lo + hi) / 2))
	probe $lo $mid || split $lo $mid
	probe $((mid + 1)) $hi || split $((mid + 1)) $hi
}
for r in "$@"; do
	lo=${r%-*}; hi=${r#*-}
	if probe $lo $hi; then echo "bisect-fence: $lo-$hi clean ($passes passes)"; else echo "bisect-fence: $lo-$hi fails, splitting"; split $lo $hi; fi
done
pkill -x fence
echo "bisect-fence: $probes probes, $fails failing; bad ordinals: ${bad:-none}"
echo "ORD $bad"
echo DONE
