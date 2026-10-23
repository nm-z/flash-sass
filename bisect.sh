#!/bin/bash
# bisect.sh <die> <block MiB> <passes> <lo-hi> [<lo-hi> ...]: find the bad blocks inside the given ranges of a die by binary search.
# A run that fails (uncorrectable ECC error) says that at least one block it touched is bad, but NOT which one: the error reaches the
# process some time after the bad read, so the block a failing call was "on" is only where it surfaced. A run that is clean proves its
# blocks good (for that many passes). So a range is tested as a whole; clean: all innocent; failing: split in halves and test each;
# a failing single block is guilty. Only the blocks of the ranges are touched, the rest of the die stays allocated and untouched.
# Prints "bad <block>" per guilty block and "BAD ranges" at the end; the die is reset after every failure.
die=$1; mib=$2; passes=$3; shift 3
cd ~/claude/flash-sass-dev/kern
bad=""
probes=0; fails=0
probe() {   # probe lo hi: 0 clean, 1 failing
	local lo=$1 hi=$2
	probes=$((probes + 1))
	./memtest-new $die $passes chunk=$mib only=$lo-$hi > /tmp/bisect-run.log 2>&1; local rc=$?
	if [ $rc -eq 0 ]; then return 0; fi
	if [ $rc -ne 3 ]; then echo "bisect: rc=$rc on $lo-$hi: $(tail -n1 /tmp/bisect-run.log | cut -c1-120)"; exit 1; fi
	fails=$((fails + 1)); bash /tmp/reset-die.sh $die > /dev/null; return 1
}
split() {   # split lo hi: the range is known to fail
	local lo=$1 hi=$2
	if [ $lo -eq $hi ]; then echo "bisect: bad $lo ($(awk -v x=$lo -v m=$mib 'BEGIN{printf "%.4f", x*m/1024}') GiB)"; bad+="$lo "; return; fi
	local mid=$(((lo + hi) / 2))
	probe $lo $mid || split $lo $mid
	probe $((mid + 1)) $hi || split $((mid + 1)) $hi
}
for r in "$@"; do
	lo=${r%-*}; hi=${r#*-}
	if probe $lo $hi; then echo "bisect: $lo-$hi clean ($passes passes)"; else echo "bisect: $lo-$hi fails, splitting"; split $lo $hi; fi
done
echo "bisect: $probes probes, $fails failing; bad blocks: ${bad:-none}"
echo "BAD $mib $bad"
echo DONE
