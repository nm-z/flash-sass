#!/bin/bash
# proveset.sh <die> <block MiB> <known bad ranges> [passes] [max rounds]: the proof that a die works everywhere except its fence.
# Everything outside the fence is tested for <passes> passes (4 patterns each); a block that fails, whenever it does, is added to the fence
# on its own (innocent until it fails), the die is reset and the proof starts again. It ends with a clean run: that run is the proof,
# and the printed fence is the list to put into fences/die<N>.txt.
die=$1; mib=$2; bad=$3; passes=${4:-5}; max=${5:-100}
cd ~/claude/flash-sass-dev/kern
extra=""
for ((round = 1; round <= max; round++)); do
	skip="$bad${extra:+,$extra}"
	./memtest-new $die $passes chunk=$mib skip=$skip > /tmp/proveset-run.log 2>&1; rc=$?
	if [ $rc -eq 0 ]; then echo "proveset: round $round clean ($passes passes, 4 patterns, everything but: $skip)"; echo "FENCE $mib $skip"; echo DONE; exit 0; fi
	if [ $rc -ne 3 ]; then echo "proveset: round $round rc=$rc: $(tail -n1 /tmp/proveset-run.log | cut -c1-150)"; echo DONE; exit 1; fi
	x=$(sed -n 's/.*while on chunk \([0-9]*\) =.*/\1/p' /tmp/proveset-run.log)
	[ -n "$x" ] || { echo "proveset: no block in the report"; echo DONE; exit 1; }
	extra="${extra:+$extra,}$x"
	echo "proveset: round $round: block $x failed ($(awk -v x=$x -v m=$mib 'BEGIN{printf "%.4f", x*m/1024}') GiB), fenced on its own"
	bash /tmp/reset-die.sh $die > /dev/null
done
echo "proveset: still failing after $max rounds, fence so far: $skip"; echo DONE; exit 1
