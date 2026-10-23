#!/bin/bash
# blockmap.sh <die> <chunk MiB> <ranges of suspect blocks> [max rounds]: find the bad blocks inside memory a coarser map condemned.
# Every block is innocent until it fails on its own: memtest touches only the suspect blocks (the rest of the die stays allocated and
# untouched), the first failing block is named by the failing call, it is marked bad and the die is reset; the next round tests the
# suspects that are left. A clean round clears all of them. The retirement table of a die that is full (Xid 64) cannot repair anything,
# so the bad blocks stay bad: they are the fence.
# Output: "blockmap: bad <block>" lines and a final "blockmap: bad blocks: ..." and "blockmap: good suspects: ..." list.
die=$1; mib=$2; suspects=$3; max=${4:-400}
cd ~/claude/flash-sass-dev/kern
declare -A unk bad
expand() { local r a b; for r in ${1//,/ }; do a=${r%-*}; b=${r#*-}; for ((i = a; i <= b; i++)); do unk[$i]=1; done; done; }
tolist() {   # the keys of the set named $1 as ranges
	local -n s=$1; local prev=-2 start=-2 out="" k
	for k in $(printf '%s\n' "${!s[@]}" | sort -n); do
		if [ "$k" -eq $((prev + 1)) ]; then prev=$k; continue; fi
		[ $start -ge 0 ] && out+="$start-$prev,"
		start=$k; prev=$k
	done
	[ $start -ge 0 ] && out+="$start-$prev,"
	echo "${out%,}"
}
expand "$suspects"
round=0
while [ ${#unk[@]} -gt 0 ] && [ $round -lt $max ]; do
	round=$((round + 1))
	./memtest-new $die 1 chunk=$mib only=$(tolist unk) > /tmp/blockmap-run.log 2>&1; rc=$?
	if [ $rc -eq 0 ]; then echo "blockmap: round $round clean, ${#unk[@]} suspects cleared"; break; fi
	if [ $rc -ne 3 ]; then echo "blockmap: round $round rc=$rc: $(tail -n1 /tmp/blockmap-run.log | cut -c1-150)"; break; fi
	x=$(sed -n 's/.*while on chunk \([0-9]*\) =.*/\1/p' /tmp/blockmap-run.log)
	if [ -z "$x" ] || [ -z "${unk[$x]}" ]; then echo "blockmap: round $round: failure outside the suspects: $(tail -n1 /tmp/blockmap-run.log | cut -c1-150)"; break; fi
	bad[$x]=1; unset "unk[$x]"; echo "blockmap: bad $x ($(awk -v x=$x -v m=$mib 'BEGIN{printf "%.4f", x*m/1024}') GiB), ${#unk[@]} suspects left"
	bash /tmp/reset-die.sh $die | tail -n1 > /dev/null
done
echo "blockmap: bad blocks: $(tolist bad)"
echo "blockmap: good suspects: $(tolist unk)"
echo DONE
