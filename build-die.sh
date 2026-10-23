#!/bin/bash
# build-die.sh NAME [ENV=VAL ...]: assemble chain/kern/NAME.cubin from die.pl with the env settings (NOPFOLD=1: through nopfold.rb)
cd /home/nate/codex/flash-goal-140-1200/chain/kern || exit 1
n=$1; shift
if [ "${NOPFOLD:-0}" = 1 ]; then
	env ATTN_SHORT_TILE=1 ROUTER_LOOP=1 DIE_TRACE=0 "$@" perl die.pl | ruby nopfold.rb > $n.sass || exit 1
else
	env ATTN_SHORT_TILE=1 ROUTER_LOOP=1 DIE_TRACE=0 "$@" perl die.pl > $n.sass || exit 1
fi
rm -f $n.cubin; cp skel_die.cubin $n.cubin && chmod 644 $n.cubin && PATH=/opt/cuda/bin:$PATH perl -I../maxas/lib ../maxas/bin/maxas.pl -i $n.sass $n.cubin > $n.maxas.log 2>&1; tail -2 $n.maxas.log
if grep -qi "badly formed\|error\|unknown\|not a \|called at" $n.maxas.log; then echo "build-die: maxas failed for $n"; rm -f $n.cubin; exit 1; fi
chmod 444 $n.cubin; sha256sum $n.cubin
