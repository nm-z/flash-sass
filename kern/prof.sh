#!/bin/bash
# prof.sh [die ...] (PROF_ALL=1: every phase kind, not the twelve longest): where the last token of the last response spent its time on each die (default: all six), from the phase
# timestamps CTA 0 of the die writes at every phase end. The die's wait for its peers shows up in the phase that follows an
# exchange (actq256, hcwsum, ...), so compare dies: the one with the least waiting is the one the others wait for.
# Never send `quit` to the socket: it ends the core and with it the resident PLE table (a 20 minute reload).
SOCK=${FLASH_SOCK:-/tmp/flash.sock}
DIES=${*:-0 1 2 3 4 5}
for d in $DIES; do
	echo "== die $d"
	echo "prof $d${PROF_ALL:+ all}" | socat -t 5 - "UNIX-CONNECT:$SOCK"
done
