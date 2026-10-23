#!/bin/bash
# stop the engine if any of dies 0-5 reaches LIMIT C (default 92): archy died twice under inference load with die 4 at 86 C. Logs to ~/flash-watchdog.log
LIMIT=${1:-92}
log=/home/nate/flash-watchdog.log
echo "$(date +%T) watchdog start, limit ${LIMIT} C" >> $log
while true; do
	t=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v lim=$LIMIT '$1 < 6 && $2+0 >= lim {print "die " $1 " " $2 " C"}')
	if [ -n "$t" ]; then
		echo "$(date +%T) HOT: $t; stopping flashd and flashcore" >> $log
		pkill -x flashd; pkill -x flashcore
		sleep 30
	fi
	sleep 1
done
