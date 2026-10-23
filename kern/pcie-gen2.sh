#!/bin/bash
# pcie-gen2.sh: target link speed gen2 (5 GT/s) and retrain for the given root ports (default 00:03.0, the slot-4 root port: the only link that makes errors at gen3). Lasts until the next reboot.
# Revert one port: setpci -s <port> CAP_EXP+0x30.w=0003, then set bit 5 of CAP_EXP+0x10.w.
for P in ${@:-00:03.0}; do
	before=$(sudo -n setpci -s $P CAP_EXP+0x30.w)
	sudo -n setpci -s $P CAP_EXP+0x30.w=0002 || { echo "$P: cannot write the target speed"; continue; }
	v=$(sudo -n setpci -s $P CAP_EXP+0x10.w)
	sudo -n setpci -s $P CAP_EXP+0x10.w=$(printf '%04x' $((0x$v | 0x20)))
	for i in 1 2 3 4 5 6 7 8; do
		s=$(sudo -n lspci -vv -s $P 2>&1 | rg -a -o 'LnkSta:\s+Speed [^,]*, Width [^,]*' | head -1 | tr -s ' \t' ' ')
		case "$s" in *"5GT/s"*) break;; esac
		timeout 1 tail -f /dev/null
	done
	echo "$P: target $before -> 0002, now $s"
done
