#!/bin/bash
# pcie-slot4-gen2.sh [root port]: slot 4 (root port 00:03.0, card 3, dies 4 and 5): target link speed gen2 (5 GT/s), then retrain the link. Run as root at boot by pcie-slot4-gen2.service.
# At gen3 this link makes about 2,800 receiver errors, 430 bad packets and 62 replay timeouts per token and decode swings between 11 and 32 tok/s; at gen2 it is clean (SPEC, kern/linkmatrix_test.sh).
P=${1:-00:03.0}
setpci -s "$P" CAP_EXP+0x30.w=0002 || { echo "pcie-slot4-gen2: cannot write the target speed of $P" >&2; exit 1; }
v=$(setpci -s "$P" CAP_EXP+0x10.w)
setpci -s "$P" CAP_EXP+0x10.w="$(printf '%04x' $((0x$v | 0x20)))"   # LnkCtl bit 5: retrain the link
for _ in $(seq 20); do
	s=$(lspci -vv -s "$P" | grep -o 'LnkSta:[[:space:]]*Speed [^,]*, Width [^,]*' | head -1)
	case "$s" in *"5GT/s"*) echo "pcie-slot4-gen2: $P $s"; exit 0;; esac
	sleep 0.5
done
echo "pcie-slot4-gen2: $P did not retrain to 5GT/s ($s)" >&2; exit 1
