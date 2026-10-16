#!/bin/bash
# burnin.sh <die> [rounds]: memtest the die; on an uncorrectable ECC error reset it (which
# activates the retirement of the failing page) and test again, until a pass is clean.
# The die must have no CUDA process on it (holders run with CUDA_VISIBLE_DEVICES=<die>).
die=$1; rounds=${2:-30}
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$die")
reset_die() {
	sudo rmmod nvidia_drm nvidia_modeset
	sudo nvidia-smi -r -i "$die"
	local rc=$?
	sudo modprobe nvidia_modeset
	sudo modprobe nvidia_drm
	sudo nvidia-smi -pm 1 > /dev/null
	return $rc
}
for ((round = 1; round <= rounds; round++)); do
	./memtest "$die" 1
	rc=$?
	echo "burnin: round $round memtest rc=$rc; retired $(nvidia-smi --query-gpu=retired_pages.dbe,retired_pages.sbe,retired_pages.pending --format=csv,noheader -i "$die")"
	if [ $rc -eq 0 ]; then echo "burnin: die $die ($bus) clean after $round round(s)"; exit 0; fi
	if [ $rc -ne 3 ]; then echo "burnin: memtest failed without an ECC report (rc=$rc)"; exit $rc; fi
	# the driver records the retirement (Xid 63) a few seconds after the DBE; a reset before that
	# loses it and the same cell fails again. Poll the pending flag (each query takes a fraction of
	# a second) until it is set, up to 600 queries.
	for ((w = 0; w < 600; w++)); do
		[ "$(nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader -i "$die")" = "Yes" ] && break
	done
	echo "burnin: pending after ${w} queries: $(nvidia-smi --query-gpu=retired_pages.dbe,retired_pages.pending --format=csv,noheader -i "$die")"
	reset_die || { echo "burnin: reset failed"; exit 1; }
done
echo "burnin: die $die still failing after $rounds rounds"
exit 3
