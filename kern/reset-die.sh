#!/bin/bash
# reset-die.sh <die>: the burn-in recipe, with timeouts so a hung driver cannot take the shell with it
d=$1
sudo -n rmmod nvidia_drm nvidia_modeset
timeout 120 sudo -n nvidia-smi -r -i $d; rc=$?
sudo -n modprobe nvidia_modeset; sudo -n modprobe nvidia_drm
sudo -n nvidia-smi -pm 1 > /dev/null
sudo -n nvidia-smi -ac 2505,1177 -i $d > /dev/null
echo "reset rc=$rc; $(nvidia-smi --query-gpu=retired_pages.double_bit.count,retired_pages.pending --format=csv,noheader -i $d)"
