#!/bin/bash
# powercycle.sh: cold-cycle archy (power it off, wake it with a magic packet). Run it on engi, not on archy.
# A cold boot resets all eight GPUs and applies pending ECC mode changes; use it when `nvidia-smi -r` hangs in the driver.
# WoL on archy: Wake-on: g + BIOS token enabled (skill wake-device); the broadcast goes out engi's wired NIC.
# Stopping first: holders, flashcore, flashd (serve.sh stop), other users of the box (codex workers, the llama-server services on dies 6/7).
MAC=98:90:96:bb:be:75; HOST=archy; IP=192.168.1.218; BCAST=192.168.1.255

wol() {
	printf "%b" "$({ printf 'ffffffffffff'; for i in $(seq 16); do printf '%s' "${MAC//:/}"; done; } | sed 's/../\\x&/g')" \
		| socat -u - UDP4-DATAGRAM:$BCAST:9,broadcast
}

echo "powercycle: powering $HOST off"
ssh $HOST 'sync; sudo -n systemctl poweroff'
# systemd waits on processes stuck in the driver (D state) for up to 90 s per unit; if the box is still answering after 150 s, cut it
for i in $(seq 150); do ping -c1 -W1 $IP > /dev/null || break; done
if ping -c1 -W1 $IP > /dev/null; then
	echo "powercycle: still up after 150 s, SysRq power off"
	ssh $HOST 'echo s | sudo -n tee /proc/sysrq-trigger > /dev/null; echo u | sudo -n tee /proc/sysrq-trigger > /dev/null; echo o | sudo -n tee /proc/sysrq-trigger > /dev/null'
	for i in $(seq 60); do ping -c1 -W1 $IP > /dev/null || break; done
fi
ping -c1 -W1 $IP > /dev/null && { echo "powercycle: $HOST did not go down"; exit 1; }
echo "powercycle: $HOST is down, waking it"
# the packet is resent every 10 s until the host answers (the first one can arrive while the NIC is still powering down)
up=0
for i in $(seq 18); do
	wol
	if ping -c1 -W10 $IP > /dev/null; then up=1; break; fi
done
[ $up = 1 ] || { echo "powercycle: no answer from $HOST after 3 minutes (WoL not working? press the power button)"; exit 1; }
echo "powercycle: $HOST answers ping, waiting for ssh"
for i in $(seq 120); do ssh -o ConnectTimeout=2 -o BatchMode=yes $HOST true && break; done
ssh $HOST uptime
