#!/bin/bash
# linkmatrix_test.sh [-n requests] [-r rounds] [config ...]
#
# Decode speed of the running engine (flashd on 127.0.0.1:18090) for each combination of PCIe link speed of the three cards' root ports. A config is three digits,
# the generation (1, 2, 3) of card 1, card 2, card 3 (slots 1, 2, 4; dies 0 1, 2 3, 4 5): 332 = card 1 gen3, card 2 gen3, card 3 gen2. With no config, all 27 run in a
# fixed shuffled order (so drift does not favour one end of the table). Each round sets the links, sends one warm-up request and then -n requests (default 3) of 40 tokens
# at temperature 0; -r rounds (default 1) are averaged. Prints one table: the generation of each card and the decode tok/s, nothing else. Needs sudo for setpci/lspci.
# The links are left at FINAL (default 332, the fastest steady setting measured: card 3 makes errors at gen3, see SPEC) and keep it until the next reboot.
#   env: PORTS="00:01.0 00:02.0 00:03.0" (root ports of cards 1 2 3), URL, FINAL
N=3; R=1
while getopts n:r: o; do case $o in n) N=$OPTARG;; r) R=$OPTARG;; *) exit 2;; esac; done; shift $((OPTIND - 1))
read -r P1 P2 P3 <<< "${PORTS:-00:01.0 00:02.0 00:03.0}"
URL=${URL:-http://127.0.0.1:18090}; FINAL=${FINAL:-332}
spd() { case $1 in 1) echo 2.5GT/s;; 2) echo 5GT/s;; 3) echo 8GT/s;; esac; }
setspeed() {   # root port, generation
	local P=$1 want cur v
	sudo -n setpci -s "$P" CAP_EXP+0x30.w=000$2
	v=$(sudo -n setpci -s "$P" CAP_EXP+0x10.w); sudo -n setpci -s "$P" CAP_EXP+0x10.w="$(printf '%04x' $((0x$v | 0x20)))"   # LnkCtl bit 5: retrain
	want=$(spd "$2")
	for _ in $(seq 10); do cur=$(sudo -n lspci -vv -s "$P" | grep -o 'LnkSta:[[:space:]]*Speed [^,]*' | head -1 | grep -o '[0-9.]*GT/s'); [ "$cur" = "$want" ] && return 0; timeout 1 tail -f /dev/null; done
	echo "linkmatrix_test: $P did not reach $want" >&2
}
setcfg() { setspeed "$P1" "${1:0:1}"; setspeed "$P2" "${1:1:1}"; setspeed "$P3" "${1:2:1}"; timeout 2 tail -f /dev/null; }
req() { curl -s -m 120 "$URL/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40,"temperature":0}' | jq -r '.timings.predicted_per_second'; }
if [ $# -gt 0 ]; then cfgs="$*"; else cfgs=$(for a in 1 2 3; do for b in 1 2 3; do for c in 1 2 3; do echo "$a$b$c"; done; done; done | shuf --random-source=<(yes 7)); fi
declare -A sum cnt
for _ in $(seq "$R"); do
	for c in $cfgs; do
		setcfg "$c"; req > /dev/null
		for _ in $(seq "$N"); do v=$(req); sum[$c]=$(awk -v s="${sum[$c]:-0}" -v v="$v" 'BEGIN { print s + v }'); cnt[$c]=$((${cnt[$c]:-0} + 1)); done
	done
done
setcfg "$FINAL"
echo "| card 1 | card 2 | card 3 | decode tok/s |"; echo "|---|---|---|---|"
for c in $(printf '%s\n' $cfgs | sort -u); do awk -v c="$c" -v s="${sum[$c]}" -v n="${cnt[$c]}" 'BEGIN { printf "| gen%s | gen%s | gen%s | %.1f |\n", substr(c,1,1), substr(c,2,1), substr(c,3,1), s / n }'; done
