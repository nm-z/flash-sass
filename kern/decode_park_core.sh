#!/usr/bin/env bash
# Temporarily park an idle resident engine while preserving its process and holders.
# Requires a concrete configuration; never discovers or kills another worker's job.
set -euo pipefail
if [[ $# != 2 && $# != 3 ]]; then echo 'usage: decode_park_core.sh CONFIG.json NEW_OUTPUT_DIRECTORY [--startup-only]' >&2; exit 2; fi
extra=()
if [[ $# == 3 ]]; then [[ $3 == --startup-only ]] || exit 2; extra=(--startup-only); fi
config=$(realpath "$1")
out=$(realpath -m "$2")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ ! -e "$out" ]] || { echo "output exists: $out" >&2; exit 2; }
mkdir -p "$out"
cp "$config" "$out/config.json"
lock=$(jq -er '.lock_path' "$config")
live_pid=$(jq -er '.parking.pid' "$config")
live_socket=$(jq -er '.parking.socket' "$config")
live_library=$(jq -er '.parking.library' "$config")
null_library=$(jq -er '.parking.null_library' "$config")
clock_gpus=$(jq -er '.parking.clock_gpus | map(tostring) | join(",")' "$config")
sm=$(jq -er '.parking.sm_mhz' "$config")
mem=$(jq -er '.parking.memory_mhz' "$config")
rpc() {
    perl -MIO::Socket::UNIX -MSocket -e '
      alarm 30; my($path,$command)=@ARGV;
      my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$path,Timeout=>10) or die "connect $path: $!\n";
      print {$s} "$command\n" or die "send: $!\n";
      my $line=<$s>; die "empty response\n" unless defined $line;
      print $line; die "core rejected command\n" if $line =~ /^error/;
      close $s or die $!;
    ' "$live_socket" "$1"
}
exec 9>"$lock"
flock -x 9
# Both occupancy and process state must be idle. Validate each explicitly allowed
# passive process, including its identity, before touching the resident engine.
idle_samples=0
while :; do
    nvidia-smi > "$out/idle-nvidia-smi.txt"
    ps -eo pid,ppid,state,wchan:32,pcpu,etime,args > "$out/idle-ps.txt"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$out/idle-compute-pids.txt"
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits > "$out/idle-utilization.csv"
    if ! awk -F, 'NF==2 && $2+0 != 0 {bad=1} END {exit bad}' "$out/idle-utilization.csv"; then idle_samples=0; sleep 2; continue; fi
    allowed=$(jq -r '.passive_pids[].pid' "$config" | paste -sd, -)
    if ! awk -v allowed="$allowed" 'BEGIN {n=split(allowed,a,",");for(i=1;i<=n;i++)ok[a[i]]=1} NF && !ok[$1] {bad=1} END {exit bad}' "$out/idle-compute-pids.txt"; then idle_samples=0; sleep 2; continue; fi
    if ! perl -MJSON::PP -e '
      local $/; open my $f,"<",$ARGV[0] or die $!; my $c=decode_json(<$f>);
      for my $p (@{$c->{passive_pids}}) {
        my $pid=$p->{pid}; open my $s,"<","/proc/$pid/stat" or exit 1; my $v=<$s>;
        $v =~ s/^\d+ \(.*\) //; my @v=split /\s+/,$v;
        exit 1 unless $v[0] eq "S" && $v[19]==$p->{start_ticks};
        exit 1 unless readlink("/proc/$pid/exe") eq $p->{exe};
        open my $w,"<","/proc/$pid/wchan" or exit 1; my $wait=<$w>;
        exit 1 unless $wait =~ /^(?:_*do_sys_pause|_*skb_wait_for_more_packets)\s*$/;
      }
    ' "$config"; then idle_samples=0; sleep 2; continue; fi
    idle_samples=$((idle_samples + 1))
    (( idle_samples >= 3 )) && break
    sleep 2
done
nvidia-smi -q -x > "$out/gpu-before.xml"
nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.applications.memory --format=csv,noheader,nounits > "$out/application-clocks-before.csv"
[[ $(readlink "/proc/$live_pid/exe") == $(jq -r '.parking.executable' "$config") ]] || { echo 'resident executable changed' >&2; exit 1; }
loaded=$(awk -v pid="$live_pid" '$NF ~ ("/tmp/flashcore-" pid "-gen[0-9]+\\.so$") {print $NF;exit}' "/proc/$live_pid/maps")
[[ -f "$loaded" ]] || { echo 'cannot identify resident loaded library' >&2; exit 1; }
old_hash=$(sha256sum "$loaded" | cut -d' ' -f1)
[[ $(sha256sum "$live_library" | cut -d' ' -f1) == "$old_hash" ]] || { echo 'loaded and configured library differ; preserve them and stop' >&2; exit 1; }
cp "$loaded" "$out/resident-library-backup.so"
printf '%s\n' "$old_hash" > "$out/resident-library-sha256.txt"
rpc config > "$out/resident-config-before.txt"
parked=0
clocks_changed=0
cleanup() {
    local code=$? restoration=0 gpu old_sm old_mem restore_library
    trap - EXIT INT TERM
    if (( parked )); then
        restore_library="$live_library"
        if [[ $(sha256sum "$live_library" | cut -d' ' -f1) != "$old_hash" ]]; then restore_library="$out/resident-library-backup.so"; fi
        printf '%s\n' "$restore_library" > "$out/restored-library-path.txt"
        rpc "reload $restore_library" > "$out/restore-response.txt" 2>&1 || restoration=1
        rpc config > "$out/resident-config-after.txt" 2>&1 || restoration=1
        cmp -s "$out/resident-config-before.txt" "$out/resident-config-after.txt" || restoration=1
    fi
    if (( clocks_changed )); then
        while IFS=, read -r gpu old_sm old_mem; do
            gpu=${gpu// /}; old_sm=${old_sm// /}; old_mem=${old_mem// /}
            [[ ,$clock_gpus, == *,$gpu,* ]] || continue
            sudo -n nvidia-smi -i "$gpu" -ac "$old_mem,$old_sm" >> "$out/restore-clocks.txt" 2>&1 || restoration=1
        done < "$out/application-clocks-before.csv"
    fi
    nvidia-smi -q -x > "$out/gpu-after.xml" || restoration=1
    jq -n --argjson exit_code "$code" --argjson restoration_exit "$restoration" '{exit:$exit_code,restoration_exit:$restoration_exit}' > "$out/exit.json"
    if (( restoration )); then echo "restoration failed; inspect $out" >&2; exit 1; fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# The restoration trap is armed before either reversible state change.
parked=1
rpc "reload $null_library" > "$out/park-response.txt"
clocks_changed=1
sudo -n nvidia-smi -i "$clock_gpus" -ac "$mem,$sm" > "$out/set-clocks.txt" 2>&1
perl "$script_dir/decode_pair_setup.pl" --config "$config" --out "$out/capture" --lock-fd 9 "${extra[@]}"
