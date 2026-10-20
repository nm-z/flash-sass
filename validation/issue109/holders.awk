# Input: ps -C hold -o pid=,args=. Output: one verified PID per logical die.
$1 ~ /^[0-9]+$/ && $3 == gguf && $5 == "0" && $6 == "48" {
    if (!match($7, /^\/tmp\/hold-h([0-5])\.manifest$/, die)) next
    logical = die[1] + 0
    if ($4 != logical) next
    split_found = 0
    for (field = 8; field <= NF; field++)
        if ($field == "split=" logical "/6") split_found++
    if (split_found != 1) next
    if (seen_die[logical]++ || seen_pid[$1]++) invalid = 1
    pid[logical] = $1
    count++
}
END {
    if (invalid || count != 6) {
        print "Expected six distinct full-model holders with matching device/manifest/split arguments" > "/dev/stderr"
        exit 1
    }
    for (logical = 0; logical < 6; logical++)
        printf "%s%s", logical ? " " : "", pid[logical]
    printf "\n"
}
