BEGIN {
	split("hcmixx dnnormx bf16hcup bf16hcx bf16hcrx bf16hcrwx xsend bf16hcxp", names, " ")
	for (i = 1; i <= length(names); i++) required[names[i]] = 1
}

/^[[:alnum:]_]+_SETUP:[[:space:]]*$/ {
	name = $1
	sub(/_SETUP:$/, "", name)
	current = name in required ? name : ""
	if (current != "") setup_count[current]++
}

{
	line = $0
	if (current != "" && line ~ /STG\.E\.WT/) store_count[current]++

	# The setup has a predicated early exit for inactive threads. Fence once on
	# the unpredicated final branch, after that thread has finished all peer stores.
	if (current != "" && line ~ /BRA POSITION_END;/ && line !~ /@/) {
		branch_count[current]++
		if (!store_count[current]) {
			print "xchg_peer_fence: peer store not found in " current > "/dev/stderr"
			failed = 1
			exit 2
		}
		print "--:-:-:-:6      MEMBAR.SYS;"
		insert_count[current]++
		print line
		current = ""
		next
	}

	print line
}

END {
	for (name in required) {
		if (!failed && (setup_count[name] != 1 || !store_count[name] || branch_count[name] != 1 || insert_count[name] != 1)) {
			print "xchg_peer_fence: expected one peer-writing final branch in " name > "/dev/stderr"
			exit 2
		}
	}
}
