BEGIN {
	seen_send = 0
	inserted = 0
	pending_branch = 0
	failed = 0
}

/^xsend_SENT:[[:space:]]*$/ {
	if (seen_send != 0) {
		print "xchg_sys_fence: duplicate xsend_SENT label" > "/dev/stderr"
		failed = 1
		exit 2
	}
	seen_send++
	pending_branch = 1
	print
	next
}

pending_branch {
	if ($0 !~ /BRA xsend_END;/) {
		print "xchg_sys_fence: unexpected instruction after xsend_SENT" > "/dev/stderr"
		failed = 1
		exit 2
	}
	print "--:-:-:-:6      MEMBAR.SYS;"
	inserted++
	pending_branch = 0
	print
	next
}

{ print }

END {
	if (!failed && (seen_send != 1 || inserted != 1 || pending_branch)) {
		print "xchg_sys_fence: expected one xsend exit fence insertion" > "/dev/stderr"
		exit 2
	}
}
