#!/bin/bash
# queue_stub.sh: a stand-in for flashcore's command socket (run under socat, one process per connection) for queue_test.sh.
# `info` answers; `chat <n_new> ...` is logged to $STUB_LOG, then held for n_new/100 seconds (so max_tokens = 100*seconds + id + 1) and answered with one
# token and `done`. If flashd closes the connection while it is held (the abort), it logs `abort <id>` instead of `end <id>`.
while IFS= read -r line; do
	case $line in
	info) printf 'info 230000 4096\n' ;;
	chat\ *)
		set -- $line; n=$2; id=$((n % 100)); hold=$((n / 100))
		echo "chat $id" >> "$STUB_LOG"
		if read -r -t "$hold" _; then rc=0; else rc=$?; fi
		if [ "$rc" -gt 128 ]; then printf 'tok 100\ndone 1 1 0 1000.0 10.0 0 0\n'; echo "end $id" >> "$STUB_LOG"
		else echo "abort $id" >> "$STUB_LOG"; fi
		exit 0 ;;
	esac
done
