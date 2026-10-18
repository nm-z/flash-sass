#!/bin/bash
set -euo pipefail
exec ruby /home/nate/codex/flash-pr30-run/kern/trace_validation.rb --config /home/nate/codex/flash-pr30-run/evidence/schema2/config-cta1.json --preflight "$1"
