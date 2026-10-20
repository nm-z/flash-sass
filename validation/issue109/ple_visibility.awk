function fail(message) {
    print "ple_visibility.awk: " message > "/dev/stderr"
    failed = 1
}

{
    source[NR] = $0
    if ($0 == "plerows_BEGIN:") {
        begin_count++
        begin_line = NR
    }
    if ($0 == "plerows_END:") {
        end_count++
        end_line = NR
    }
    if (index($0, "PLE_VISIBILITY_PATCH_") != 0) patch_marker_count++
}

END {
    if (NR == 0) fail("empty generated SASS input")
    if (begin_count != 1) fail("expected exactly one plerows_BEGIN anchor")
    if (end_count != 1) fail("expected exactly one plerows_END anchor")
    if (begin_count == 1 && end_count == 1 && end_line <= begin_line)
        fail("plerows anchors are out of order")
    if (patch_marker_count != 0) fail("input already contains a PLE visibility patch marker")

    if (!failed) {
        for (i = begin_line; i <= end_line; i++) {
            line = source[i]
            if (line == "plerows_WAITRDY:") {
                wait_label_count++
                wait_label_line = i
            }
            if (index(line, "BRA plerows_WAITRDY;") != 0) {
                wait_branch_count++
                wait_branch_line = i
                if (line !~ /^[[:space:]]*--:-:-:-:6[[:space:]]+@P3 BRA plerows_WAITRDY;[[:space:]]*$/)
                    fail("wait branch does not match the expected plerows SASS anchor")
            }
            if (line ~ /MEMBAR\.SYS/) fail("unexpected MEMBAR.SYS already exists in plerows block")

            if (index(line, "LDG") != 0 && index(line, "plerows_hw") != 0) {
                if (!match(line,
                    /^[[:space:]]*--:-:3:-:[0-9]+[[:space:]]+LDG\.E\.U16[[:space:]]+plerows_hw([0-9][0-9]),[[:space:]]+\[plerows_tp\+0x([0-9a-f]+)\];[[:space:]]*$/,
                    capture)) {
                    fail("unexpected plerows payload load instruction")
                    continue
                }
                hw = capture[1] + 0
                offset = strtonum("0x" capture[2])
                if (hw != payload_count || offset != hw * 2)
                    fail("payload loads are missing, duplicated, reordered, or have an unexpected offset")
                if (payload_count == 0) first_payload_line = i
                payload_count++
                payload_line[i] = 1
                last_payload_line = i
                output[i] = line
                changed = sub(/LDG\.E\.U16/, "LDG.E.CV.U16", output[i])
                if (changed != 1) fail("validated payload load could not be transformed")
            }
        }
        if (wait_label_count != 1) fail("expected exactly one plerows_WAITRDY label")
        if (wait_branch_count != 1) fail("expected exactly one plerows_WAITRDY branch")
        if (payload_count != 45) fail("expected exactly 45 ordered U16 payload loads")
        if (wait_label_count == 1 && wait_branch_count == 1 && payload_count == 45 &&
            !(begin_line < wait_label_line && wait_label_line < wait_branch_line &&
              wait_branch_line < first_payload_line))
            fail("wait label, branch, and payload loads are not in the expected order")

        for (i = 1; i <= NR; i++) {
            if ((i < begin_line || i > end_line) &&
                index(source[i], "LDG") != 0 && index(source[i], "plerows_hw") != 0)
                fail("plerows payload load anchor appears outside the unique plerows block")
        }
    }

    if (failed) exit 2

    for (i = 1; i <= NR; i++) {
        if (i == wait_branch_line) {
            print source[i]
            print "# PLE_VISIBILITY_PATCH_BEGIN"
            print "--:-:-:-:6      MEMBAR.SYS;"
        } else {
            print (payload_line[i] ? output[i] : source[i])
            if (i == last_payload_line) print "# PLE_VISIBILITY_PATCH_END"
        }
    }
}
