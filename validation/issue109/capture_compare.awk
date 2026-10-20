# Compare the exported engine-capture TSV matrix against its mode 0/0 frame
# references. Invoke with the 64 TSV files as arguments, for example:
#   gawk -v expected_revision=20e0e581e6f10147778e88597d679c822e1da682 \
#        -v expected_cubin=0c4c2a72b076127e -f validation/issue109/capture_compare.awk \
#        validation/issue109/capture-enabled-37ed/engine-capture.*.tsv

BEGIN {
    FS = "\t"
    VERSION = "# issue109 stage fingerprints v1; additional XCOPY synchronization; not a production pass"
    COLUMNS = "position\tstage\tdie\tfnv64\tnonfinite"
    STAGES[1] = "embed.S"
    STAGES[2] = "b0.S"
    STAGES[3] = "ple.kn"
    STAGES[4] = "ple.qn"
    STAGES[5] = "ple.pG"
    STAGES[6] = "ple.px"
    STAGES[7] = "ple.S"
    STAGES[8] = "b1.S"
    for (i = 1; i <= 8; i++) STAGE_ID[STAGES[i]] = i
    if (ARGC < 2) {
        print "usage: gawk -v expected_revision=<40-hex> -v expected_cubin=<16-hex> -f capture_compare.awk engine-capture.*.tsv" > "/dev/stderr"
        usage_error = 1
        exit 2
    }
    if (length(expected_revision) != 40 || expected_revision !~ /^[0-9a-f]+$/ ||
        length(expected_cubin) != 16 || expected_cubin !~ /^[0-9a-f]+$/) {
        print "capture_compare: valid -v expected_revision and -v expected_cubin are required" > "/dev/stderr"
        usage_error = 1
        exit 2
    }
}

function reject(message) {
    invalid++
    printf "capture_compare: %s: %s\n", current_file == "" ? "input" : current_file, message > "/dev/stderr"
}

function expected_resume(request) {
    if (request == 5) return 1
    if (request == 8) return 2
    if (request == 11) return 3
    if (request == 14) return 5
    return 0
}

function reference_request(request) {
    if (request == 17) return 17
    if (request == 2 || request == 4 || request == 5 || request == 7 ||
        request == 8 || request == 10 || request == 11 || request == 13 ||
        request == 14 || request == 16) return 2
    if (request == 3 || request == 6 || request == 9 || request == 12 ||
        request == 15) return 3
    return 0
}

function data_key(file_key, position, stage, die) {
    return file_key SUBSEP position SUBSEP stage SUBSEP die
}

FNR == 1 {
    current_file = FILENAME
    current_key = ""
    current_name_ok = 1
    file_count++
    if (input_path[FILENAME]++) reject("same input file supplied more than once")
    if ($0 != VERSION) reject("unrecognized or missing TSV version line")

    basename = FILENAME
    sub(/^.*\//, "", basename)
    if (!match(basename, /^engine-capture\.([0-9]+)\.([0-9]+)\.tsv$/, name_match)) {
        reject("filename must be engine-capture.<pid>.<request>.tsv")
        current_name_ok = 0
    } else {
        filename_pid = name_match[1] + 0
        filename_request = name_match[2] + 0
        if (filename_pid < 1 || filename_request < 2 || filename_request > 17)
            reject("filename PID or request is outside the expected matrix")
    }
    next
}

FNR == 2 {
    if (!current_name_ok) { reject("metadata cannot be bound to an invalid filename"); next }
    if (!match($0, /^# pid=([0-9]+) request=([0-9]+) revision=([0-9a-f]+) cubin_fnv=([0-9a-f]+) kv_host=([01]) paired=([01]) resumed=([0-9]+) total=([0-9]+)$/, meta)) {
        reject("metadata line does not match the v1 field grammar")
        next
    }

    pid = meta[1] + 0
    request = meta[2] + 0
    revision = meta[3]
    cubin = meta[4]
    kv = meta[5] + 0
    paired = meta[6] + 0
    resume = meta[7] + 0
    total = meta[8] + 0
    if (pid != filename_pid || request != filename_request)
        reject("filename PID/request does not match metadata")
    if (length(meta[3]) != 40 || length(meta[4]) != 16)
        reject("revision or cubin fingerprint has the wrong length")
    if (revision != expected_revision || cubin != expected_cubin)
        reject("revision/cubin identity does not match the required build")
    if (request < 2 || request > 17)
        reject("request is outside 2..17")

    wanted_total = request == 17 ? 10 : 9
    wanted_resume = expected_resume(request)
    if (total != wanted_total || resume != wanted_resume)
        reject(sprintf("request %d requires total=%d resumed=%d", request, wanted_total, wanted_resume))
    if (resume < 0 || resume >= total)
        reject("resume/total range is invalid")

    if (!identity_seen) {
        common_revision = revision
        common_cubin = cubin
        identity_seen = 1
    } else if (revision != common_revision || cubin != common_cubin) {
        reject("revision/cubin identity differs across capture files")
    }

    current_key = request SUBSEP kv SUBSEP paired
    mode_key = kv SUBSEP paired
    if ((mode_key in mode_pid) && mode_pid[mode_key] != pid)
        reject("one mode uses more than one PID")
    else
        mode_pid[mode_key] = pid
    if ((pid in pid_mode) && pid_mode[pid] != mode_key)
        reject("one PID is shared across modes")
    else
        pid_mode[pid] = mode_key
    if (current_key in source_file) {
        reject("duplicate request/mode file; previous file is " source_file[current_key])
        current_key = ""
        next
    }
    source_file[current_key] = FILENAME
    file_pid[current_key] = pid
    file_request[current_key] = request
    file_kv[current_key] = kv
    file_paired[current_key] = paired
    file_resume[current_key] = resume
    file_total[current_key] = total
    next
}

FNR == 3 {
    if ($0 != COLUMNS) reject("column header does not match the v1 schema")
    next
}

FNR > 3 {
    if (current_key == "") { reject("data row has no valid, unique metadata identity"); next }
    if (NF != 5) { reject("data row must contain exactly five tab-separated fields"); next }
    if ($1 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || $5 != "0" ||
        length($4) != 16 || $4 !~ /^[0-9a-f]+$/) {
        reject("data row has invalid position, die, FNV64, or nonfinite value")
        next
    }
    if (!($2 in STAGE_ID)) { reject("unknown stage label: " $2); next }

    position = $1 + 0
    die = $3 + 0
    stage = STAGE_ID[$2]
    if (position < file_resume[current_key] || position >= file_total[current_key]) {
        reject("position lies outside the captured resume/total interval")
        next
    }
    if (die < 0 || die > 5) { reject("logical die must be in 0..5"); next }

    key = data_key(current_key, position, stage, die)
    if (key in hash) { reject("duplicate position/stage/die row"); next }
    hash[key] = "h" $4
    rows[current_key]++
    row_count++
    next
}

END {
    if (usage_error) exit 2
    current_file = "matrix"
    expected_files = 16 * 4
    if (ARGC - 1 != expected_files) reject("exactly 64 TSV input paths are required")
    if (file_count != expected_files) reject("one or more input files are empty or unreadable")

    expected_row_count = 0
    for (request = 2; request <= 17; request++) {
        total = request == 17 ? 10 : 9
        resume = expected_resume(request)
        per_file_rows = 6 * 8 * (total - resume)
        expected_row_count += 4 * per_file_rows
        for (kv = 0; kv <= 1; kv++) {
            for (paired = 0; paired <= 1; paired++) {
                current_key = request SUBSEP kv SUBSEP paired
                if (!(current_key in source_file)) {
                    reject(sprintf("missing request=%d kv_host=%d paired=%d file", request, kv, paired))
                    continue
                }
                if (file_resume[current_key] != resume || file_total[current_key] != total)
                    reject(sprintf("request=%d mode=%d/%d has incorrect resume/total metadata", request, kv, paired))
                if (rows[current_key] != per_file_rows)
                    reject(sprintf("request=%d mode=%d/%d rows=%d expected=%d", request, kv, paired, rows[current_key], per_file_rows))
            }
        }
    }
    if (row_count != expected_row_count)
        reject(sprintf("matrix has %d valid rows; expected %d", row_count, expected_row_count))
    expected_reference_rows = 6 * 8 * 9 + 6 * 8 * 9 + 6 * 8 * 10
    expected_comparisons = expected_row_count - expected_reference_rows
    for (mode_key in mode_pid) distinct_mode_count++
    if (distinct_mode_count != 4) reject("matrix must use exactly four distinct mode PIDs")

    for (request = 2; request <= 17; request++) {
        ref_request = reference_request(request)
        if (!ref_request) { reject("no reference mapping for request " request); continue }
        ref_key = ref_request SUBSEP 0 SUBSEP 0
        total = request == 17 ? 10 : 9
        resume = expected_resume(request)
        for (kv = 0; kv <= 1; kv++) {
            for (paired = 0; paired <= 1; paired++) {
                current_key = request SUBSEP kv SUBSEP paired
                if (!(current_key in source_file) || !(ref_key in source_file)) continue
                if (current_key == ref_key) {
                    reference_rows += 6 * 8 * (total - resume)
                    continue
                }
                for (position = resume; position < total; position++) {
                    for (stage = 1; stage <= 8; stage++) {
                        for (die = 0; die <= 5; die++) {
                            current_data = data_key(current_key, position, stage, die)
                            ref_data = data_key(ref_key, position, stage, die)
                            if (!(current_data in hash) || !(ref_data in hash)) continue
                            hashes_compared++
                            if (hash[current_data] != hash[ref_data]) {
                                hash_mismatches++
                                if (!first_diff) {
                                    first_diff = 1
                                    first_request = request
                                    first_kv = kv
                                    first_paired = paired
                                    first_reference = ref_request
                                    first_position = position
                                    first_stage = STAGES[stage]
                                    first_die = die
                                    first_got = hash[current_data]
                                    first_want = hash[ref_data]
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    printf "capture_compare: files=%d/%d total_rows=%d/%d reference_rows=%d compared=%d/%d hash_mismatches=%d invalid=%d\n",
           file_count, expected_files, row_count, expected_row_count,
           reference_rows, hashes_compared, expected_comparisons, hash_mismatches, invalid
    if (first_diff)
        printf "first sampled difference: request=%d mode=%d/%d reference_request=%d position=%d stage=%s die=%d got=%s expected=%s\n",
               first_request, first_kv, first_paired, first_reference,
               first_position, first_stage, first_die, substr(first_got, 2), substr(first_want, 2)
    else if (hashes_compared)
        print "first sampled difference: none"

    if (file_count != expected_files || row_count != expected_row_count ||
        reference_rows != expected_reference_rows || hashes_compared != expected_comparisons ||
        hash_mismatches || invalid)
        exit 1
    print "capture_compare: PASS"
}
