BEGIN {
    in_q6kk1536acc = 0
    loops = bounds = loads = adds = 0
    timer_checks = 0
    timer_limit = DIE_KERNEL_LIMIT_UNITS
    if (timer_limit == "") timer_limit = "0x46"
    if (timer_limit !~ /^0x[[:xdigit:]]+$/ || strtonum(timer_limit) < 1 || strtonum(timer_limit) > 2147483647) {
        print "DIE_KERNEL_LIMIT_UNITS must be a positive 31-bit hexadecimal value" > "/dev/stderr"
        invalid_timer_limit = 1
    }
}

$0 == "q6kk1536acc_LOOP:" {
    print
    in_q6kk1536acc = 1
    loops++
    print "--:-:-:-:1      ISETP.LT.U32.AND P2, PT, q6kk1536acc_row, q6kk1536acc_param_rows, P0;"
    next
}

/ISETP\.GE\.U32\.AND P5, PT, [[:alpha:]_][[:alnum:]_]*, 0x46, (P[0-9]+|PT);/ {
    sub(/0x46,/, timer_limit ",")
    timer_checks++
}

in_q6kk1536acc && $0 == "q6kk1536acc_END:" {
    in_q6kk1536acc = 0
}

in_q6kk1536acc && /ISETP\.LT\.U32\.AND P2, PT, q6kk1536acc_row, q6kk1536acc_param_rows, P0;/ {
    bounds++
    next
}

in_q6kk1536acc && /LDG\.E q6kk1536acc_yold, \[q6kk1536acc_yp\];/ {
    sub(/LDG\.E/, "@P2 LDG.E")
    loads++
}

in_q6kk1536acc && /FADD q6kk1536acc_accf, q6kk1536acc_yold, q6kk1536acc_accf;/ {
    sub(/FADD/, "@P2 FADD")
    adds++
}

{ print }

END {
    if (invalid_timer_limit || loops != 1 || bounds != 1 || loads != 1 || adds != 1 || timer_checks != 5) {
        print "q6kk1536acc bound transform failed: loops=" loops \
              " bounds=" bounds " loads=" loads " adds=" adds \
              " timer_checks=" timer_checks > "/dev/stderr"
        exit 1
    }
}
