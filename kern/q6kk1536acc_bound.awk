BEGIN {
    in_q6kk1536acc = 0
    loops = bounds = loads = adds = 0
}

$0 == "q6kk1536acc_LOOP:" {
    print
    in_q6kk1536acc = 1
    loops++
    print "--:-:-:-:1      ISETP.LT.U32.AND P2, PT, q6kk1536acc_row, q6kk1536acc_param_rows, P0;"
    next
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
    if (loops != 1 || bounds != 1 || loads != 1 || adds != 1) {
        print "q6kk1536acc bound transform failed: loops=" loops \
              " bounds=" bounds " loads=" loads " adds=" adds > "/dev/stderr"
        exit 1
    }
}
