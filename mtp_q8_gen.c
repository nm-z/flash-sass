/* Standalone SASS Q8 GEMV for the MTP drafter. Four FP32 accumulation chains
 * per lane, four coalesced 128-element loads in flight, no shared memory.
 * Uses the existing repack_b32(Q8_0) storage format, preserving weight values. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void out(const char *s) { fputs(s,stdout); }
static void multiply(const char *dst,const char *src,unsigned n) {
    int first=1;
    for(int b=0;b<31;b++) if(n&(1u<<b)) {
        if(first) printf("--:-:-:-:6 SHL %s, %s, 0x%x;\n",dst,src,b);
        else printf("--:-:-:-:6 ISCADD %s, %s, %s, 0x%x;\n",dst,src,dst,b);
        first=0;
    }
}
int main(int argc,char **argv) {
    int phase=argc==3 && !strcmp(argv[2],"--phase");
    int k=(argc==2||phase)?atoi(argv[1]):0;
    if(k!=320&&k!=640&&k!=2560&&k!=5120&&k!=6144&&k!=10240) return 2;
    unsigned pad=(unsigned)((k/16+15)&~15),stride=pad+(unsigned)k;
    printf("# Kernel: mtp_q8\n# K=%d, packed row stride=%u.\n",k,stride);
    out("<CONSTANT_MAPPING>\n"
        "param_W0 : c[0x0][0x140]\nparam_W1 : c[0x0][0x144]\n"
        "param_X0 : c[0x0][0x148]\nparam_X1 : c[0x0][0x14c]\n"
        "param_Y0 : c[0x0][0x150]\nparam_Y1 : c[0x0][0x154]\n"
        "param_rows : c[0x0][0x158]\nparam_xstride : c[0x0][0x15c]\n"
        "param_idx0 : c[0x0][0x160]\nparam_idx1 : c[0x0][0x164]\n");
    if(phase) out("param_expert : c[0x0][0x168]\nparam_count0 : c[0x0][0x170]\nparam_count1 : c[0x0][0x174]\n");
    out("</CONSTANT_MAPPING>\n<REGISTER_MAPPING>\n"
        "0 : tid\n1 : warp\n2 : lane\n3 : row\n4 : base\n5 : ix\n6 : bx\n7 : tmp\n"
        "8,9 : wp<0-1>\n10,11 : qp<0-1>\n12,13 : sp<0-1>\n"
        "14,15 : xp<0-1>\n16,17 : yp<0-1>\n18 : off\n19 : ey\n20 : erow\n21 : rows\n"
        "24-39 : x<00-15>\n40-43 : q<0-3>\n44-47 : d<0-3>\n"
        "49 : a0\n50 : a1\n51 : a2\n48 : a3\n"
        "54 : f0\n55 : f1\n52 : f2\n53 : f3\n56 : sh\n"
        "</REGISTER_MAPPING>\n"
        "--:-:6:-:1 S2R tid, SR_TID.X;\n--:-:6:-:1 S2R bx, SR_CTAID.X;\n");
    if(phase) {
        out("--:-:-:-:6 MOV ey, param_expert;\n"
            "--:-:-:-:6 MOV sp0, param_count0;\n--:-:-:-:6 MOV sp1, param_count1;\n"
            "--:-:-:-:6 LOP.OR tmp, sp0, sp1;\n"
            "--:-:-:-:6 ISETP.NE.U32.AND P0, PT, tmp, RZ, PT;\n"
            "--:-:-:-:6 MOV32I tmp, 0xffffffff;\n"
            "--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n"
            "--:-:1:-:2 @P0 LDG.E tmp, [sp];\n"
            "01:-:-:-:6 ISETP.GE.U32.AND P5, PT, ey, tmp, PT;\n"
            "--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n"
            "--:-:-:-:6 @P5 EXIT;\n");
    } else out("--:-:6:-:2 S2R ey, SR_CTAID.Y;\n");
    out("20:-:-:-:6 SHR.U32 warp, tid, 0x5;\n"
        "--:-:-:-:6 LOP32I.AND lane, tid, 0x1f;\n"
        "--:-:-:-:6 ISCADD row, bx, warp, 0x3;\n"
        "--:-:-:-:6 MOV rows, param_rows;\n"
        "--:-:-:-:6 ISETP.GE.U32.AND P5, PT, row, rows, PT;\n"
        "--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n"
        "--:-:-:-:6 @P5 EXIT;\n"
        "--:-:-:-:6 MOV sp0, param_idx0;\n--:-:-:-:6 MOV sp1, param_idx1;\n"
        "--:-:-:-:6 LOP.OR tmp, sp0, sp1;\n"
        "--:-:-:-:6 ISETP.NE.U32.AND P0, PT, tmp, RZ, PT;\n"
        "--:-:-:-:6 SHL tmp, ey, 0x2;\n"
        "--:-:-:-:6 IADD sp0.CC, sp0, tmp;\n--:-:-:-:6 IADD.X sp1, sp1, RZ;\n"
        "--:-:-:-:6 MOV erow, RZ;\n--:-:1:-:2 @P0 LDG.E erow, [sp];\n"
        "01:-:-:-:6 XMAD tmp, erow, rows, row;\n"
        "--:-:-:-:6 XMAD.PSL erow, erow, rows.H1, tmp;\n");
    multiply("off","erow",stride);
    out("--:-:-:-:6 IADD wp0.CC, off, param_W0;\n--:-:-:-:6 IADD.X wp1, RZ, param_W1;\n"
        "--:-:-:-:6 SHR.U32 tmp, lane, 0x3;\n--:-:-:-:6 SHL tmp, tmp, 0x1;\n"
        "--:-:-:-:6 IADD sp0.CC, wp0, tmp;\n--:-:-:-:6 IADD.X sp1, wp1, RZ;\n"
        "--:-:-:-:6 SHL ix, lane, 0x2;\n");
    printf("--:-:-:-:6 IADD tmp, ix, 0x%x;\n",pad);
    out("--:-:-:-:6 IADD qp0.CC, wp0, tmp;\n--:-:-:-:6 IADD.X qp1, wp1, RZ;\n"
        "--:-:-:-:6 MOV tmp, param_xstride;\n"
        "--:-:-:-:6 XMAD off, ey, tmp, RZ;\n"
        "--:-:-:-:6 ISCADD off, lane, off, 0x4;\n"
        "--:-:-:-:6 IADD xp0.CC, off, param_X0;\n--:-:-:-:6 IADD.X xp1, RZ, param_X1;\n"
        "--:-:-:-:6 XMAD tmp, ey, rows, row;\n"
        "--:-:-:-:6 XMAD.PSL tmp, ey, rows.H1, tmp;\n"
        "--:-:-:-:6 SHL off, tmp, 0x2;\n"
        "--:-:-:-:6 IADD yp0.CC, off, param_Y0;\n--:-:-:-:6 IADD.X yp1, RZ, param_Y1;\n"
        "--:-:-:-:6 ISETP.EQ.U32.AND P6, PT, lane, RZ, PT;\n"
        "--:-:-:-:6 MOV base, RZ;\n"
        "--:-:-:-:1 MOV a0, RZ;\n--:-:-:-:1 MOV a1, RZ;\n"
        "--:-:-:-:1 MOV a2, RZ;\n--:-:-:-:6 MOV a3, RZ;\nLOOP:\n");
    for(int j=0;j<4;j++) {
        printf("--:-:-:-:6 IADD tmp, ix, 0x%x;\n",128*j);
        printf("--:-:-:-:6 ISETP.LT.U32.AND P%d, PT, tmp, 0x%x, PT;\n",j,k);
    }
    for(int j=0;j<4;j++) {
        for(int t=0;t<4;t++) printf("--:-:-:-:1 MOV x%02d, RZ;\n",4*j+t);
        printf("--:-:-:-:1 MOV q%d, RZ;\n--:-:-:-:1 MOV d%d, RZ;\n",j,j);
    }
    for(int j=0;j<4;j++) {
        printf("--:2:1:-:1 @P%d LDG.E.CI q%d, [qp+0x%x];\n",j,j,128*j);
        printf("--:2:1:-:1 @P%d LDG.E.CI.128 x%02d, [xp+0x%x];\n",j,4*j,512*j);
        printf("--:2:1:-:2 @P%d LDG.E.CI.U16 d%d, [sp+0x%x];\n",j,j,8*j);
    }
    for(int j=0;j<4;j++) {
        printf("%s:-:3:-:2 F2F.F32.F16 d%d, d%d;\n",j?"--":"01",j,j);
        for(int t=0;t<4;t++) printf("--:-:-:-:%d BFE f%d, q%d, 0x%x;\n",t==3?6:1,t,j,0x800+8*t);
        for(int t=0;t<4;t++) printf("--:-:4:-:%d I2F.F32.S32 f%d, f%d;\n",t==3?2:1,t,t);
        for(int t=0;t<4;t++) printf("%s:-:-:-:%d FMUL f%d, f%d, d%d;\n",t?"--":"0c",t==3?3:1,t,t,j);
        for(int t=0;t<4;t++) printf("--:-:-:-:%d FFMA a%d, f%d, x%02d, a%d;\n",t==3?3:1,t,t,4*j+t,t);
    }
    out("--:-:-:-:6 IADD base, base, 0x200;\n"
        "--:-:-:-:6 IADD ix, ix, 0x200;\n");
    printf("--:-:-:-:6 ISETP.LT.U32.AND P4, PT, base, 0x%x, PT;\n",k);
    out("02:-:-:-:6 IADD qp0.CC, qp0, 0x200;\n--:-:-:-:6 IADD.X qp1, qp1, RZ;\n"
        "--:-:-:-:6 IADD xp0.CC, xp0, 0x800;\n--:-:-:-:6 IADD.X xp1, xp1, RZ;\n"
        "--:-:-:-:6 IADD sp0.CC, sp0, 0x20;\n--:-:-:-:6 IADD.X sp1, sp1, RZ;\n"
        "--:-:-:-:6 @P4 BRA LOOP;\n"
        "--:-:-:-:1 FADD a0, a0, a1;\n--:-:-:-:6 FADD a2, a2, a3;\n"
        "--:-:-:-:6 FADD a0, a0, a2;\n");
    for(int s=16;s;s>>=1) printf("--:-:5:-:2 SHFL.BFLY PT, sh, a0, 0x%x, 0x1f;\n10:-:-:-:6 FADD a0, a0, sh;\n",s);
    out("--:2:-:-:6 @P6 STG.E [yp], a0;\n02:-:-:-:f EXIT;\n");
    return ferror(stdout)?1:0;
}
