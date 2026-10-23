/* Specialize the existing emitted SASS without changing its arithmetic emitter.
 * Q2MOEGN flag 32 selects a CTA-owned 32-row gate/up/SiLU/quantization tile.
 * The existing five-half-block dot product and shuffle order are copied exactly.
 * All added code is C or SASS; the legacy emitter remains unchanged. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *src;
static void die(const char *s) { fprintf(stderr, "decode_fuse: %s\n", s); exit(1); }
static char *at(const char *s) { char *p = strstr(src, s); if (!p) die(s); return p; }
static void part(const char *a, const char *b) {
    const char *p = at(a), *q = strstr(p, b); if (!q) die(b);
    fwrite(p, 1, (size_t)(q-p), stdout);
}
static void out(const char *s) { fputs(s, stdout); }
static void delay(void) { out("--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n"); }
static void dispatch(const char *start,const char *end) {
    const char *jump=strstr(start,"BRX ");
    if(jump && jump<end) { fwrite(start,1,(size_t)(end-start),stdout); return; }
    char names[128][96]; int n=0;
    for(const char *p=start;(p=strstr(p,"@P6 BRA ")) && p<end;p++) {
        if(n==128 || sscanf(p+8,"%95[^;];",names[n])!=1) die("dispatch entry");
        n++;
    }
    if(n<60) die("unexpected opcode table");
    out("OP_DISPATCH:\n");
    printf("--:-:-:-:6 ISETP.GE.U32.AND P6, PT, op, 0x%x, PT;\n",n);
    out("--:-:-:-:6 SHL fw0, op, 0x5;\n"
        "--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n"
        "--:-:-:-:6 @P6 BRA POSITION_END;\n"
        "--:-:-:-:6 BRX fw0 GA_DISPATCH_TABLE;\nGA_DISPATCH_TABLE:\n");
    for(int i=0;i<n;i++) printf("--:-:-:-:6 BRA %s;\n--:-:-:-:6 NOP;\n--:-:-:-:6 NOP;\n",names[i]);
}
static void silu(void) {
    out("--:-:-:-:6 FADD ga_t4, -ga_gate, RZ;\n"
        "--:-:-:-:6 FMNMX ga_t4, ga_t4, 87, PT;\n"
        "--:-:-:-:6 FMNMX ga_t4, ga_t4, -87, !PT;\n"
        "--:-:-:-:6 FMUL ga_t1, ga_t4, ga_log2e;\n"
        "--:-:-:-:6 FADD ga_t2, ga_t1, 12582912;\n"
        "--:-:-:-:6 FADD ga_t2, ga_t2, -12582912;\n"
        "--:-:-:-:6 FFMA ga_t3, ga_t2, -ga_ln2hi, ga_t4;\n"
        "--:-:-:-:6 FFMA ga_t3, ga_t2, -ga_ln2lo, ga_t3;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_c720, ga_t3, ga_c120;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_t1, ga_t3, ga_c24;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_t1, ga_t3, ga_c6;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_t1, ga_t3, ga_chalf;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_t1, ga_t3, ga_cone;\n"
        "--:-:-:-:6 FFMA ga_t1, ga_t1, ga_t3, ga_cone;\n"
        "--:-:3:-:2 F2I.S32.F32 ga_t2, ga_t2;\n"
        "04:-:-:-:6 IADD ga_t2, ga_t2, 0x7f;\n"
        "--:-:-:-:6 SHL ga_t2, ga_t2, 0x17;\n"
        "--:-:-:-:6 FMUL ga_t4, ga_t1, ga_t2;\n"
        "--:-:-:-:6 FADD ga_t4, ga_t4, 1;\n"
        "--:-:3:-:2 MUFU.RCP ga_t5, ga_t4;\n"
        "04:-:-:-:6 FFMA ga_t1, ga_t4, ga_t5, ga_mneg1;\n"
        "--:-:-:-:6 FADD.FTZ ga_t1, -ga_t1, -RZ;\n"
        "--:-:-:-:6 FFMA ga_t5, ga_t5, ga_t1, ga_t5;\n"
        "--:-:-:-:6 FMUL ga_value, ga_gate, ga_t5;\n"
        "--:-:-:-:6 FMUL ga_value, ga_value, q2moegn_accf;\n");
}
static void ptr(const char *base, int shift) {
    printf("--:-:-:-:6 SHL ga_smoff, ga_index, 0x%x;\n", shift);
    printf("02:-:-:-:6 IADD ga_addr0.CC, %s0, ga_smoff;\n", base);
    printf("--:-:-:-:6 IADD.X ga_addr1, %s1, RZ;\n", base);
}
static void fused(void) {
    out("\nGA_SETUP:\n");
    part("--:-:-:-:6      MOV q2moegn_tid, vtid;", "q2moegn_EXPERT:\n");
    out("--:-:-:-:6 MOV ga_gw0, q2moegn_param_W0;\n"
        "--:-:-:-:6 MOV ga_gw1, q2moegn_param_W1;\n"
        "--:-:-:-:6 MOV ga_gy0, q2moegn_param_Y0;\n"
        "--:-:-:-:6 MOV ga_gy1, q2moegn_param_Y1;\n"
        "--:-:-:-:6 MOV ga_uy0, q2moegn_param_WE0;\n"
        "--:-:-:-:6 MOV ga_uy1, q2moegn_param_WE1;\n");
    const char *p[] = {"ga_uw", "ga_q", "ga_s32", "ga_s16", "ga_as"};
    for (int i=0;i<5;i++) for (int j=0;j<2;j++)
        printf("--:-:1:-:2 LDS %s%d, [RZ+0x%x];\n",p[i],j,0xbfd0+8*i+4*j);
    const char *cn[] = {"log2e","ln2hi","ln2lo","c720","c120","c24","c6","chalf","cone","mneg1"};
    const unsigned cv[] = {0x3fb8aa3b,0x3f317200,0x35bfbe8e,0x3ab60b61,0x3c088889,0x3d2aaaab,0x3e2aaaab,0x3f000000,0x3f800000,0xbf800000};
    for (int i=0;i<10;i++) printf("--:-:-:-:1 MOV32I ga_%s, 0x%08x;\n",cn[i],cv[i]);
    out("--:-:-:-:6 MOV32I ga_c127, 0x3c010204;\n"
        "--:-:-:-:6 MOV ga_tile, cta;\n"
        "01:-:-:-:6 NOP;\nGA_EXPERT:\n");
    part("--:-:-:-:1      MOV q2moegn_tid, q2moegn_tidk;", "--:-:-:-:6      ISCADD q2moegn_row, q2moegn_bid, q2moegn_grp, 0x4;");
    out("GA_TILE:\n"
        "--:-:-:-:6 MOV ga_half, RZ;\nGA_HALF:\n"
        "--:-:-:-:6 MOV ga_stage, RZ;\n"
        "--:-:-:-:6 MOV q2moegn_param_W0, ga_gw0;\n"
        "--:-:-:-:6 MOV q2moegn_param_W1, ga_gw1;\n"
        "--:-:-:-:6 MOV q2moegn_param_Y0, ga_gy0;\n"
        "--:-:-:-:6 MOV q2moegn_param_Y1, ga_gy1;\nGA_DOT:\n"
        "02:-:-:-:6 LOP32I.AND q2moegn_lane, tid, 0xf;\n"
        "--:-:-:-:6 SHR.U32 q2moegn_grp, tid, 0x4;\n"
        "--:-:-:-:6 LOP32I.AND q2moegn_half, q2moegn_lane, 0x1;\n"
        "--:-:-:-:6 SHR.U32 q2moegn_blk0, q2moegn_lane, 0x1;\n"
        "--:-:-:-:6 ISETP.EQ.U32.AND P0, PT, q2moegn_lane, RZ, PT;\n"
        "--:-:-:-:6 ISCADD q2moegn_row, ga_tile, q2moegn_grp, 0x5;\n"
        "--:-:-:-:6 IADD q2moegn_row, q2moegn_row, ga_half;\n");
    part("--:-:-:-:6      SHL q2moegn_off, q2moegn_row, 0x4;", "q2moegn_LOOP:\n");
    part("--:2:1:-:1      LDG.E.CI.U16 q2moegn_d0, [q2moegn_ws+0x0];", "--:2:-:-:1  @P2 STG.E [q2moegn_yp], q2moegn_accf;");
    out("--:2:-:-:6 @P0 STG.E [q2moegn_yp], q2moegn_accf;\n"
        "--:-:-:-:6 ISETP.NE.U32.AND P4, PT, ga_stage, RZ, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_ACT;\n"
        "--:-:-:-:6 MOV ga_gate, q2moegn_accf;\n"
        "--:-:-:-:6 MOV ga_stage, 0x1;\n"
        "--:-:-:-:6 MOV q2moegn_param_W0, ga_uw0;\n"
        "--:-:-:-:6 MOV q2moegn_param_W1, ga_uw1;\n"
        "--:-:-:-:6 MOV q2moegn_param_Y0, ga_uy0;\n"
        "--:-:-:-:6 MOV q2moegn_param_Y1, ga_uy1;\n"
        "--:-:-:-:6 BRA GA_DOT;\nGA_ACT:\n");
    silu();
    out("--:-:-:-:6 LOP32I.AND ga_smoff, q2moegn_row, 0x1f;\n"
        "--:-:-:-:6 SHL ga_smoff, ga_smoff, 0x2;\n"
        "--:2:-:-:6 @P0 STS [ga_smoff+0x2000], ga_value;\n"
        "--:-:-:-:6 IADD ga_half, ga_half, 0x10;\n"
        "--:-:-:-:6 ISETP.LT.U32.AND P4, PT, ga_half, 0x20, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_HALF;\n"
        "3f:-:-:-:6 BAR.SYNC 0x0;\n"
        "--:-:-:-:6 ISETP.GE.U32.AND P4, PT, tid, 0x20, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_QUANT_DONE;\n"
        "--:-:-:-:6 SHL ga_smoff, tid, 0x2;\n"
        "--:-:1:-:2 LDS ga_value, [ga_smoff+0x2000];\n"
        "01:-:-:-:6 FMNMX ga_max, |ga_value|, 7.8886090522101180541e-31, !PT;\n");
    for (int k=1;k<=16;k*=2) printf("--:-:5:-:2 SHFL.BFLY PT, ga_tmp, ga_max, 0x%x, 0x1f;\n10:-:-:-:6 FMNMX ga_max, ga_max, ga_tmp, !PT;\n",k);
    out("--:-:3:-:2 MUFU.RCP ga_rcp, ga_max;\n"
        "04:-:-:-:6 FFMA ga_tmp, ga_max, ga_rcp, ga_mneg1;\n"
        "--:-:-:-:6 FADD.FTZ ga_tmp, -ga_tmp, -RZ;\n"
        "--:-:-:-:6 FFMA ga_rcp, ga_rcp, ga_tmp, ga_rcp;\n"
        "--:-:-:-:6 FMUL ga_rcp, ga_rcp, 127;\n"
        "--:-:-:-:6 FMUL ga_max, ga_max, ga_c127;\n"
        "--:-:-:-:6 FMUL ga_value, ga_value, ga_rcp;\n"
        "--:-:3:-:2 F2I.S32.F32 ga_qi, ga_value;\n"
        "--:-:-:-:6 XMAD ga_index, q2moegn_e, 0x280, RZ;\n"
        "--:-:-:-:6 ISCADD ga_index, ga_tile, ga_index, 0x5;\n"
        "--:-:-:-:6 IADD ga_index, ga_index, tid;\n");
    ptr("ga_q",1);
    out("04:2:-:-:6 STG.E.U16 [ga_addr], ga_qi;\n"
        "--:-:-:-:6 MOV ga_sum16, ga_qi;\n");
    for (int k=1;k<=8;k*=2) printf("--:-:5:-:2 SHFL.BFLY PT, ga_tmp, ga_sum16, 0x%x, 0x1f;\n10:-:-:-:6 IADD ga_sum16, ga_sum16, ga_tmp;\n",k);
    out("--:-:5:-:2 SHFL.BFLY PT, ga_tmp, ga_sum16, 0x10, 0x1f;\n"
        "10:-:-:-:6 IADD ga_sum32, ga_sum16, ga_tmp;\n"
        "--:-:-:-:6 SHR.U32 ga_index, ga_index, 0x4;\n"
        "--:-:-:-:6 LOP32I.AND ga_tmp, tid, 0xf;\n"
        "--:-:-:-:6 ISETP.EQ.U32.AND P0, PT, ga_tmp, RZ, PT;\n");
    ptr("ga_s16",2);
    out("--:2:-:-:6 @P0 STG.E [ga_addr], ga_sum16;\n"
        "--:-:-:-:6 SHR.U32 ga_index, ga_index, 0x1;\n"
        "--:-:-:-:6 ISETP.EQ.U32.AND P0, PT, tid, RZ, PT;\n");
    ptr("ga_s32",2);
    out("--:2:-:-:6 @P0 STG.E [ga_addr], ga_sum32;\n");
    ptr("ga_as",2);
    out("--:2:-:-:6 @P0 STG.E [ga_addr], ga_max;\n"
        "GA_QUANT_DONE:\n3f:-:-:-:6 BAR.SYNC 0x0;\n"
        "--:-:-:-:6 IADD ga_tile, ga_tile, 0x10;\n"
        "--:-:-:-:6 ISETP.LT.U32.AND P4, PT, ga_tile, 0x14, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_TILE;\n"
        "--:-:-:-:6 IADD ga_tile, ga_tile, -0x14;\n"
        "--:-:-:-:6 IADD q2moegn_e, q2moegn_e, 0x1;\n"
        "--:-:-:-:6 ISETP.LT.U32.AND P4, PT, q2moegn_e, q2moegn_ne, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_EXPERT;\n"
        "--:-:-:-:6 BRA POSITION_END;\n");
}
int main(int argc, char **argv) {
    if (argc!=2) die("usage: decode_fuse die.sass");
    FILE *f=fopen(argv[1],"rb"); if (!f) die("open input");
    fseek(f,0,SEEK_END); long n=ftell(f); rewind(f);
    src=calloc((size_t)n+1,1); if (!src || fread(src,1,(size_t)n,f)!=(size_t)n) die("read input"); fclose(f);
    char *m=at("</REGISTER_MAPPING>"); fwrite(src,1,(size_t)(m-src),stdout);
    out("    180 : ga_flag\n    181 : ga_tile\n    182 : ga_half\n    183 : ga_stage\n"
        "    184,185 : ga_gw<0-1>\n    186,187 : ga_uw<0-1>\n"
        "    188 : ga_gate\n    189 : ga_value\n    190-194 : ga_t<1-5>\n"
        "    195 : ga_log2e\n    196 : ga_ln2hi\n    197 : ga_ln2lo\n"
        "    198 : ga_c720\n    199 : ga_c120\n    200 : ga_c24\n    201 : ga_c6\n"
        "    202 : ga_chalf\n    203 : ga_cone\n    204 : ga_mneg1\n"
        "    205 : ga_max\n    206 : ga_rcp\n    207 : ga_tmp\n    208 : ga_qi\n"
        "    209 : ga_sum16\n    210 : ga_sum32\n    212 : ga_smoff\n    213 : ga_index\n"
        "    214,215 : ga_q<0-1>\n    216,217 : ga_s32<0-1>\n"
        "    218,219 : ga_s16<0-1>\n    220,221 : ga_as<0-1>\n"
        "    222,223 : ga_addr<0-1>\n    224 : ga_c127\n"
        "    226,227 : ga_gy<0-1>\n    228,229 : ga_uy<0-1>\n");
    char *insert=at("--:-:-:-:6      MOV q2moegn_tid, vtid;");
    char *ds=at("OP_DISPATCH:\n"), *de=at("\nembrow_SETUP:\n");
    fwrite(m,1,(size_t)(ds-m),stdout);
    dispatch(ds,de);
    fwrite(de,1,(size_t)(insert-de),stdout);
    out("--:-:1:-:2 LDS ga_flag, [RZ+0xbff8];\n"
        "01:-:-:-:6 LOP32I.AND ga_flag, ga_flag, 0x20;\n"
        "--:-:-:-:6 ISETP.NE.U32.AND P4, PT, ga_flag, RZ, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_SETUP;\n");
    char *end=at("\nPOSITION_END:\n");
    fwrite(insert,1,(size_t)(end-insert),stdout);
    fused();
    out("\nPOSITION_END:\n"
        "3f:-:1:-:2 LDS bar_g, [RZ+0xbf54];\n"
        "01:-:-:-:6 ISETP.EQ.U32.AND P5, PT, bar_g, RZ, PT;\n"); delay();
    out("--:-:-:-:6 @P5 BRA PHASE_END;\n");
    char *tail=end+strlen("\nPOSITION_END:\n");
    char *spin=strstr(tail,"\nBSPIN:\n");
    if(!spin) die("grid barrier poll");
    char *arrive=strstr(spin,"\n--:-:-:-:6      MEMBAR.GL.IVALLD;\n02:-:-:-:6      IADD bar_t, cta, tid;");
    if(!arrive) die("grid barrier arrival");
    fwrite(tail,1,(size_t)(spin-tail),stdout);
    out("\n02:-:1:-:2 LDS bar_old, [RZ+0xbf58];\n"
        "01:-:-:-:6 LOP32I.AND bar_old, bar_old, 0x40;\n"
        "--:-:-:-:6 ISETP.NE.U32.AND P4, PT, bar_old, RZ, PT;\n"); delay();
    out("--:-:-:-:6 @P4 BRA GA_BARRIER_ARRIVED;\n");
    fwrite(spin,1,(size_t)(arrive-spin),stdout);
    out("\nGA_BARRIER_ARRIVED:\n"); out(arrive);
    free(src); return ferror(stdout)?1:0;
}
