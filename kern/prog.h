/* Phase programs for the persistent kernel (die.pl): descriptor builders per op
 * and the block programs. Parameter offsets are each op's standalone
 * CONSTANT_MAPPING (descriptor word 2 + (addr - 0x140) / 4, see die.pl). */
#ifndef FLASH_PROG_H
#define FLASH_PROG_H
#include "dieops.h"
#include <cuda.h>
#include <stdint.h>

typedef struct { uint32_t w[32]; } phase_t;
typedef struct { phase_t *p; int n, cap; } prog_t;

void prog_init(prog_t *P);
void prog_free(prog_t *P);
phase_t *prog_add(prog_t *P, int op, unsigned units);
void ph_ptr(phase_t *p, int addr, CUdeviceptr v);
void ph_u32(phase_t *p, int addr, uint32_t v);
void ph_f32(phase_t *p, int addr, float v);

/* ---- one phase per op (pointers are device addresses) ---- */
void ph_rms(prog_t *P, int op, CUdeviceptr X, CUdeviceptr W, CUdeviceptr Y, float eps, unsigned vectors);
void ph_bf16(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Y, unsigned rows, CUdeviceptr bias);
void ph_hc(prog_t *P, int op, CUdeviceptr P0, CUdeviceptr P1, CUdeviceptr P2, unsigned n, unsigned units);
void ph_actq(prog_t *P, int op, CUdeviceptr X, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors);
void ph_ffnact(prog_t *P, CUdeviceptr G, CUdeviceptr U, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors);
void ph_gemvk(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr AS);
void ph_q2(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS, CUdeviceptr WE);
void ph_b32(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS);
void ph_conv1d(prog_t *P, CUdeviceptr X, CUdeviceptr STATE, CUdeviceptr W, CUdeviceptr U, unsigned t, unsigned n);
void ph_dnstep(prog_t *P, CUdeviceptr S, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr A, CUdeviceptr B, CUdeviceptr SSMA, CUdeviceptr O, float eps);
void ph_dnnorm(prog_t *P, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, CUdeviceptr Y, float eps);
void ph_router(prog_t *P, CUdeviceptr R, CUdeviceptr SH, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr S);

/* the GEMV opcode for a GGUF type and K (0 if none), and whether the op is Q2_0 / b32 / K-quant */
int gemv_op(uint32_t type, int K, int acc);
int gemv_op_is_q2(int op);
int gemv_op_is_b32(int op);

/* ---- a Gated DeltaNet block (one die, every row) ---- */
typedef struct {
	CUdeviceptr hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
	CUdeviceptr hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
	CUdeviceptr attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_dt_bias, ssm_conv1d, ssm_a, ssm_norm, ssm_out;
	uint32_t attn_qkv_type, attn_gate_type, ssm_out_type;
	CUdeviceptr ffn_gate_inp, ffn_gate_inp_shexp, ffn_gate_exps, ffn_up_exps, ffn_down_exps;
	CUdeviceptr ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
	uint32_t ffn_gate_shexp_type, ffn_up_shexp_type, ffn_down_shexp_type;
} dn_weights_t;

/* activation / state buffers (device); sizes in floats unless noted */
typedef struct {
	CUdeviceptr S;                    /* 10240 in/out stream */
	CUdeviceptr xh, down, inj, t, up, z;   /* 10240, 320, 4, 320, 10240, 2560 */
	CUdeviceptr zq, zs32, zs16, zas;  /* int16 2560, i32 80, i32 160, f32 80 */
	CUdeviceptr qkv, gate, a, b, u, o, on;  /* 10240, 6144, 48, 48, 10240, 6144, 6144 */
	CUdeviceptr oq, os32, os16, oas;  /* int16 6144, 192, 384, 192 */
	CUdeviceptr y;                    /* 2560 */
	CUdeviceptr r, sh, idx, w, s;     /* 512, 1, u32 10, f32 10, f32 1 */
	CUdeviceptr g, gu;                /* 11 x 640 each */
	CUdeviceptr hq, hs32, hs16, has;  /* int16 11x640, i32 11x20, i32 11x40, f32 11x20 */
	CUdeviceptr out;                  /* 2560 */
	CUdeviceptr conv_state, dn_state; /* 4 x 10240 ring, 48 x 128 x 128 */
} dn_bufs_t;

void prog_deltanet_block(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps);
#endif
