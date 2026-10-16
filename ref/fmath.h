/* Transcendentals exactly as the SASS library kern/fmath.pl computes them: only
 * IEEE fp32 adds, multiplies and fmaf plus the rcp_rn sequence, so a C twin is
 * bit-exact. Ranges: exp for x in [-87, 88] (clamped), the rest as noted. */
#ifndef FLASH_FMATH_H
#define FLASH_FMATH_H
float exp_gpu(float x);        /* 2^n * p(r), n = rint(x log2e), |r| <= ln2/2, degree-6 Taylor */
float rcp_gpu(float x);        /* IEEE 1/x (the ptxas rcp.rn fast path) */
float sigmoid_gpu(float x);    /* rcp_gpu(1 + exp_gpu(-x)) */
float silu_gpu(float x);       /* x * sigmoid_gpu(x) */
float log1p_gpu(float e);      /* log(1 + e) for e in [0, 1]: 2 atanh series in u = e/(2+e) to u^11 */
float softplus_gpu(float x);   /* max(x, 0) + log1p_gpu(exp_gpu(-|x|)) */
#endif
