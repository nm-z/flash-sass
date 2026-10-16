/* Gated DeltaNet head step exactly as kern/dnstep.sass computes it (one CTA per
 * value head), so the C version is a bit-exact twin.
 *
 * State S for one head is value-major: S[c][i], c = value index (0..127),
 * i = key index (0..127). Thread (g, p), g = 0..31, p = 0..7, owns columns
 * c = 4g..4g+3 and keys i = 16p..16p+15.
 *
 *   q^ = q * rcp(max(sqrt(sum q^2), eps))      (sum: 4-per-lane fmaf chains, xor 16..1 butterfly)
 *   k^ likewise
 *   alpha = exp(ssm_a * softplus(a)), beta = sigmoid(b)          (fmath twins)
 *   S[c][i] = alpha * S[c][i]
 *   r_c = sum_i k^_i S[c][i]     (per part p a 16-term fmaf chain, parts combined xor 1, 2, 4)
 *   e_c = v_c - r_c ; be_c = beta * e_c
 *   S[c][i] = fmaf(k^_i, be_c, S[c][i])
 *   o_c = sum_i q^_i S[c][i]     (same reduction shape) * 0.088388347648318f (1/sqrt 128)
 */
#ifndef FLASH_DELTANET_H
#define FLASH_DELTANET_H
/* S: 128*128 floats (updated in place), q/k/v: 128 floats, a/b/ssm_a scalars, o: 128 floats out */
void deltanet_step_gpu(float *S, const float *q, const float *k, const float *v, float a, float b, float ssm_a, float eps, float *o);
/* the L2 normalisation alone (q^ = q * rcp(max(sqrt(sum q^2), eps))), 128 elements */
void l2norm128_gpu(const float *q, float eps, float *qn);
#endif

/* Depthwise causal conv (k = 4) + SiLU over n channels, as kern/conv1d.sass: the
 * ring state[4][n] holds x(t) in slot t mod 4; y_c = ((w[c][0] x_c(t-3)) + w[c][1]
 * x_c(t-2) + w[c][2] x_c(t-1)) + w[c][3] x_c(t) by FMUL then three fmaf, u = SiLU(y)
 * (fmath twin); x(t) is written into slot t mod 4. w is [c][k] (k contiguous). */
void conv1d_silu_gpu(const float *x, float *state, const float *w, unsigned t, float *u, int n);

/* kern/dnnorm.sass: readout norm + gate per value head (128 wide): lane l owns 4
 * elements; ss by fmaf chains and the xor 16..1 butterfly; m = ss * (1/128); v = m +
 * eps; s = sqrtf(v); r = rcp(s); y = ((o * r) * w) * sigmoid(gate). o, gate, y: 6144
 * (48 heads x 128, head-major); w: 128 (ssm_norm). */
void dnnorm_gpu(const float *o, const float *w, const float *gate, float eps, float *y);
