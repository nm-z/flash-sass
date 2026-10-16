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
