/* Hyper-connection vector ops exactly as kern/hcops.sass computes them (fmath twins). */
#ifndef FLASH_HC_H
#define FLASH_HC_H
/* t[i] = SiLU(d[i] * 0.25), i < 320 */
void hc_silu_gpu(const float *d, float *t);
/* z[c] = 0.25 * (((g0 xh0 + g1 xh1) + g2 xh2) + g3 xh3), g_l = sigmoid(up[2560 l + c]), xh_l = xh[2560 l + c]
 * (FMUL then three fmaf), c < 2560 */
void hc_mix_gpu(const float *up, const float *xh, float *z);
/* S[2560 l + c] = fmaf(w_l, y[c], S[2560 l + c]), w_l = 2 * sigmoid(0.25 * inj[l]) */
void hc_write_gpu(const float *inj, const float *y, float *S);
#endif
