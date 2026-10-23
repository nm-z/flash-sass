/* MoE router exactly as kern/router.sass: m0 = max r; 10 rounds of (max value,
 * lowest index among the unselected); e_j = exp(m_j - m0) (fmath); den = sum of
 * e_j in selection order (sequential adds); w_j = e_j * rcp(den); the shared gate
 * s = sigmoid(sh). r has 512 logits. */
#ifndef FLASH_ROUTER_H
#define FLASH_ROUTER_H
void router_gpu(const float *r, float sh, unsigned *idx, float *w, float *s);
#endif
