#ifndef QSA_ORACLE_GPU_H
#define QSA_ORACLE_GPU_H
#include <stdint.h>
int qsaoracle_gpu_init(const char *cubin, int ordinal);
void qsaoracle_gpu_finish(void);
unsigned long qsaoracle_gpu_launch_count(void);
/* Production ATTNQKQ V quantizer; dequantization is a host-side observation. */
int qsaoracle_gpu_quant(const float *, unsigned, int8_t *, uint16_t *, float *);
int qsaoracle_gpu_scores(const float *iq, const uint16_t *pool, unsigned nv,
                         float *scores);
/* Returns UINT_MAX on error; dense bypass returns every ID without a launch. */
unsigned qsaoracle_gpu_select(const float *scores, unsigned nv, unsigned nblk,
                              uint32_t *ids);
int qsaoracle_gpu_attention(const float *query, const float *key,
                            const float *value, unsigned n, unsigned capacity,
                            const float *gate, float *out, float *pm, float *pd,
                            float *po);
int qsaoracle_gpu_index(const float *, unsigned, const float *, const float *,
                        const float *, const float *, const float *, float *,
                        uint16_t *, float *);
unsigned qsaoracle_gpu_scores_select(const float *, const uint16_t *, unsigned,
                                     unsigned, float *, uint32_t *);
#endif
