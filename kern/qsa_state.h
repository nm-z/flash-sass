/* Shared production QSA capacity and per-layer descriptor contract. */
#ifndef FLASH_QSA_STATE_H
#define FLASH_QSA_STATE_H
#include "prog.h"
#include <stddef.h>
static inline unsigned qsa_cache_positions(unsigned positions) {
    return (positions + 127u) & ~127u;
}
static inline unsigned qsa_compact_positions(unsigned nblk) {
    return (4u * (nblk + 1u) + 2048u + 31u) / 32u * 32u;
}
static inline size_t qsa_layer_cache_bytes(unsigned positions) {
    return (size_t)positions * 1088u;
}
static inline void qsa_bind_layer(dn_bufs_t *b, unsigned nblk, CUdeviceptr kq, CUdeviceptr ks, CUdeviceptr vq, CUdeviceptr vs, CUdeviceptr pkt, CUdeviceptr ring) {
    b->qsa = 1; b->nblk = nblk;
    b->kq = kq; b->ks = ks; b->vq = vq; b->vs = vs; b->pkt = pkt; b->ring = ring;
}
#endif
