/* Candidate capture serialization; the readback callback supplies observed bytes.
 * No numerical comparison, original-checkpoint provenance, or quality assertion
 * is made by this API. Partial writes never receive a complete manifest. */
#ifndef FLASH_QSA_CAPTURE_H
#define FLASH_QSA_CAPTURE_H
#include <stddef.h>
#include <stdint.h>
#define QSA_CAPTURE_VOCAB 248320u
#define QSA_CAPTURE_LAYERS 12u
#define QSA_CAPTURE_MAX_VISIBLE 262144u
#define QSA_CAPTURE_MAX_DIES 6u
/* Every buffer here is otherwise overwritten by the next QSA layer on that die.
 * Compact KV is retained too, so same-selection attention can be replayed. */
#define QSA_CAPTURE_SCRATCH_FIELDS(X) \
    X(qg) X(k) X(v) X(qo) X(iqk) X(iq) X(sc) X(hst) X(sel) \
    X(pm) X(pd) X(po) X(aout) X(kc) X(vc)
typedef struct {
    unsigned layer, die;
#define CAPTURE_ADDRESS(name) uint64_t name;
    QSA_CAPTURE_SCRATCH_FIELDS(CAPTURE_ADDRESS)
#undef CAPTURE_ADDRESS
    uint64_t pkt, ring, kq, vq, ks, vs;
    uint64_t qnorm, knorm, iqnorm, iknorm;
} qsa_capture_layer;
typedef struct {
    unsigned offset, count;
    uint64_t logits, ids;
} qsa_capture_die;
typedef struct {
    unsigned visible, nblk, dies;
    int kv_host, ple_disk;
    const uint32_t *ids;
    const char *revision;
    uint64_t cubin_fnv;
    qsa_capture_die die[QSA_CAPTURE_MAX_DIES];
    qsa_capture_layer layer[QSA_CAPTURE_LAYERS];
} qsa_capture_plan;
typedef int (*qsa_capture_read)(void *arg, unsigned die, uint64_t address, void *out, size_t bytes);
/* 0: capture written, NOT acceptance; -1: invalid domain, I/O, or readback failure.
 * The destination must not exist. On failure a partial directory may remain. */
int qsa_capture_write(const qsa_capture_plan *plan, qsa_capture_read readback,
                      void *arg, const char *new_dir, char *error, size_t error_size);
#endif
