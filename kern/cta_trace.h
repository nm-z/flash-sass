/* Bounded per-physical-CTA trace ABI. Stamps are device globaltimer nanoseconds. */
#ifndef CTA_TRACE_H
#define CTA_TRACE_H
#include <stdint.h>
#include <stddef.h>
enum { CTA_TRACE_CTAS = 16, CTA_TRACE_BYTES = 64 };
typedef struct {
    uint64_t wait_start, wait_end, body_start, body_end, end;
    uint32_t token, phase, cta, opcode, units, nonce;
} cta_trace_record;
_Static_assert(sizeof(cta_trace_record) == 64, "trace ABI size");
_Static_assert(offsetof(cta_trace_record, token) == 40, "trace metadata");
_Static_assert(offsetof(cta_trace_record, nonce) == 60, "trace nonce");
static inline const char *cta_trace_error(const cta_trace_record *r,
    uint32_t token, uint32_t phase, uint32_t cta, uint32_t opcode,
    uint32_t nonce, uint64_t older_than, uint32_t expected_units)
{
    if (!nonce || r->nonce != nonce) return "launch nonce";
    if (r->token != token) return "token";
    if (r->phase != phase) return "phase";
    if (r->cta != cta) return "physical CTA";
    if (r->opcode != opcode) return "opcode";
    if (r->units != expected_units) return "nonempty work count";
    if (!r->wait_start || r->wait_start <= older_than) return "stale timestamp";
    if (r->wait_start > r->wait_end || r->wait_end > r->body_start ||
        r->body_start > r->body_end || r->body_end > r->end ||
        r->wait_start == r->end) return "timestamp order";
    return NULL;
}
static inline int cta_trace_valid(const cta_trace_record *r,
    uint32_t token, uint32_t phase, uint32_t cta, uint32_t opcode,
    uint32_t nonce, uint64_t older_than, uint32_t expected_units)
{
    return cta_trace_error(r, token, phase, cta, opcode, nonce,
                           older_than, expected_units) == NULL;
}
static inline uint32_t cta_attnpartn_units(uint32_t keys,
                                         uint32_t physical_cta, int remapped)
{
    uint32_t ck = ((keys + 8191u) / 8192u) * 1024u, count = 0;
    for (uint32_t b = physical_cta; b < 32; b += CTA_TRACE_CTAS) {
        uint32_t chunk = remapped ? b >> 2 : b & 7u;
        if ((uint64_t)chunk * ck < keys) ++count;
    }
    return count;
}
#endif
