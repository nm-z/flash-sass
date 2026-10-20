/* Isolated issue110 diagnostic ABI. Records are last observations, not fault PCs.
 * The GPU writes one naturally aligned uint64_t per CTA to mapped host memory.
 * No synchronization is added to the model program by this interface. */
#ifndef FLASH_PHASE_PROBE_H
#define FLASH_PHASE_PROBE_H
#include "prog.h"
#include <stddef.h>
#include <stdint.h>

#define PHASE_PROBE_CTL_OFFSET 0xe8u
#define PHASE_PROBE_CTAS 16u
#define PHASE_PROBE_DIES 6u
#define PHASE_PROBE_PHASES 4096u

typedef struct {
    uint32_t opcode, units, flags;
} phase_probe_descriptor;

typedef struct {
    uint32_t base_token, token, primary_index, physical_index, mode;
    phase_probe_descriptor descriptor;
} phase_probe_record;

/* Tag: high32=kernel token base; low32=((primary descriptor index+1)<<5)|MODE.
 * MODE is 0 (scalar or generic second body), 8 (generic first body), or 24
 * (fused pair). A zero tag means no observation. The uploaded primary flags
 * distinguish scalar mode zero from the second body of a generic pair. */
int phase_probe_decode(uint64_t tag, const phase_probe_descriptor *program,
                       size_t count, uint32_t first, uint32_t end,
                       phase_probe_record *out);

/* These functions are for one engine host thread and its selected contexts.
 * create makes the diagnostic active; a library built without the probe flag
 * simply never calls it. Fault reporting reads host memory only and exits 70. */
CUresult phase_probe_create(const CUcontext *contexts, unsigned count);
void phase_probe_destroy(void);
int phase_probe_enabled(void);
CUdeviceptr phase_probe_pointer(unsigned die);
/* A single host load, also available after a CUDA fault; zero if unavailable. */
uint64_t phase_probe_observe(unsigned die, unsigned cta);
void phase_probe_upload_begin(void);
int phase_probe_program(unsigned die, const prog_t *program);
void phase_probe_launch(uint64_t request, uint32_t first, uint32_t end);
void phase_probe_done(void);
void phase_probe_fault(CUresult error, const char *expression, int line);
#endif
