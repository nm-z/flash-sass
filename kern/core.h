/* core.h: the interface between the persistent core (flashcore.c) and the reloadable engine
 * library (engine.c -> libengine.so).
 *
 * What takes minutes to load stays in the core process and survives engine changes:
 *   - the per-die CUDA contexts, peer access and the weight arenas (opened through CUDA IPC from the
 *     `hold` processes, which keep the weights resident and are restarted only when the split changes),
 *   - the PLE table in pinned host memory (28.8 GB; CUDA 11.4 cannot export pinned host memory to
 *     another process or pin a file mapping, so the process that pins it has to be the one whose
 *     kernels read it, and that process has to be the one that stays).
 * Everything that changes while developing is in the engine library, which the core dlopen()s:
 * the buffers, the programs, the request loop. `reload [path]` unloads the old library (engine_fini
 * frees its device memory) and loads the new one, in seconds. The cubin is reloaded the same way
 * (`cubin <path>`) without touching the library. */
#ifndef FLASH_CORE_H
#define FLASH_CORE_H
#include "hold.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#define CORE_ABI 1
#define CORE_MAXD 6

typedef struct {
	int abi;                  /* CORE_ABI: the engine refuses a core it was not built for */
	int nd;                   /* dies in use */
	int maxt;                 /* positions per response (FLASH_MAXT) */
	CUcontext ctx[CORE_MAXD];
	hold_arena ar[CORE_MAXD]; /* the weights of die d */
	uint8_t *ple_host;        /* the pinned PLE table; ple_compact: it is a zero table of 16 rows per position (no table loaded) */
	int ple_compact;
} core_t;

/* exported by libengine.so */
int engine_init(core_t *c);                  /* 0 on success; builds all device state */
void engine_fini(void);                      /* frees it */
/* serves one client's commands; returns 0 when the client is gone, 1 to quit the process, 2 to reload the library */
int engine_session(FILE *in, FILE *out);
const char *engine_reload_path(void);        /* after a 2: the library to load next (NULL: the same file) */

#endif
