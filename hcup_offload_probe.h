#ifndef FLASH_HCUP_OFFLOAD_PROBE_H
#define FLASH_HCUP_OFFLOAD_PROBE_H
#include <cuda.h>
#include <stdio.h>
/* Diagnostic build only. Borrows existing six-die contexts/functions and two
 * immutable IPC weight prefixes; creates/destroys no context or module. */
int hcup_offload_probe(FILE *out,const CUcontext *contexts,const CUfunction *functions,
    unsigned dies,unsigned width,CUdeviceptr source_weights,CUdeviceptr helper_weights);
#endif
