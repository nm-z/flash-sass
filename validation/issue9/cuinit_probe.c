#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>

int cuInit(unsigned int flags) {
    (void)flags;
    fputs("CUDA_BOUNDARY_REACHED\n", stderr);
    fflush(stderr);
    _Exit(99);
}
