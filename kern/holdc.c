#define _POSIX_C_SOURCE 200809L
#include "hold.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hold_open(hold_arena *a, const char *manifest) {
	FILE *f = fopen(manifest, "r");
	if (!f) { perror(manifest); return 1; }
	char hex[256];
	memset(a, 0, sizeof *a);
	if (fscanf(f, "arena %zu die %d handle %255s", &a->bytes, &a->die, hex) != 3) { fclose(f); return 1; }
	CUipcMemHandle h;
	if (strlen(hex) != 2 * sizeof h) { fclose(f); return 1; }
	for (size_t i = 0; i < sizeof h; i++) { unsigned v; sscanf(hex + 2 * i, "%2x", &v); ((unsigned char *)&h)[i] = (unsigned char)v; }
	CUresult r = cuIpcOpenMemHandle(&a->base, h, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	if (r != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r, &s); fprintf(stderr, "cuIpcOpenMemHandle: %s (is hold running on die %d?)\n", s, a->die); fclose(f); return 1; }
	int cap = 64;
	a->t = malloc(sizeof(hold_tensor) * cap);
	hold_tensor t;
	while (fscanf(f, "%95s %zu %zu %u %d %d %d %d", t.name, &t.off, &t.bytes, &t.type, &t.rows, &t.K, &t.stride, &t.experts) == 8) {
		if (a->n == cap) { cap *= 2; a->t = realloc(a->t, sizeof(hold_tensor) * cap); }
		a->t[a->n++] = t;
	}
	fclose(f);
	return 0;
}

const hold_tensor *hold_find(const hold_arena *a, const char *name) {
	for (int i = 0; i < a->n; i++) if (!strcmp(a->t[i].name, name)) return &a->t[i];
	return NULL;
}

CUdeviceptr hold_ptr(const hold_arena *a, const char *name) {
	const hold_tensor *t = hold_find(a, name);
	return t ? a->base + t->off : 0;
}

void hold_close(hold_arena *a) {
	if (a->base) cuIpcCloseMemHandle(a->base);
	free(a->t);
	memset(a, 0, sizeof *a);
}
