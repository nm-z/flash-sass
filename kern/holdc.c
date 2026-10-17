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
	int rc;
	while ((rc = fscanf(f, "%95s %zu %zu %u %d %d %d %d", t.name, &t.off, &t.bytes, &t.type, &t.rows, &t.K, &t.stride, &t.experts)) == 8) {
		/* every tensor must lie inside the arena the handle opens: a manifest of another split or a stale one names ranges that do not */
		if (t.off > a->bytes || t.bytes > a->bytes - t.off) { fprintf(stderr, "%s: tensor %s [%zu, +%zu) is outside the %zu-byte arena\n", manifest, t.name, t.off, t.bytes, a->bytes); fclose(f); cuIpcCloseMemHandle(a->base); free(a->t); memset(a, 0, sizeof *a); return 1; }
		if (a->n == cap) { cap *= 2; a->t = realloc(a->t, sizeof(hold_tensor) * cap); }
		a->t[a->n++] = t;
	}
	if (rc != EOF || ferror(f)) { fprintf(stderr, "%s: unreadable tensor line after %d tensors (a truncated or damaged manifest)\n", manifest, a->n); fclose(f); cuIpcCloseMemHandle(a->base); free(a->t); memset(a, 0, sizeof *a); return 1; }
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

int hold_ordinal(int die) {
	const char *v = getenv("CUDA_VISIBLE_DEVICES");
	if (!v || !*v) return die;
	int ord = 0;
	for (const char *p = v; *p; ) {
		char *end; long d = strtol(p, &end, 10);
		if (end == p) break;
		if (d == die) return ord;
		ord++; p = end; while (*p == ',' || *p == ' ') p++;
	}
	return -1;
}
