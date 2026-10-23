/* Validation-only engine library. See validation/issue18/README.md. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REUSE_GUARD_BYTES 256u
#define REUSE_GUARD_VALUE 0xa5
typedef struct {
	CUdeviceptr data, base;
	void *host_data, *host_base;
	size_t bytes;
	CUcontext context;
	int live;
} reuse_guard;
static reuse_guard *reuse_guards;
static size_t reuse_guard_count, reuse_guard_capacity;

/* Calls in these wrappers resolve to the real CUDA APIs before the aliases
 * below intercept allocations made by the included production engine. */
static reuse_guard *reuse_add_guard(size_t bytes) {
	if (bytes > SIZE_MAX - 2 * REUSE_GUARD_BYTES) return NULL;
	if (reuse_guard_count == reuse_guard_capacity) {
		size_t capacity = reuse_guard_capacity ? reuse_guard_capacity * 2 : 1024;
		void *p = realloc(reuse_guards, capacity * sizeof *reuse_guards);
		if (!p) return NULL;
		reuse_guards = p; reuse_guard_capacity = capacity;
	}
	reuse_guard *g = &reuse_guards[reuse_guard_count++];
	memset(g, 0, sizeof *g);
	g->bytes = bytes; g->live = 1;
	if (cuCtxGetCurrent(&g->context) != CUDA_SUCCESS) { g->live = 0; return NULL; }
	return g;
}
static CUresult reuse_alloc(CUdeviceptr *p, size_t bytes) {
	reuse_guard *g = reuse_add_guard(bytes);
	if (!g) return CUDA_ERROR_OUT_OF_MEMORY;
	CUresult r = cuMemAlloc(&g->base, bytes + 2 * REUSE_GUARD_BYTES);
	if (r) { g->live = 0; return r; }
	g->data = g->base + REUSE_GUARD_BYTES; *p = g->data;
	r = cuMemsetD8(g->base, REUSE_GUARD_VALUE, REUSE_GUARD_BYTES);
	if (!r) r = cuMemsetD8(g->data + bytes, REUSE_GUARD_VALUE, REUSE_GUARD_BYTES);
	return r;
}
static CUresult reuse_host_alloc(void **p, size_t bytes, unsigned flags) {
	reuse_guard *g = reuse_add_guard(bytes);
	if (!g) return CUDA_ERROR_OUT_OF_MEMORY;
	CUresult r = cuMemHostAlloc(&g->host_base, bytes + 2 * REUSE_GUARD_BYTES, flags);
	if (r) { g->live = 0; return r; }
	g->host_data = (unsigned char *)g->host_base + REUSE_GUARD_BYTES;
	*p = g->host_data;
	memset(g->host_base, REUSE_GUARD_VALUE, REUSE_GUARD_BYTES);
	memset((unsigned char *)g->host_data + bytes, REUSE_GUARD_VALUE, REUSE_GUARD_BYTES);
	return CUDA_SUCCESS;
}
static CUresult reuse_host_map(CUdeviceptr *p, void *host, unsigned flags) {
	for (size_t i = 0; i < reuse_guard_count; i++) {
		reuse_guard *g = &reuse_guards[i];
		if (g->live && g->host_data == host) {
			CUresult r = cuMemHostGetDevicePointer(p, g->host_base, flags);
			if (!r) *p += REUSE_GUARD_BYTES;
			return r;
		}
	}
	return cuMemHostGetDevicePointer(p, host, flags);
}
static CUresult reuse_free(CUdeviceptr p) {
	for (size_t i = 0; i < reuse_guard_count; i++) {
		reuse_guard *g = &reuse_guards[i];
		if (g->live && !g->host_data && g->data == p) {
			g->live = 0; return cuMemFree(g->base);
		}
	}
	return cuMemFree(p);
}
static CUresult reuse_host_free(void *p) {
	for (size_t i = 0; i < reuse_guard_count; i++) {
		reuse_guard *g = &reuse_guards[i];
		if (g->live && g->host_data == p) {
			g->live = 0; return cuMemFreeHost(g->host_base);
		}
	}
	return cuMemFreeHost(p);
}
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemHostAlloc
#undef cuMemFreeHost
#undef cuMemHostGetDevicePointer
#define cuMemAlloc reuse_alloc
#define cuMemFree reuse_free
#define cuMemHostAlloc reuse_host_alloc
#define cuMemFreeHost reuse_host_free
#define cuMemHostGetDevicePointer reuse_host_map
#define engine_session engine_original_session
#define engine_fini engine_original_fini
#include "engine.c"
#undef engine_session
#undef engine_fini

void engine_fini(void) {
	engine_original_fini();
	free(reuse_guards); reuse_guards = NULL;
	reuse_guard_count = reuse_guard_capacity = 0;
}

static int reuse_check_guards(FILE *out, const char *stage) {
	unsigned char b[2 * REUSE_GUARD_BYTES];
	size_t checked = 0; int bad = 0;
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
	for (size_t i = 0; i < reuse_guard_count; i++) {
		reuse_guard *g = &reuse_guards[i];
		if (!g->live) continue;
		checked++;
		if (g->host_data) {
			memcpy(b, g->host_base, REUSE_GUARD_BYTES);
			memcpy(b + REUSE_GUARD_BYTES, (unsigned char *)g->host_data + g->bytes, REUSE_GUARD_BYTES);
		} else {
			CK(cuCtxSetCurrent(g->context));
			CK(cuMemcpyDtoH(b, g->base, REUSE_GUARD_BYTES));
			CK(cuMemcpyDtoH(b + REUSE_GUARD_BYTES, g->data + g->bytes, REUSE_GUARD_BYTES));
		}
		for (size_t j = 0; j < sizeof b; j++) if (b[j] != REUSE_GUARD_VALUE) {
			fprintf(out, "reusecheck GUARD_FAIL stage=%s allocation=%zu side=%s byte=%zu\n",
				stage, i, j < REUSE_GUARD_BYTES ? "before" : "after", j % REUSE_GUARD_BYTES);
			bad = 1; break;
		}
	}
	CK(cuCtxSetCurrent(ctx[0]));
	fprintf(out, "reusecheck guards stage=%s allocations=%zu result=%s\n", stage, checked, bad ? "FAIL" : "PASS");
	return bad;
}
typedef struct {
	uint32_t tokens[3];
	int ntokens;
	float *logits[MAXD];
	size_t count[MAXD];
	run_stats stats;
} reuse_result;
static int reuse_device_fault;
static void reuse_drop(reuse_result *r) {
	for (int d = 0; d < nd; d++) free(r->logits[d]);
	memset(r, 0, sizeof *r);
}
static int reuse_run(FILE *out, const char *stage, const uint32_t *ids, int prompt,
		int snapshot, reuse_result *r) {
	const uint32_t stop[2] = {0xffffffffu, 0xffffffffu};
	const sampling sm = {0, 0.f, 1.f, 0};
	memset(r, 0, sizeof *r);
	int rc = engine_run(ids, prompt, 2, stop, &sm, snapshot, NULL, 0, NULL, &r->stats);
	if (rc) {
		if (rc == -2) reuse_device_fault = 1;
		fprintf(out, "reusecheck RUN_FAIL stage=%s rc=%d\n", stage, rc); return 1;
	}
	int bad = reuse_check_guards(out, stage);
	r->ntokens = r->stats.ntok - (prompt - 1);
	if (r->ntokens != 3) {
		fprintf(out, "reusecheck TOKEN_COUNT_FAIL stage=%s got=%d expected=3\n", stage, r->ntokens);
		return 1;
	}
	CK(cuCtxSetCurrent(ctx[0]));
	CK(cuMemcpyDtoH(r->tokens, db[0].toks + 4u * (unsigned)(prompt - 1), sizeof r->tokens));
	for (int d = 0; d < nd; d++) {
		r->count[d] = (size_t)sp[d][0].on;
		r->logits[d] = malloc(sizeof(float) * r->count[d]);
		if (!r->logits[d]) return 1;
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuMemcpyDtoH(r->logits[d], db[d].hlo, sizeof(float) * r->count[d]));
		for (size_t i = 0; i < r->count[d]; i++) if (!isfinite(r->logits[d][i])) {
			fprintf(out, "reusecheck NONFINITE_LOGIT stage=%s die=%d index=%zu\n", stage, d, i);
			bad = 1; break;
		}
	}
	CK(cuCtxSetCurrent(ctx[0]));
	fprintf(out, "reusecheck run stage=%s prompt=%d snapshot=%d resumed=%d positions=%d run_ms=%.6f build_ms=%.6f tokens=%u,%u,%u\n",
		stage, prompt, r->stats.snapshot, r->stats.resumed, r->stats.ntok, r->stats.run_ms,
		r->stats.build_ms, r->tokens[0], r->tokens[1], r->tokens[2]);
	fflush(out);
	return bad;
}
static int reuse_equal(FILE *out, const char *stage, const reuse_result *a,
		const reuse_result *b, int expected_resume) {
	int bad = b->stats.resumed != expected_resume || a->ntokens != b->ntokens ||
		memcmp(a->tokens, b->tokens, sizeof a->tokens);
	size_t logits = 0;
	for (int d = 0; d < nd; d++) {
		logits += a->count[d];
		if (a->count[d] != b->count[d] ||
				memcmp(a->logits[d], b->logits[d], sizeof(float) * a->count[d])) {
			fprintf(out, "reusecheck LOGIT_MISMATCH stage=%s die=%d\n", stage, d); bad = 1;
		}
	}
	fprintf(out, "reusecheck compare stage=%s expected_resume=%d actual_resume=%d tokens=3 logits=%zu equality=bitwise result=%s\n",
		stage, expected_resume, b->stats.resumed, logits, bad ? "FAIL" : "PASS");
	return bad;
}
static int reuse_check(FILE *out, int longer, int shorter, int first_snapshot) {
	if (longer <= shorter || shorter < 5 || longer > MAXT - 2 ||
			first_snapshot < 1 || first_snapshot > shorter - 4) {
		fprintf(out, "reusecheck FAIL bad_sizes long=%d short=%d snap=%d maxt=%d\n",
			longer, shorter, first_snapshot, MAXT); return 1;
	}
	if (nlayers_run != NB || (!core->ple_disk && core->ple_compact)) {
		fprintf(out, "reusecheck FAIL requires all48layers and actual PLE table\n"); return 1;
	}
	fprintf(out, "reusecheck begin rev=%s kv_host=%d layers=%d nblk=%u dies=%d maxt=%d long=%d short=%d first_snapshot=%d fresh_allocations=%s\n",
		FLASH_REV, kv_host, nlayers_run, nblk_qsa, nd, MAXT, longer, shorter, first_snapshot,
		last_positions == 0 ? "yes" : "no");
	uint32_t *short_ids = malloc(sizeof(uint32_t) * (unsigned)shorter);
	uint32_t *long_ids = malloc(sizeof(uint32_t) * (unsigned)longer);
	if (!short_ids || !long_ids) { free(short_ids); free(long_ids); return 1; }
	/* Legal embedding rows; the loaded model supplies every weight and state. */
	for (int t = 0; t < shorter; t++) short_ids[t] = 1u + ((unsigned)t * 977u) % 200000u;
	int bad = reuse_check_guards(out, "allocation");
	for (int residue = 0; residue < 4 && !bad; residue++) {
		int snapshot = first_snapshot + residue;
		reuse_result cold = {0}, warm = {0}, discard = {0}, rollback = {0};
		char label[80]; snap_s = 0;
		snprintf(label, sizeof label, "cold-r%d", snapshot % 4);
		bad |= reuse_run(out, label, short_ids, shorter, 0, &cold);
		if (bad) goto done;
		for (int t = 0; t < longer; t++) long_ids[t] = 1u + ((unsigned)t * 991u + 17u) % 200000u;
		snprintf(label, sizeof label, "long-r%d", snapshot % 4);
		bad |= reuse_run(out, label, long_ids, longer, 0, &discard);
		reuse_drop(&discard); if (bad) goto done;
		snprintf(label, sizeof label, "short-after-long-r%d", snapshot % 4);
		bad |= reuse_run(out, label, short_ids, shorter, 0, &warm);
		if (!bad) bad |= reuse_equal(out, label, &cold, &warm, 0);
		if (bad) goto done;
		for (int t = 0; t < longer; t++)
			long_ids[t] = t < snapshot ? short_ids[t] : 1u + ((unsigned)t * 983u + 31u) % 200000u;
		snap_s = 0;
		snprintf(label, sizeof label, "snapshot-long-r%d", snapshot % 4);
		bad |= reuse_run(out, label, long_ids, longer, snapshot, &discard);
		if (discard.stats.snapshot != snapshot) {
			fprintf(out, "reusecheck SNAPSHOT_FAIL requested=%d actual=%d\n", snapshot, discard.stats.snapshot);
			bad = 1;
		}
		reuse_drop(&discard); if (bad) goto done;
		snprintf(label, sizeof label, "rollback-short-r%d", snapshot % 4);
		bad |= reuse_run(out, label, short_ids, shorter, 0, &rollback);
		if (!bad) bad |= reuse_equal(out, label, &cold, &rollback, snapshot);
done:
		reuse_drop(&cold); reuse_drop(&warm); reuse_drop(&discard); reuse_drop(&rollback);
	}
	free(short_ids); free(long_ids); snap_s = 0;
	fprintf(out, "reusecheck %s\n", bad ? "FAIL" : "PASS"); fflush(out);
	return bad;
}
/* Only this fixture's first command is intercepted. glibc's FILE pushback
 * replays other input to the unchanged production parser, including reload. */
int engine_session(FILE *in, FILE *out) {
	char *line = NULL; size_t capacity = 0;
	ssize_t n = getline(&line, &capacity, in);
	if (n < 0) { free(line); return 0; }
	if (!strncmp(line, "reusecheck", 10) &&
			(line[10] == ' ' || line[10] == '\n' || line[10] == '\0')) {
		int longer = 96, shorter = 64, snapshot = 16;
		sscanf(line + 10, "%d %d %d", &longer, &shorter, &snapshot);
		free(line); reuse_device_fault = 0;
		reuse_check(out, longer, shorter, snapshot);
		/* Preserve the production core's CUDA-context recovery protocol. */
		return reuse_device_fault ? 3 : 0;
	}
	for (ssize_t i = n; i > 0; i--) if (ungetc((unsigned char)line[i - 1], in) == EOF) {
		free(line); fprintf(out, "error: validation parser pushback failed\n"); fflush(out); return 0;
	}
	free(line);
	return engine_original_session(in, out);
}
