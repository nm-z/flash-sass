/* nullengine: a stand-in libengine.so that only exercises flashcore's plumbing (load, serve, reload,
 * fall back when a replacement fails to initialise). NULLENGINE_FAIL=1 makes engine_init fail;
 * NULLENGINE_TAG is printed so successive builds can be told apart. Commands: echo <text>, reload
 * [path], quit. */
#include "core.h"
#include <stdlib.h>
#include <string.h>

static char rp[512];
int engine_init(core_t *c) {
	fprintf(stderr, "nullengine[%s]: init: %d dies, table at %p (compact %d, first words %02x %02x), context 0 %p\n", getenv("NULLENGINE_TAG") ? getenv("NULLENGINE_TAG") : "-",
	        c->nd, (void *)c->ple_host, c->ple_compact, c->ple_host[4095], c->ple_host[0], (void *)c->ctx[0]);
#ifdef FAILINIT
	return -1;
#else
	return getenv("NULLENGINE_FAIL") ? -1 : 0;
#endif
}
void engine_fini(void) { fprintf(stderr, "nullengine[%s]: fini\n", getenv("NULLENGINE_TAG") ? getenv("NULLENGINE_TAG") : "-"); }
const char *engine_reload_path(void) { return rp[0] ? rp : NULL; }
int engine_session(FILE *in, FILE *out) {
	char line[512];
	while (fgets(line, sizeof line, in)) {
		if (!strncmp(line, "quit", 4)) return 1;
		if (!strncmp(line, "fault", 5)) return 3;   /* pretend the engine hit a CUDA error */
		if (!strncmp(line, "reload", 6)) { rp[0] = 0; sscanf(line + 6, "%511s", rp); return 2; }
		if (!strncmp(line, "echo ", 5)) { fputs(line + 5, out); fflush(out); }
	}
	return 0;
}
