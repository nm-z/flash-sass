/* jinja_test <template-file> <vars.json>: renders the template with the JSON object's members
 * as variables and prints the result (an error goes to stderr, exit 1). */
#include "jinja.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, size_t *n) {
	FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
	size_t cap = 4096, k = 0; char *s = malloc(cap); int c;
	while ((c = fgetc(f)) != EOF) { if (k + 1 >= cap) { cap *= 2; s = realloc(s, cap); } s[k++] = (char)c; }
	fclose(f); s[k] = 0; *n = k; return s;
}

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: jinja_test <template> <vars.json>\n"); return 2; }
	size_t tn, vn; char *ts = slurp(argv[1], &tn), *vs = slurp(argv[2], &vn), err[512];
	jinja *t = jinja_compile(ts, tn, err, sizeof err);
	if (!t) { fprintf(stderr, "compile: %s\n", err); return 1; }
	arena *a = arena_new(); const char *je = NULL;
	jv *vars = json_parse(a, vs, vn, &je);
	if (!vars) { fprintf(stderr, "vars: %s\n", je); return 1; }
	size_t on; char *out = jinja_render(t, a, vars, &on, err, sizeof err);
	if (!out) { fprintf(stderr, "render: %s\n", err); return 1; }
	fwrite(out, 1, on, stdout);
	free(out); arena_free(a); jinja_free(t);
	return 0;
}
