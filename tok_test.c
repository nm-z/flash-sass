/* tok_test <gguf> <file> [raw]: the file's bytes -> ids, printed like llama-tokenize --ids
 * ("[1, 2, 3]"); raw = control tokens stay text (llama-tokenize --no-parse-special). With
 * "pieces" instead of a file, decodes ids from the remaining arguments back to text. */
#include "tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: tok_test <gguf> <file> [raw] | <gguf> pieces <id>...\n"); return 2; }
	tok_vocab *v = tok_load(argv[1]); if (!v) return 1;
	if (!strcmp(argv[2], "template")) { const char *t = tok_template(v); if (!t) return 1; fputs(t, stdout); return 0; }
	if (!strcmp(argv[2], "pieces")) {
		char buf[1024];
		for (int i = 3; i < argc; i++) { int n = tok_piece(v, (uint32_t)atoi(argv[i]), buf, sizeof buf); if (n > 0) fwrite(buf, 1, (size_t)n, stdout); }
		return 0;
	}
	FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
	char *s = NULL; size_t cap = 0, n = 0; int c;
	while ((c = fgetc(f)) != EOF) { if (n == cap) { cap = cap ? 2 * cap : 4096; s = realloc(s, cap); } s[n++] = (char)c; }
	fclose(f);
	int max = (int)n + 16; uint32_t *ids = malloc(sizeof *ids * (size_t)max);
	int k = tok_encode(v, s ? s : "", n, !(argc > 3 && !strcmp(argv[3], "raw")), ids, max);
	printf("[");
	for (int i = 0; i < k; i++) printf(i ? ", %u" : "%u", ids[i]);
	printf("]\n");
	return 0;
}
