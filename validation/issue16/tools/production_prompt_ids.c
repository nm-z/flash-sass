#define _POSIX_C_SOURCE 200809L
#include "jinja.h"
#include "tok.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *size) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return NULL; }
	if (fseek(f, 0, SEEK_END)) { perror(path); fclose(f); return NULL; }
	long n = ftell(f);
	if (n < 0 || fseek(f, 0, SEEK_SET)) { perror(path); fclose(f); return NULL; }
	char *data = malloc((size_t)n + 1);
	if (!data) { fprintf(stderr, "input allocation failed\n"); fclose(f); return NULL; }
	if (fread(data, 1, (size_t)n, f) != (size_t)n) {
		if (ferror(f)) perror(path); else fprintf(stderr, "short read from %s\n", path);
		free(data); fclose(f); return NULL;
	}
	if (fclose(f)) { perror(path); free(data); return NULL; }
	data[n] = '\0'; *size = (size_t)n; return data;
}

static int write_file(const char *path, const void *data, size_t size) {
	FILE *f = fopen(path, "wb");
	if (!f) { perror(path); return 1; }
	int bad = fwrite(data, 1, size, f) != size;
	if (fclose(f)) bad = 1;
	if (bad) { fprintf(stderr, "short write to %s\n", path); return 1; }
	return 0;
}

static int write_ids(const char *path, const uint32_t *ids, int count) {
	FILE *f = fopen(path, "wb");
	if (!f) { perror(path); return 1; }
	int bad = 0;
	for (int i = 0; i < count; ++i) if (fprintf(f, "%u\n", ids[i]) < 0) { bad = 1; break; }
	if (fclose(f)) bad = 1;
	if (bad) { fprintf(stderr, "failed writing token IDs to %s\n", path); return 1; }
	return 0;
}

int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "usage: production_prompt_ids <tokenizer-gguf> <user-prompt> <rendered-prompt> <ids.txt>\n");
		return 2;
	}
	size_t user_n = 0;
	char *user = read_file(argv[2], &user_n);
	if (!user) return 1;
	if (user_n > INT_MAX - 64) { fprintf(stderr, "user prompt is too large\n"); free(user); return 2; }

	tok_vocab *vocab = tok_load(argv[1]);
	if (!vocab) { free(user); return 1; }
	const char *source = tok_template(vocab);
	if (!source || !*source) { fprintf(stderr, "tokenizer GGUF has no chat template\n"); free(user); return 1; }
	char error[512];
	jinja *template = jinja_compile(source, strlen(source), error, sizeof error);
	if (!template) { fprintf(stderr, "chat template compile failed: %s\n", error); free(user); return 1; }

	arena *a = arena_new();
	if (!a) { fprintf(stderr, "Jinja arena allocation failed\n"); jinja_free(template); free(user); return 1; }
	jv *messages = jv_new(a, J_LIST), *message = jv_new(a, J_DICT);
	jv_set(a, message, "role", 4, jv_str(a, "user", 4));
	jv_set(a, message, "content", 7, jv_str(a, user, user_n));
	jv_push(a, messages, message);
	jv *vars = jv_new(a, J_DICT);
	jv_set(a, vars, "messages", 8, messages);
	jv_set(a, vars, "add_generation_prompt", 21, jv_bool(a, 1));
	/* Match flashd's default THINK_DEFAULT=1: leave enable_thinking undefined. */
	size_t rendered_n = 0;
	char *rendered = jinja_render(template, a, vars, &rendered_n, error, sizeof error);
	if (!rendered) {
		fprintf(stderr, "chat template render failed: %s\n", error);
		arena_free(a); jinja_free(template); free(user); return 1;
	}
	if (rendered_n > INT_MAX - 16) { fprintf(stderr, "rendered prompt is too large\n"); free(rendered); arena_free(a); jinja_free(template); free(user); return 2; }
	int cap = (int)rendered_n + 16;
	uint32_t *ids = malloc((size_t)cap * sizeof *ids);
	if (!ids) { fprintf(stderr, "token ID allocation failed\n"); free(rendered); arena_free(a); jinja_free(template); free(user); return 1; }
	int count = tok_encode(vocab, rendered, rendered_n, 1, ids, cap);
	if (count < 0) { fprintf(stderr, "tokenizer output exceeded its byte-sized capacity\n"); free(ids); free(rendered); arena_free(a); jinja_free(template); free(user); return 1; }
	if (count < 2053) {
		fprintf(stderr, "prompt has %d tokens; at least 2053 are required to enter selected-block gather\n", count);
		free(ids); free(rendered); arena_free(a); jinja_free(template); free(user); return 3;
	}
	if (count >= 262144) {
		fprintf(stderr, "prompt has %d tokens; model_test supports fewer than 262144\n", count);
		free(ids); free(rendered); arena_free(a); jinja_free(template); free(user); return 3;
	}
	if (write_file(argv[3], rendered, rendered_n) || write_ids(argv[4], ids, count)) {
		free(ids); free(rendered); arena_free(a); jinja_free(template); free(user); return 1;
	}
	printf("{\"prompt_tokens\":%d,\"rendered_bytes\":%zu,\"vocabulary_size\":%d,\"chat_template\":true,\"parse_special\":true,\"add_generation_prompt\":true,\"thinking_default\":true}\n",
	       count, rendered_n, tok_count(vocab));
	free(ids); free(rendered); arena_free(a); jinja_free(template); free(user);
	return 0;
}
