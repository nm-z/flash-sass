/*
 * ggufhdr: print a GGUF header (metadata and tensor table) from a file or a
 * truncated head of one. Sizes come from the gaps between tensor offsets, so
 * bits per weight are reported per tensor without knowing the type.
 *
 *   ggufhdr FILE            everything
 *   ggufhdr FILE kv         metadata only
 *   ggufhdr FILE tensors    tensor table only (tab separated)
 *   ggufhdr FILE summary    bytes per type and per block
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *f;
static int eof;

static uint64_t rd(int n) {
	uint64_t v = 0;
	if (fread(&v, n, 1, f) != 1) eof = 1;
	return v;
}
static char *rds(void) {
	uint64_t n = rd(8);
	if (eof || n > (1u << 24)) { eof = 1; return strdup("?"); }
	char *s = malloc(n + 1);
	if (fread(s, 1, n, f) != n) eof = 1;
	s[n] = 0;
	return s;
}
static int scalar_size(uint32_t t) {
	switch (t) {
	case 0: case 1: case 7: return 1;
	case 2: case 3: return 2;
	case 4: case 5: case 6: return 4;
	case 10: case 11: case 12: return 8;
	default: return 0;
	}
}
static void print_scalar(uint32_t t) {
	int n = scalar_size(t);
	uint64_t v = rd(n);
	switch (t) {
	case 1: printf("%d", (int8_t)v); break;
	case 3: printf("%d", (int16_t)v); break;
	case 5: printf("%d", (int32_t)v); break;
	case 11: printf("%" PRId64, (int64_t)v); break;
	case 6: { float x; memcpy(&x, &v, 4); printf("%g", x); break; }
	case 12: { double x; memcpy(&x, &v, 8); printf("%g", x); break; }
	case 7: printf(v ? "true" : "false"); break;
	default: printf("%" PRIu64, v);
	}
}
static void print_value(uint32_t t, int show) {
	if (t == 8) {
		char *s = rds();
		if (show) {
			if (strlen(s) > 120) { s[117] = 0; printf("\"%s...\"", s); } else printf("\"%s\"", s);
		}
		free(s);
	} else if (t == 9) {
		uint32_t et = (uint32_t)rd(4);
		uint64_t n = rd(8);
		if (show) printf("[type %u x %" PRIu64 "]", et, n);
		int all = n <= 64;
		if (show) printf(all ? " [" : " [");
		for (uint64_t i = 0; i < n && !eof; i++) {
			int s = show && (all || i < 8);
			if (et == 8 || et == 9) print_value(et, s);
			else print_scalar(et), (void)0;
			if (!s && et != 8 && et != 9) {
				/* printed nothing but consumed the value */
			}
			if (s && i + 1 < n) printf(all || i < 7 ? ", " : ", ...");
		}
		if (show) printf("]");
	} else print_scalar(t), (void)0;
}
/* print_scalar always prints; wrap it so hidden array elements consume silently */
static void consume_scalar(uint32_t t) { rd(scalar_size(t)); }

static const char *tname(uint32_t t) {
	static const char *n[] = { "F32", "F16", "Q4_0", "Q4_1", "?4", "?5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
		"Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
		"I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "?31", "?32", "?33", "TQ1_0", "TQ2_0", "?36", "?37", "?38", "MXFP4" };
	static char buf[16];
	if (t < sizeof n / sizeof n[0]) return n[t];
	snprintf(buf, sizeof buf, "type%u", t);
	return buf;
}

struct tensor { char *name; uint32_t nd; uint64_t dims[4]; uint32_t type; uint64_t off; uint64_t elems; uint64_t bytes; };

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: ggufhdr FILE [kv|tensors|summary]\n"); return 2; }
	const char *mode = argc > 2 ? argv[2] : "all";
	int show_kv = !strcmp(mode, "all") || !strcmp(mode, "kv");
	int show_t = !strcmp(mode, "all") || !strcmp(mode, "tensors");
	int show_s = !strcmp(mode, "all") || !strcmp(mode, "summary");
	f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	char magic[4];
	if (fread(magic, 4, 1, f) != 1 || memcmp(magic, "GGUF", 4)) { fprintf(stderr, "not a GGUF file\n"); return 1; }
	uint32_t ver = (uint32_t)rd(4);
	uint64_t nt = rd(8), nkv = rd(8);
	printf("gguf v%u tensors %" PRIu64 " kv %" PRIu64 "\n", ver, nt, nkv);
	uint64_t align = 32;
	for (uint64_t i = 0; i < nkv && !eof; i++) {
		char *k = rds();
		uint32_t t = (uint32_t)rd(4);
		if (show_kv) printf("kv %s = ", k);
		if (t == 9) {
			uint32_t et = (uint32_t)rd(4);
			uint64_t n = rd(8);
			int all = n <= 64;
			if (show_kv) printf("[%s x %" PRIu64 "] [", tname(et) + 0 * et, n);
			for (uint64_t j = 0; j < n && !eof; j++) {
				int s = show_kv && (all || j < 6);
				if (et == 8) { char *v = rds(); if (s) printf("\"%s\"", v); free(v); }
				else if (s) print_scalar(et);
				else consume_scalar(et);
				if (s && j + 1 < n) printf(all || j < 5 ? ", " : ", ...");
			}
			if (show_kv) printf("]\n");
		} else {
			if (t == 8) { char *v = rds(); if (show_kv) printf("\"%s\"\n", v); free(v); }
			else {
				uint64_t raw = 0;
				long pos = ftell(f);
				if (show_kv) { print_scalar(t); printf("\n"); } else consume_scalar(t);
				if (!strcmp(k, "general.alignment")) { fseek(f, pos, SEEK_SET); raw = rd(scalar_size(t)); align = raw; }
			}
		}
		free(k);
	}
	if (eof) { fprintf(stderr, "truncated inside metadata\n"); return 1; }
	struct tensor *ts = calloc(nt, sizeof *ts);
	for (uint64_t i = 0; i < nt && !eof; i++) {
		ts[i].name = rds();
		ts[i].nd = (uint32_t)rd(4);
		ts[i].elems = 1;
		for (uint32_t d = 0; d < ts[i].nd && d < 4; d++) { ts[i].dims[d] = rd(8); ts[i].elems *= ts[i].dims[d]; }
		ts[i].type = (uint32_t)rd(4);
		ts[i].off = rd(8);
	}
	if (eof) { fprintf(stderr, "truncated inside tensor table\n"); return 1; }
	long hdr_end = ftell(f);
	uint64_t data_start = ((uint64_t)hdr_end + align - 1) / align * align;
	fseek(f, 0, SEEK_END);
	uint64_t file_len = (uint64_t)ftell(f);
	/* sizes from offset gaps; tensors are stored in offset order but sort defensively */
	for (uint64_t i = 0; i < nt; i++) {
		uint64_t next = UINT64_MAX;
		for (uint64_t j = 0; j < nt; j++) if (ts[j].off > ts[i].off && ts[j].off < next) next = ts[j].off;
		ts[i].bytes = next == UINT64_MAX ? 0 : next - ts[i].off;
	}
	printf("header %ld bytes, data at %" PRIu64 ", alignment %" PRIu64 ", file %" PRIu64 " bytes%s\n", hdr_end, data_start, align, file_len,
	       file_len < data_start ? " (truncated head)" : "");
	if (show_t) {
		printf("name\ttype\tdims\toffset\tbytes\tbpw\n");
		for (uint64_t i = 0; i < nt; i++) {
			printf("%s\t%s\t", ts[i].name, tname(ts[i].type));
			for (uint32_t d = 0; d < ts[i].nd; d++) printf("%" PRIu64 "%s", ts[i].dims[d], d + 1 < ts[i].nd ? "x" : "");
			printf("\t%" PRIu64 "\t%" PRIu64 "\t", ts[i].off, ts[i].bytes);
			if (ts[i].bytes) printf("%.4f\n", 8.0 * (double)ts[i].bytes / (double)ts[i].elems); else printf("last\n");
		}
	}
	if (show_s) {
		uint64_t by_type[64] = { 0 }, el_type[64] = { 0 }, n_type[64] = { 0 };
		uint64_t blk[128] = { 0 }, other = 0;
		for (uint64_t i = 0; i < nt; i++) {
			uint32_t t = ts[i].type < 64 ? ts[i].type : 63;
			by_type[t] += ts[i].bytes; el_type[t] += ts[i].elems; n_type[t]++;
			int b;
			if (sscanf(ts[i].name, "blk.%d.", &b) == 1 && b >= 0 && b < 128) blk[b] += ts[i].bytes; else other += ts[i].bytes;
		}
		printf("type\ttensors\telements\tbytes\tbpw\n");
		for (int t = 0; t < 64; t++) if (n_type[t])
			printf("%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%.4f\n", tname(t), n_type[t], el_type[t], by_type[t],
			       by_type[t] ? 8.0 * (double)by_type[t] / (double)el_type[t] : 0.0);
		printf("block\tbytes\n");
		for (int b = 0; b < 128; b++) if (blk[b]) printf("blk.%d\t%" PRIu64 "\n", b, blk[b]);
		printf("other\t%" PRIu64 "\n", other);
	}
	return 0;
}
