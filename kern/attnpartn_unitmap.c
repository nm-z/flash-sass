/* attnpartn_unitmap.c: offline check of the attnpartn unit-id remap (no GPU): cc -O2 -o unitmap attnpartn_unitmap.c && ./unitmap
 * Both decodes must be bijections onto (kv, hh, c); the table shows how many CTAs the units with keys land on and how many run back to back
 * (a CTA runs units b and b + 16). The output slots depend only on (kv, hh, c), so the results do not change, only which CTA computes which unit. */
#include <stdio.h>
#include <string.h>
static void old_(int b, int *kv, int *hh, int *c) { *c = b & 7; *hh = (b >> 3) & 1; *kv = b >> 4; }
static void new_(int b, int *kv, int *hh, int *c) { *c = b >> 2; *hh = (b >> 1) & 1; *kv = b & 1; }
int main(void) {
	void (*dec[2])(int, int *, int *, int *) = { old_, new_ }; const char *nm[2] = { "old", "new" };
	for (int m = 0; m < 2; m++) {   /* a bijection of the 32 unit ids onto (kv, hh, c) */
		int seen[2][2][8]; memset(seen, 0, sizeof seen); int ok = 1;
		for (int b = 0; b < 32; b++) { int kv, hh, c; dec[m](b, &kv, &hh, &c); if (kv > 1 || hh > 1 || c > 7 || seen[kv][hh][c]++) ok = 0; }
		for (int kv = 0; kv < 2; kv++) for (int hh = 0; hh < 2; hh++) for (int c = 0; c < 8; c++) if (seen[kv][hh][c] != 1) ok = 0;
		printf("%s decode: %s\n", nm[m], ok ? "bijection onto (kv, hh, c)" : "NOT a bijection");
	}
	int ns[] = { 500, 1024, 2048, 2049, 2050, 2051, 4096, 8192 };
	printf("%6s | %-34s | %-34s\n", "n keys", "old: busy units / CTAs / deepest CTA", "new: busy units / CTAs / deepest CTA");
	for (unsigned i = 0; i < sizeof ns / sizeof *ns; i++) {
		int n = ns[i], CK = ((n + 8191) / 8192) * 1024; char col[2][64];
		for (int m = 0; m < 2; m++) {
			int depth[16] = { 0 }, busy = 0, ctas = 0, deep = 0;
			for (int b = 0; b < 32; b++) { int kv, hh, c; dec[m](b, &kv, &hh, &c); if (c * CK < n) { busy++; depth[b % 16]++; } }   /* the CTA runs units b and b + 16 one after the other */
			for (int k = 0; k < 16; k++) { if (depth[k]) ctas++; if (depth[k] > deep) deep = depth[k]; }
			snprintf(col[m], sizeof col[m], "%2d units / %2d CTAs / %d sequential", busy, ctas, deep);
		}
		printf("%6d | %-34s | %-34s\n", n, col[0], col[1]);
	}
	return 0;
}
