/* aerpoll: count PCIe correctable-error events on the given devices while something runs.
 *
 * The AER correctable error status register is made of sticky bits, not counters. This reads it in a loop and writes back what it shows (write 1 to clear), so every
 * new event sets a bit again and is counted. Events closer together than one poll cycle merge, so the numbers are lower bounds. Needs root (extended config space).
 *
 *   aerpoll <seconds> <bdf> [<bdf> ...]     bdf like 00:03.0 or 09:00.0; a SIGTERM ends it early and prints the same summary
 *
 * Bits: RxErr (0) a receiver error at the physical layer, BadTLP (6) a packet failed its LCRC (NAK, replay), BadDLLP (7) a data-link packet failed its CRC,
 * Rollover (8) the replay counter wrapped (a packet was replayed 4 times), Timeout (12) the replay timer expired (no ACK in time: the ~2 ms stall). */
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop;
static void on_term(int s) { (void)s; stop = 1; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

#define MAXD 32
static struct { char name[16]; int fd; unsigned aer; unsigned long ev[16], polls; } d[MAXD];
static const struct { int bit; const char *name; } bits[] = { { 0, "RxErr" }, { 6, "BadTLP" }, { 7, "BadDLLP" }, { 8, "Rollover" }, { 12, "Timeout" }, { 13, "AdvNonFatal" }, { 14, "CorrInt" }, { 15, "HdrOF" } };

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: aerpoll <seconds> <bdf> [<bdf> ...]\n"); return 2; }
	double secs = atof(argv[1]); int n = 0;
	for (int i = 2; i < argc && n < MAXD; i++) {
		char path[96]; snprintf(path, sizeof path, "/sys/bus/pci/devices/0000:%s/config", argv[i]);
		int fd = open(path, O_RDWR); if (fd < 0) { perror(path); return 1; }
		unsigned off = 0x100, aer = 0;
		for (int k = 0; k < 48 && off >= 0x100; k++) {   /* the extended capability list: find AER (id 1) */
			uint32_t h; if (pread(fd, &h, 4, off) != 4 || h == 0 || h == 0xffffffffu) break;
			if ((h & 0xffff) == 1) { aer = off; break; }
			off = (h >> 20) & 0xfff;
		}
		if (!aer) { fprintf(stderr, "%s: no AER capability, skipped\n", argv[i]); close(fd); continue; }
		snprintf(d[n].name, sizeof d[n].name, "%s", argv[i]); d[n].fd = fd; d[n].aer = aer; n++;
	}
	signal(SIGTERM, on_term); signal(SIGINT, on_term);
	double t0 = now();
	for (int i = 0; i < n; i++) { uint32_t v; if (pread(d[i].fd, &v, 4, d[i].aer + 0x10) == 4 && v) pwrite(d[i].fd, &v, 4, d[i].aer + 0x10); }   /* start from clear */
	while (!stop && now() - t0 < secs) {
		for (int i = 0; i < n; i++) {
			uint32_t v; if (pread(d[i].fd, &v, 4, d[i].aer + 0x10) != 4) continue;
			d[i].polls++;
			if (v) { for (int b = 0; b < 16; b++) if (v & (1u << b)) d[i].ev[b]++; pwrite(d[i].fd, &v, 4, d[i].aer + 0x10); }
		}
	}
	double el = now() - t0;
	printf("aerpoll: %.2f s", el);
	for (int i = 0; i < n; i++) {
		printf("\n%-8s %8lu polls (%.0f/s) ", d[i].name, d[i].polls, d[i].polls / el);
		for (unsigned k = 0; k < sizeof bits / sizeof *bits; k++) printf("%s=%lu ", bits[k].name, d[i].ev[bits[k].bit]);
	}
	printf("\n");
	return 0;
}
