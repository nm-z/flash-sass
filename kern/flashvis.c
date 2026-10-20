/* flashvis: the vision encoder of the chat engine, a small daemon around llama.cpp's libmtmd (MIT) running the
 * Qwen3.8-Flash-Next projector (mmproj: ViT 1152 x 27, patch 16, merge 2, no deepstack) on an available GPU,
 * with a CPU fallback. FLASHVIS_DEVICE selects auto (default), cpu, or a ggml device name such as CUDA6.
 *
 *   flashvis <mmproj.gguf> <text.gguf> <socket> [threads]
 *
 * The text model is opened vocab-only (mtmd wants it for the embedding width and the special tokens). One client at a
 * time; each request is
 *     enc <nbytes>\n<image file bytes: jpeg / png / bmp / gif ...>
 * and the answer
 *     ok <ntok> <nx> <ny> <npos> <dim>\n   then ntok x dim float32 (the embeddings that replace the <|image_pad|> tokens) and
 *     ntok x 3 uint32 (t, y, x: the M-RoPE position of each token relative to the position of the first one)
 * or  err <message>\n. npos is the number of sequence positions the image occupies (max(nx, ny)): the text after it
 * continues at the first position + npos. The image is resized by the projector's own preprocessing (patch 16, merge 2,
 * pixels within the model's image_min / image_max tokens, see kern/vision/preprocessor_config.json). */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int read_full(int fd, void *buf, size_t n) { uint8_t *p = buf; while (n) { ssize_t r = read(fd, p, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; } p += r; n -= (size_t)r; } return 0; }
static int write_full(int fd, const void *buf, size_t n) { const uint8_t *p = buf; while (n) { ssize_t r = write(fd, p, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; } p += r; n -= (size_t)r; } return 0; }
static int read_line(int fd, char *line, size_t cap) {
	size_t n = 0;
	for (;;) { char c; ssize_t r = read(fd, &c, 1); if (r <= 0) return -1; if (c == '\n') break; if (n + 1 < cap) line[n++] = c; }
	line[n] = 0; return 0;
}
static void err(int fd, const char *msg) { char b[300]; int n = snprintf(b, sizeof b, "err %s\n", msg); write_full(fd, b, (size_t)n); }

/* The text engine may already occupy the first GPU. Pick the visible GPU with the
 * most free memory instead of letting mtmd default to device zero. */
static int vision_device(ggml_backend_dev_t *selected) {
	const char *name = getenv("FLASHVIS_DEVICE");
	*selected = NULL;
	if (name && !strcmp(name, "cpu")) return 0;
	if (name && *name && strcmp(name, "auto")) {
		ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
		if (!dev || (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU &&
		             ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU)) {
			fprintf(stderr, "flashvis: FLASHVIS_DEVICE=%s is not an available GPU; use auto, cpu, or a ggml GPU name\n", name);
			return -1;
		}
		*selected = dev;
		return 0;
	}
	size_t best_free = 0;
	for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
		ggml_backend_dev_t dev = ggml_backend_dev_get(i);
		enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
		if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
		size_t available = 0, total = 0;
		ggml_backend_dev_memory(dev, &available, &total);
		if (!*selected || available > best_free) { *selected = dev; best_free = available; }
	}
	if (*selected) fprintf(stderr, "flashvis: auto selected %s (%zu MiB free)\n", ggml_backend_dev_name(*selected), best_free >> 20);
	else fprintf(stderr, "flashvis: no GPU backend available; using CPU\n");
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: flashvis <mmproj.gguf> <text.gguf> <socket> [threads]\n"); return 2; }
	signal(SIGPIPE, SIG_IGN);
	int threads = argc > 4 ? atoi(argv[4]) : 8;
	/* llama_backend_init alone can skip plugin discovery when CPU is already linked. */
#ifdef FLASHVIS_BACKEND_DIR
	ggml_backend_load_all_from_path(FLASHVIS_BACKEND_DIR);
#else
	ggml_backend_load_all();
#endif
	llama_backend_init();
	ggml_backend_dev_t device = NULL;
	if (vision_device(&device)) { llama_backend_free(); return 2; }
	struct llama_model_params mp = llama_model_default_params();
	mp.vocab_only = true;
	struct llama_model *model = llama_model_load_from_file(argv[2], mp);
	if (!model) { fprintf(stderr, "flashvis: cannot load %s (vocab only)\n", argv[2]); return 1; }
	struct mtmd_context_params cp = mtmd_context_params_default();
	cp.use_gpu = device != NULL; cp.device = device;
	cp.n_threads = threads; cp.print_timings = false; cp.warmup = false;
	mtmd_context *ctx = mtmd_init_from_file(argv[1], model, cp);
	if (!ctx && cp.use_gpu) {
		fprintf(stderr, "flashvis: projector initialization on %s failed; retrying CPU\n", ggml_backend_dev_name(device));
		cp.use_gpu = false; cp.device = NULL;
		ctx = mtmd_init_from_file(argv[1], model, cp);
	}
	if (!ctx) { fprintf(stderr, "flashvis: cannot load the projector %s\n", argv[1]); return 1; }
	if (!mtmd_support_vision(ctx)) { fprintf(stderr, "flashvis: %s has no vision encoder\n", argv[1]); return 1; }
	int dim = (int)llama_model_n_embd_inp(model); if (dim <= 0) dim = 2560;   /* a vocab-only model does not know it: the projector writes the text embedding width (clip.vision.projection_dim) */
	fprintf(stderr, "flashvis: ready, backend %s, embedding width %d, %d threads, mrope %d\n", cp.use_gpu ? ggml_backend_dev_name(cp.device) : "CPU", dim, threads, (int)mtmd_decode_use_mrope(ctx));
	int ls = socket(AF_UNIX, SOCK_STREAM, 0);
	struct sockaddr_un sa = { .sun_family = AF_UNIX }; snprintf(sa.sun_path, sizeof sa.sun_path, "%s", argv[3]);
	unlink(argv[3]);
	if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 4)) { perror(argv[3]); return 1; }
	for (;;) {
		int fd = accept(ls, NULL, NULL);
		if (fd < 0) { if (errno == EINTR) continue; perror("accept"); return 1; }
		char line[256];
		while (read_line(fd, line, sizeof line) == 0) {
			size_t nbytes = 0;
			if (sscanf(line, "enc %zu", &nbytes) != 1 || nbytes == 0 || nbytes > ((size_t)64 << 20)) { err(fd, "bad request"); break; }
			uint8_t *buf = malloc(nbytes);
			if (!buf || read_full(fd, buf, nbytes)) { free(buf); break; }
			double t0 = now();
			struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(ctx, buf, nbytes, false, mtmd_helper_init_opt_default());
			mtmd_bitmap *bm = bw.bitmap;
			free(buf);
			if (!bm) { err(fd, "cannot decode the image"); continue; }
			mtmd_input_text text = { .text = mtmd_default_marker(), .text_len = strlen(mtmd_default_marker()), .add_special = false, .parse_special = false };
			mtmd_input_chunks *chunks = mtmd_input_chunks_init();
			const mtmd_bitmap *bms[1] = { bm };
			int rc = mtmd_tokenize(ctx, chunks, &text, bms, 1);
			if (rc) { mtmd_bitmap_free(bm); mtmd_input_chunks_free(chunks); err(fd, "preprocessing failed"); continue; }
			const mtmd_input_chunk *img = NULL;
			for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) { const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i); if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c; }
			if (!img) { mtmd_bitmap_free(bm); mtmd_input_chunks_free(chunks); err(fd, "no image chunk"); continue; }
			size_t ntok = mtmd_input_chunk_get_n_tokens(img); long npos = (long)mtmd_input_chunk_get_n_pos(img);
			const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(img);
			double t1 = now();
			if (mtmd_encode_chunk(ctx, img)) { mtmd_bitmap_free(bm); mtmd_input_chunks_free(chunks); err(fd, "encoding failed"); continue; }
			double t2 = now();
			float *emb = mtmd_get_output_embd(ctx);
			uint32_t *pos = malloc(12 * ntok);
			unsigned nx = 0, ny = 0;
			for (size_t i = 0; i < ntok; i++) { struct mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, 0, i); pos[3 * i] = p.t; pos[3 * i + 1] = p.y; pos[3 * i + 2] = p.x; if (p.x + 1 > nx) nx = p.x + 1; if (p.y + 1 > ny) ny = p.y + 1; }
			char hdr[128]; int hn = snprintf(hdr, sizeof hdr, "ok %zu %u %u %ld %d\n", ntok, nx, ny, npos, dim);
			int bad = write_full(fd, hdr, (size_t)hn) || write_full(fd, emb, 4 * (size_t)dim * ntok) || write_full(fd, pos, 12 * ntok);
			fprintf(stderr, "flashvis: %ux%u image -> %zu tokens (%ux%u grid, %ld positions): preprocess %.0f ms, encode %.3f s\n", mtmd_bitmap_get_nx(bm), mtmd_bitmap_get_ny(bm), ntok, nx, ny, npos, (t1 - t0) * 1e3, t2 - t1);
			free(pos); mtmd_bitmap_free(bm); mtmd_input_chunks_free(chunks);
			if (bad) break;
		}
		close(fd);
	}
}
