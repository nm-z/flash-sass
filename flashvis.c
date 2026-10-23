/* flashvis: the vision encoder of the chat engine, a small daemon around llama.cpp's libmtmd (MIT) running the
 * Qwen3.8-Flash-Next projector (mmproj: ViT 1152 x 27, patch 16, merge 2, no deepstack) on an available GPU,
 * with GPU Flash Attention required. FLASHVIS_DEVICE selects a ggml GPU name such as CUDA6; unset selects the GPU with the most free memory.
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
#include "vision.h"
#include "vis_ggml.h"
#include "vis_supervisor.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cuda.h>
#include <errno.h>
#include <math.h>
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

	if (name && *name && strcmp(name, "auto")) {
		ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
		if (!dev || (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU &&
		             ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU)) {
			fprintf(stderr, "flashvis: FLASHVIS_DEVICE=%s is not an available GPU; use auto or a ggml GPU name\n", name);
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
	else { fprintf(stderr, "flashvis: no GPU backend available; GPU Flash Attention is required\n"); return -1; }
	return 0;
}

/* 2: encoder failure; 1: bad image; -1: closed connection. */
static int encode_image(int fd, mtmd_context *ctx, const uint8_t *buf, size_t nbytes,
                        int dim, ggml_backend_dev_t device, const char **message) {
	const char *backend = ggml_backend_dev_name(device);
	double t0 = now();
	struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(ctx, buf, nbytes, false, mtmd_helper_init_opt_default());
	mtmd_bitmap *bm = bw.bitmap;
	if (!bm) { *message = "cannot decode the image"; return 1; }
	/* Inputs above the model's documented limit are errors, rather than
	 * silently lowering their resolution to fit the encoder. */
	uint64_t pixels = (uint64_t)mtmd_bitmap_get_nx(bm) * mtmd_bitmap_get_ny(bm);
	if (pixels > (uint64_t)FLASH_IMAGE_MAX_TOKENS * 32 * 32) {
		mtmd_bitmap_free(bm);
		*message = "image exceeds the model limit of 16777216 pixels";
		return 1;
	}
	mtmd_input_text text = { .text = mtmd_default_marker(), .text_len = strlen(mtmd_default_marker()), .add_special = false, .parse_special = false };
	mtmd_input_chunks *chunks = mtmd_input_chunks_init();
	const mtmd_bitmap *bms[1] = { bm };
	int result = 1;
	uint32_t *pos = NULL;
	*message = "preprocessing failed";
	if (mtmd_tokenize(ctx, chunks, &text, bms, 1)) goto done;
	const mtmd_input_chunk *img = NULL;
	for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) { const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i); if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c; }
	*message = "no image chunk";
	if (!img) goto done;
	size_t ntok = mtmd_input_chunk_get_n_tokens(img); long npos = (long)mtmd_input_chunk_get_n_pos(img);
	const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(img);
	double t1 = now();
	result = 2; *message = "encoding failed";
	vis_ggml_begin(device);
	int encoded = mtmd_encode_chunk(ctx, img);
	const char *fa_error = vis_ggml_result();
	if (fa_error) { *message = fa_error; goto done; }
	if (encoded) goto done;
	double t2 = now();
	float *emb = mtmd_get_output_embd(ctx);
	*message = "encoder returned invalid embeddings";
	if (!emb || !ntok || ntok > SIZE_MAX / (4 * (size_t)dim) || ntok > SIZE_MAX / 12) goto done;
	/* Check before publishing even the header: flashd caches successful output. */
	for (size_t i = 0; i < ntok * (size_t)dim; i++) if (!isfinite(emb[i])) goto done;
	pos = malloc(12 * ntok);
	*message = "cannot allocate image positions";
	if (!pos) goto done;
	unsigned nx = 0, ny = 0;
	for (size_t i = 0; i < ntok; i++) { struct mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, 0, i); pos[3 * i] = p.t; pos[3 * i + 1] = p.y; pos[3 * i + 2] = p.x; if (p.x + 1 > nx) nx = p.x + 1; if (p.y + 1 > ny) ny = p.y + 1; }
	char hdr[128]; int hn = snprintf(hdr, sizeof hdr, "ok %zu %u %u %ld %d\n", ntok, nx, ny, npos, dim);
	result = (write_full(fd, hdr, (size_t)hn) || write_full(fd, emb, 4 * (size_t)dim * ntok) || write_full(fd, pos, 12 * ntok)) ? -1 : 0;
	fprintf(stderr, "flashvis: %ux%u image -> %zu tokens (%ux%u grid, %ld positions): preprocess %.0f ms, encode %.3f s, backend %s\n", mtmd_bitmap_get_nx(bm), mtmd_bitmap_get_ny(bm), ntok, nx, ny, npos, (t1 - t0) * 1e3, t2 - t1, backend);
done:
	free(pos); mtmd_bitmap_free(bm); mtmd_input_chunks_free(chunks);
	return result;
}

int main(int argc, char **argv) {
	/* Fork+exec supervision starts before the parent initializes CUDA. */
	int fd = vis_supervisor_worker_fd(&argc, argv);
	if (argc < 4) { fprintf(stderr, "usage: flashvis <mmproj.gguf> <text.gguf> <socket> [threads]\n"); return 2; }
	if (fd < 0) return vis_supervisor_main(argc, argv);
	signal(SIGPIPE, SIG_IGN);
	int threads = argc > 4 ? atoi(argv[4]) : 8;
	/* The driver FA launch synchronizes its context. Keep graph execution
	 * explicit; CUDA stream capture cannot contain that synchronization. */
	setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
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
	cp.use_gpu = true; cp.device = device;
	cp.n_threads = threads; cp.print_timings = false; cp.warmup = false;
	/* Match the model's pixel limits on the required GPU path. */
	cp.image_min_tokens = FLASH_IMAGE_MIN_TOKENS;
	cp.image_max_tokens = FLASH_IMAGE_MAX_TOKENS;
	cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
	cp.cb_eval = vis_ggml_observe;
	mtmd_context *ctx = mtmd_init_from_file(argv[1], model, cp);
	if (!ctx) { fprintf(stderr, "flashvis: cannot load the projector %s\n", argv[1]); return 1; }
	if (!mtmd_support_vision(ctx)) { fprintf(stderr, "flashvis: %s has no vision encoder\n", argv[1]); return 1; }
	int dim = (int)llama_model_n_embd_inp(model); if (dim <= 0) dim = 2560;   /* a vocab-only model does not know it: the projector writes the text embedding width (clip.vision.projection_dim) */
	fprintf(stderr, "flashvis: ready, backend %s, embedding width %d, %d threads, mrope %d\n", ggml_backend_dev_name(device), dim, threads, (int)mtmd_decode_use_mrope(ctx));
	if (write_full(fd, "ready\n", 6)) return 1;
	char line[256];
	while (read_line(fd, line, sizeof line) == 0) {
			size_t nbytes = 0;
			if (sscanf(line, "enc %zu", &nbytes) != 1 || nbytes == 0 || nbytes > ((size_t)64 << 20)) { err(fd, "bad request"); break; }
			uint8_t *buf = malloc(nbytes);
			if (!buf || read_full(fd, buf, nbytes)) { free(buf); break; }
			const char *message = NULL;
			int rc = encode_image(fd, ctx, buf, nbytes, dim, device, &message);
			free(buf);
			if (rc < 0) break;
			if (rc) err(fd, message);
			/* A poisoned CUDA context is discarded after returning this
			 * error. The supervisor starts a new GPU worker for the next
			 * request; this image is not retried. */
			if (rc == 2 && cuCtxSynchronize() != CUDA_SUCCESS) return 1;
	}
	close(fd);
	mtmd_free(ctx); llama_model_free(model); llama_backend_free();
	return 0;
}
