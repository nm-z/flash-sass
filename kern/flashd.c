/* flashd: the HTTP front-end of the Flash-Next engine, speaking the subset of llama-server's API
 * that https://nm-z.com/chat/ uses:
 *
 *   GET  /v1/models            {"data":[{"id": name}]}
 *   GET  /props               {"default_generation_settings":{"n_ctx": positions}, "chat_template": ...}
 *   GET  /health
 *   POST /v1/chat/completions  {"messages":[...], "stream":bool, "max_tokens":n,
 *                               "chat_template_kwargs":{"enable_thinking":bool}}
 *
 * The prompt is the GGUF's own chat template (jinja.c) over the request's messages, tokenized
 * with the GGUF's vocabulary (tok.c), and run on the engine (flash, FLASH_SOCK) greedily until
 * <|im_end|>/EOS or max_tokens. The reply streams as server-sent events: reasoning tokens as
 * delta.reasoning_content up to </think>, the answer as delta.content, and the final event
 * carries timings (prompt_per_second, predicted_per_second). One request runs at a time; the
 * others wait on the engine lock (GET endpoints never wait).
 *
 *   flashd <shard-1.gguf> <flash-socket> <host:port>      env FLASHD_NAME, FLASHD_THINKING (0: off by default) */
#define _GNU_SOURCE
#include "jinja.h"
#include "tok.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

static tok_vocab *V;
static jinja *TPL;
static const char *SOCK, *NAME = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-sass";
static int POSITIONS = 2048, THINK_DEFAULT = 1;
// Output budgets in thinking mode: up to 262,144 tokens of reasoning and 131,072 of final answer (the context bounds both).
static long REASONING_MAX = 262144, ANSWER_MAX = 131072;
static double DEF_TEMP = 0.0, DEF_TOPP = 1.0; static int DEF_TOPK = 0;   /* general.sampling.* from the GGUF; temperature 0 = greedy */
static uint32_t ID_END, ID_THINK_END;
static pthread_mutex_t ENGINE = PTHREAD_MUTEX_INITIALIZER;

/* ---- output ---- */
static int wr(int fd, const char *s, size_t n) {
	while (n) { ssize_t k = send(fd, s, n, MSG_NOSIGNAL); if (k <= 0) return -1; s += k; n -= (size_t)k; }
	return 0;
}
static int wrs(int fd, const char *s) { return wr(fd, s, strlen(s)); }
static int chunk(int fd, const char *s, size_t n) { char h[24]; int k = snprintf(h, sizeof h, "%zx\r\n", n); return wr(fd, h, (size_t)k) || wr(fd, s, n) || wr(fd, "\r\n", 2); }
static void reply(int fd, int status, const char *ctype, const char *body, size_t n) {
	char h[256]; const char *r = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : status == 413 ? "Payload Too Large" : "Internal Server Error";
	int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nCache-Control: no-store\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, r, ctype, n);
	wr(fd, h, (size_t)k); wr(fd, body, n);
}
static void error(int fd, int status, const char *msg) {
	sbuf b = {0}; sb_puts(&b, "{\"error\":{\"message\":"); sb_json_str(&b, msg, strlen(msg)); sb_printf(&b, ",\"code\":%d}}", status);
	reply(fd, status, "application/json", b.s, b.n); sb_free(&b);
}

/* ---- the engine socket ---- */
static FILE *engine_open(FILE **out) {
	int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un sa = { .sun_family = AF_UNIX };
	snprintf(sa.sun_path, sizeof sa.sun_path, "%s", SOCK);
	if (s < 0 || connect(s, (struct sockaddr *)&sa, sizeof sa)) { if (s >= 0) close(s); return NULL; }
	*out = fdopen(dup(s), "w"); return fdopen(s, "r");
}

/* ---- one reply being decoded ---- */
typedef struct {
	int fd, stream, in_think, gone, started_content, started_reasoning;
	unsigned char pend[8]; int npend;
	sbuf content, reasoning;
	long created;
	long n_reason, n_answer; int cap_hit;   /* tokens so far in each part; set when one reaches its budget */
} gen;

static void emit_delta(gen *g, const char *field, const char *s, size_t n) {
	if (!n) return;
	if (g->stream && !g->gone) {
		sbuf b = {0};
		sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"%s\":", field); sb_json_str(&b, s, n);
		sb_printf(&b, "},\"finish_reason\":null}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\"}\n\n", g->created, NAME);
		if (chunk(g->fd, b.s, b.n)) g->gone = 1;
		sb_free(&b);
	}
	sbuf *acc = field[0] == 'r' ? &g->reasoning : &g->content;
	sb_put(acc, s, n);
}
/* text from the model: drop the whitespace that opens each part, hold back an incomplete UTF-8 tail */
static void emit_text(gen *g, const char *s, size_t n) {
	unsigned char buf[16 + 4096]; size_t k = 0;
	if (g->npend) memcpy(buf, g->pend, (size_t)g->npend), k = (size_t)g->npend;
	if (n > 4096) n = 4096;
	memcpy(buf + k, s, n); k += n;
	size_t keep = k;   /* an incomplete trailing character waits for its next bytes */
	for (size_t back = 1; back <= 3 && back <= k; back++) {
		unsigned char c = buf[k - back];
		if ((c & 0xc0) == 0x80) continue;
		size_t need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
		if (need > back) keep = k - back;
		break;
	}
	g->npend = (int)(k - keep); memcpy(g->pend, buf + keep, (size_t)g->npend);
	const char *p = (const char *)buf; size_t m = keep;
	int *started = g->in_think ? &g->started_reasoning : &g->started_content;
	while (!*started && m && (*p == '\n' || *p == ' ' || *p == '\r' || *p == '\t')) { p++; m--; }
	if (m) *started = 1;
	emit_delta(g, g->in_think ? "reasoning_content" : "content", p, m);
}
static void on_token(gen *g, uint32_t id) {
	if (id == ID_THINK_END) { g->in_think = 0; return; }
	if (tok_is_special(V, id)) return;
	if (g->in_think ? ++g->n_reason > REASONING_MAX : ++g->n_answer > ANSWER_MAX) { g->cap_hit = 1; return; }
	char piece[256]; int n = tok_piece(V, id, piece, sizeof piece);
	if (n > 0) emit_text(g, piece, (size_t)n);
}

/* ---- images: the vision encoder daemon (flashvis) and its cache ----
   A request's image parts (data: URLs) are decoded, run through flashvis (CPU, seconds per image) and the embeddings are kept in files
   under IMGDIR keyed by a hash of the image bytes, so the images of earlier turns, which the page sends again with every question, cost nothing the
   second time. The engine reads those files (the `img` command). */
static const char *VIS_SOCK;               /* FLASHVIS_SOCK: the encoder; NULL: no images */
static char IMGDIR[200] = "/var/tmp/flash-img";
static int ID_IMGPAD = -1;
static unsigned IMG_ROWS = 0;              /* image tokens one response can hold (the engine's FLASH_MAXIMG), from `info` */
static int b64val(int c) { return c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1; }
/* base64 (standard or URL alphabet, padding optional) -> bytes; returns the length or -1 */
static long b64decode(const char *s, size_t n, unsigned char *out) {
	size_t o = 0; unsigned acc = 0; int bits = 0;
	for (size_t i = 0; i < n; i++) {
		if (s[i] == '=' || s[i] == '\n' || s[i] == '\r' || s[i] == ' ') continue;
		int v = b64val((unsigned char)s[i]); if (v < 0) return -1;
		acc = (acc << 6) | (unsigned)v; bits += 6;
		if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)(acc >> bits); acc &= (1u << bits) - 1; }
	}
	return (long)o;
}
static unsigned long long fnv64(const unsigned char *p, size_t n) { unsigned long long h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h ^ ((unsigned long long)n << 1); }
/* the cache file of an image (written when missing): its header gives the token count and the positions the image takes.
 * Returns 0, or an error text. */
typedef struct { char path[240]; unsigned ntok, npos; } imgrec;
static const char *vis_image(const unsigned char *bytes, size_t n, imgrec *r) {
	if (!VIS_SOCK) return "this engine was started without a vision encoder";
	char *dir = IMGDIR; mkdir(dir, 0755);
	snprintf(r->path, sizeof r->path, "%s/%016llx.bin", dir, fnv64(bytes, n));
	FILE *f = fopen(r->path, "rb"); uint32_t hd[5];
	if (f) { int ok = fread(hd, 4, 5, f) == 5; fclose(f); if (ok) { r->ntok = hd[0]; r->npos = hd[3]; utime(r->path, NULL); return NULL; } }
	int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un sa = { .sun_family = AF_UNIX }; snprintf(sa.sun_path, sizeof sa.sun_path, "%s", VIS_SOCK);
	if (s < 0 || connect(s, (struct sockaddr *)&sa, sizeof sa)) { if (s >= 0) close(s); return "the vision encoder is not running"; }
	char head[64]; int hn = snprintf(head, sizeof head, "enc %zu\n", n);
	if (wr(s, head, (size_t)hn) || wr(s, (const char *)bytes, n)) { close(s); return "the vision encoder closed the connection"; }
	char line[160]; size_t ll = 0;
	while (ll + 1 < sizeof line) { char c; if (recv(s, &c, 1, 0) != 1) { close(s); return "the vision encoder closed the connection"; } if (c == '\n') break; line[ll++] = c; }
	line[ll] = 0;
	unsigned ntok, nx, ny, dim; long npos;
	if (sscanf(line, "ok %u %u %u %ld %u", &ntok, &nx, &ny, &npos, &dim) != 5) { close(s); static char em[200]; snprintf(em, sizeof em, "the vision encoder: %.150s", line); return em; }
	size_t body = (size_t)ntok * dim * 4 + (size_t)ntok * 12; unsigned char *buf = malloc(body); size_t got = 0;
	while (got < body) { ssize_t k = recv(s, buf + got, body - got, 0); if (k <= 0) break; got += (size_t)k; }
	close(s);
	if (got != body || dim != 2560) { free(buf); return "the vision encoder sent a short answer"; }
	char tmp[260]; snprintf(tmp, sizeof tmp, "%s.%d.tmp", r->path, (int)getpid());
	f = fopen(tmp, "wb"); uint32_t h5[5] = { ntok, nx, ny, (uint32_t)npos, dim };
	int ok = f && fwrite(h5, 4, 5, f) == 5 && fwrite(buf, 1, body, f) == body; if (f) fclose(f);
	free(buf);
	if (!ok || rename(tmp, r->path)) { unlink(tmp); return "cannot write the image cache"; }
	r->ntok = ntok; r->npos = (unsigned)npos;
	/* keep the newest 24 files */
	{ DIR *d = opendir(dir); struct dirent *e; struct { time_t t; char name[64]; } keep[64]; int nk = 0;
	  while (d && (e = readdir(d))) { if (e->d_name[0] == '.' || strstr(e->d_name, ".tmp")) continue; struct stat st; char p[300]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name); if (stat(p, &st)) continue; if (nk < 64) { keep[nk].t = st.st_mtime; snprintf(keep[nk].name, sizeof keep[nk].name, "%s", e->d_name); nk++; } }
	  if (d) closedir(d);
	  for (int guard = 0; nk > 24 && guard < 64; guard++) { int old = 0; for (int i = 1; i < nk; i++) if (keep[i].t < keep[old].t) old = i; char p[300]; snprintf(p, sizeof p, "%s/%s", dir, keep[old].name); unlink(p); keep[old] = keep[--nk]; } }
	return NULL;
}

/* ---- POST /v1/chat/completions ---- */
static void chat(int fd, const char *body, size_t bn) {
	arena *a = arena_new(); const char *je = NULL;
	jv *req = json_parse(a, body, bn, &je);
	if (!req || req->k != J_DICT) { error(fd, 400, "the body is not a JSON object"); arena_free(a); return; }
	jv *msgs = jv_getz(req, "messages");
	if (!msgs || msgs->k != J_LIST || !msgs->l.n) { error(fd, 400, "messages must be a non-empty list"); arena_free(a); return; }
	/* The sampler has no penalties: the thinking-mode settings presence_penalty 0.0 and repetition_penalty 1.0 are what it does, so
	   those values are accepted and any other is refused rather than silently ignored. */
	{
		static const struct { const char *key; double fixed; } pen[] = { { "presence_penalty", 0.0 }, { "frequency_penalty", 0.0 }, { "repetition_penalty", 1.0 }, { "repeat_penalty", 1.0 } };
		for (size_t i = 0; i < sizeof pen / sizeof pen[0]; i++) {
			jv *x = jv_getz(req, pen[i].key); double v = x && x->k == J_NUM ? x->d : x && x->k == J_INT ? (double)x->i : pen[i].fixed;
			if (v != pen[i].fixed) { char m[160]; snprintf(m, sizeof m, "%s is fixed at %g (the engine has no penalties)", pen[i].key, pen[i].fixed); error(fd, 400, m); arena_free(a); return; }
		}
	}
	/* Image parts: data: URLs, decoded and encoded by flashvis (cached by hash); the template only needs to see that there is an image,
	   so the data is dropped from the variables before rendering. */
	imgrec imgs[8]; int nim = 0;
	for (int i = 0; i < msgs->l.n; i++) {
		jv *c = jv_getz(msgs->l.a[i], "content");
		if (!c || c->k != J_LIST) continue;
		for (int j = 0; j < c->l.n; j++) {
			jv *part = c->l.a[j], *ty = jv_getz(part, "type");
			if (!(ty && ty->k == J_STR && ((ty->s.n == 9 && !memcmp(ty->s.s, "image_url", 9)) || (ty->s.n == 5 && !memcmp(ty->s.s, "image", 5))))) continue;
			if (nim >= 8) { error(fd, 400, "at most 8 images per request"); arena_free(a); return; }
			jv *iu = jv_getz(part, "image_url"); const char *url = NULL; size_t un = 0;
			if (iu && iu->k == J_STR) { url = iu->s.s; un = iu->s.n; } else if (iu && iu->k == J_DICT) { jv *u = jv_getz(iu, "url"); if (u && u->k == J_STR) { url = u->s.s; un = u->s.n; } }
			const char *comma = url && un > 5 && !memcmp(url, "data:", 5) ? memchr(url, ',', un < 200 ? un : 200) : NULL;
			if (!comma || !memmem(url, (size_t)(comma - url), ";base64", 7)) { error(fd, 400, "an image must be a data: URL with base64 data"); arena_free(a); return; }
			unsigned char *raw = malloc(un); long rn = b64decode(comma + 1, un - (size_t)(comma + 1 - url), raw);
			if (rn <= 0) { free(raw); error(fd, 400, "the image is not valid base64"); arena_free(a); return; }
			const char *ie = vis_image(raw, (size_t)rn, &imgs[nim]); free(raw);
			if (ie) { error(fd, 503, ie); arena_free(a); return; }
			if (iu->k == J_DICT) jv_set(a, iu, "url", 3, jv_str(a, "", 0)); else jv_set(a, part, "image_url", 9, jv_str(a, "", 0));
			nim++;
		}
	}
	unsigned img_tokens = 0; for (int i = 0; i < nim; i++) img_tokens += imgs[i].ntok;
	if (img_tokens > IMG_ROWS) { char m[160]; snprintf(m, sizeof m, "the images take %u tokens; one response holds %u", img_tokens, IMG_ROWS); error(fd, 400, m); arena_free(a); return; }
	jv *vars = jv_new(a, J_DICT);
	jv_set(a, vars, "messages", 8, msgs);
	jv_set(a, vars, "add_generation_prompt", 21, jv_bool(a, 1));
	jv *tools = jv_getz(req, "tools"); if (tools && tools->k == J_LIST && tools->l.n) jv_set(a, vars, "tools", 5, tools);
	jv *kw = jv_getz(req, "chat_template_kwargs");
	jv *think = kw ? jv_getz(kw, "enable_thinking") : NULL;
	if (think && think->k == J_BOOL) jv_set(a, vars, "enable_thinking", 15, think);
	else if (!THINK_DEFAULT) jv_set(a, vars, "enable_thinking", 15, jv_bool(a, 0));
	jv *eff = kw ? jv_getz(kw, "reasoning_effort") : NULL;
	if (!eff) eff = jv_getz(req, "reasoning_effort");   /* the OpenAI spelling */
	if (eff && eff->k == J_STR) jv_set(a, vars, "reasoning_effort", 16, eff);
	double temp = DEF_TEMP, topp = DEF_TOPP; long topk = DEF_TOPK; unsigned long long seed = 0; int have_seed = 0;
	{ jv *x;
	  if ((x = jv_getz(req, "temperature")) && (x->k == J_NUM || x->k == J_INT)) temp = x->k == J_NUM ? x->d : (double)x->i;
	  if ((x = jv_getz(req, "top_p")) && (x->k == J_NUM || x->k == J_INT)) topp = x->k == J_NUM ? x->d : (double)x->i;
	  if ((x = jv_getz(req, "top_k")) && x->k == J_INT) topk = (long)x->i;
	  if ((x = jv_getz(req, "seed")) && x->k == J_INT && x->i >= 0) { seed = (unsigned long long)x->i; have_seed = 1; } }
	if (!have_seed) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); seed = (unsigned long long)ts.tv_nsec * 1000003ull ^ (unsigned long long)ts.tv_sec * 0x9e3779b97f4a7c15ull ^ (unsigned long long)(uintptr_t)body; }
	int stream = 0; jv *sv = jv_getz(req, "stream"); if (sv && sv->k == J_BOOL) stream = sv->b;
	long max_tokens = 512; jv *mt = jv_getz(req, "max_tokens"); if (!mt) mt = jv_getz(req, "max_completion_tokens");
	if (mt && mt->k == J_INT && mt->i > 0) max_tokens = (long)mt->i;

	char err[512]; size_t pn;
	char *prompt = jinja_render(TPL, a, vars, &pn, err, sizeof err);
	if (!prompt) { error(fd, 400, err); arena_free(a); return; }

	pthread_mutex_lock(&ENGINE);
	uint32_t *ids = malloc(4 * (pn + 16));
	int np = tok_encode(V, prompt, pn, 1, ids, (int)pn + 16);
	if (np < 1 || np + 1 >= POSITIONS) {
		pthread_mutex_unlock(&ENGINE); free(ids); free(prompt); arena_free(a);
		error(fd, 400, np < 1 ? "empty prompt" : "the prompt is longer than the engine's context"); return;
	}
	/* the engine returns n_new + 1 tokens (the one predicted at the last prompt position, then n_new more) */
	long n_new = max_tokens - 1 < POSITIONS - np ? max_tokens - 1 : POSITIONS - np;
	/* the prefix cache: the next turn of this conversation will start with this prompt up to the assistant turn it appends, so
	   render the conversation plus an assistant message and keep the tokens the two share (at least one prompt token must run) */
	int snap = 0;
	{
		jv *l2 = jv_new(a, J_LIST), *am = jv_new(a, J_DICT);
		for (int i = 0; i < msgs->l.n; i++) jv_push(a, l2, msgs->l.a[i]);
		jv_set(a, am, "role", 4, jv_str(a, "assistant", 9)); jv_set(a, am, "content", 7, jv_str(a, "x", 1)); jv_push(a, l2, am);
		jv_set(a, vars, "messages", 8, l2);
		size_t pn2; char *p2 = jinja_render(TPL, a, vars, &pn2, err, sizeof err);
		jv_set(a, vars, "messages", 8, msgs);
		if (p2) {
			uint32_t *ids2 = malloc(4 * (pn2 + 16)); int n2 = tok_encode(V, p2, pn2, 1, ids2, (int)pn2 + 16);
			int l = 0; while (l < n2 && l < np && ids2[l] == ids[l]) l++;
			snap = l < np - 1 ? l : np - 1;
			if (snap < 16) snap = 0;
			free(ids2); free(p2);
		}
	}
	/* images: every <|image_pad|> placeholder becomes one token per embedding row (the engine swaps in the embeddings), and a snapshot stops
	   before the first image */
	int istart[8] = { 0 };
	if (nim) {
		int found = 0, first_img = -1; for (int i = 0; i < np; i++) if ((int)ids[i] == ID_IMGPAD) found++;
		if (found != nim || ID_IMGPAD < 0) { pthread_mutex_unlock(&ENGINE); free(ids); free(prompt); arena_free(a); error(fd, 400, "the chat template did not place one image token per image"); return; }
		uint32_t *ex = malloc(4 * ((size_t)np + img_tokens + 16)); int k = 0, im = 0;
		for (int i = 0; i < np; i++) {
			if ((int)ids[i] == ID_IMGPAD) { if (first_img < 0) first_img = i; istart[im] = k; for (unsigned q = 0; q < imgs[im].ntok; q++) ex[k++] = ids[i]; im++; }
			else ex[k++] = ids[i];
		}
		free(ids); ids = ex; np = k;
		if (snap > first_img) snap = first_img < 16 ? 0 : first_img;
		if (np + 1 >= POSITIONS) { pthread_mutex_unlock(&ENGINE); free(ids); free(prompt); arena_free(a); error(fd, 400, "the prompt is longer than the engine's context"); return; }
		n_new = max_tokens - 1 < POSITIONS - np ? max_tokens - 1 : POSITIONS - np;
	}
	FILE *out = NULL, *in = engine_open(&out);
	if (!in) { pthread_mutex_unlock(&ENGINE); free(ids); free(prompt); arena_free(a); error(fd, 500, "the engine is not running"); return; }
	fprintf(out, "stop %u %u\n", ID_END, tok_eos(V) == ID_END ? 0xffffffffu : tok_eos(V));
	fprintf(out, "sample %llu %.6g %ld %.6g\n", seed, temp, topk <= 0 ? 32 : topk, topp);
	for (int i = 0; i < nim; i++) fprintf(out, "img %s %d\n", imgs[i].path, istart[i]);
	if (snap) fprintf(out, "snap %d\n", snap);
	fprintf(out, "chat %ld", n_new); for (int i = 0; i < np; i++) fprintf(out, " %u", ids[i]); fputc('\n', out); fflush(out);
	free(ids);

	gen g = { .fd = fd, .stream = stream, .created = (long)time(NULL) };
	g.in_think = pn >= 8 && !memcmp(prompt + pn - 8, "<think>\n", 8);
	if (stream) {
		const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
		if (wrs(fd, h)) g.gone = 1;
		sbuf b = {0};
		sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\"}\n\n", g.created, NAME);
		if (!g.gone && chunk(fd, b.s, b.n)) g.gone = 1;
		sb_free(&b);
	}
	char line[256]; int ngen = 0, ended = 0, resumed = 0, snapped = 0; double run_ms = 0, first_ms = 0;
	/* a long prompt takes minutes before the first token: an idle stream is cut by proxies (Cloudflare closes after 100 s), so an SSE comment goes out
	   every 10 s while the engine is silent */
	{ struct timeval tv = { 10, 0 }; setsockopt(fileno(in), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); }
	for (;;) {
		if (!fgets(line, sizeof line, in)) {
			if ((errno == EAGAIN || errno == EWOULDBLOCK) && stream && !g.gone) { clearerr(in); if (chunk(fd, ": keepalive\n\n", 13)) { g.gone = 1; break; } continue; }
			break;
		}
		unsigned id; int e, npm; double r, f;
		if (sscanf(line, "tok %u", &id) == 1) { on_token(&g, id); if (g.gone || g.cap_hit) break; }
		else if (sscanf(line, "done %d %d %d %lf %lf %d %d", &ngen, &e, &npm, &r, &f, &resumed, &snapped) >= 5) { ended = e; run_ms = r; first_ms = f; break; }
		else if (!strncmp(line, "error", 5)) { fprintf(stderr, "flashd: engine: %s", line); break; }
	}
	fclose(in); fclose(out);
	pthread_mutex_unlock(&ENGINE);
	free(prompt);

	if (g.gone) fprintf(stderr, "flashd: client left after %d bytes of text; the engine is told to stop\n", (int)(g.content.n + g.reasoning.n));
	double pp = first_ms > 0 ? (np - resumed) / (first_ms / 1e3) : 0, dec_ms = run_ms - first_ms, dec = (ngen > 1 && dec_ms > 0) ? (ngen - 1) / (dec_ms / 1e3) : 0;
	sbuf t = {0};
	sb_printf(&t, "{\"prompt_n\":%d,\"cache_n\":%d,\"prompt_ms\":%.1f,\"prompt_per_second\":%.2f,\"predicted_n\":%d,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f}", np, resumed, first_ms, pp, ngen, dec_ms > 0 ? dec_ms : 0, dec);
	const char *fin = ended ? "stop" : "length";
	fprintf(stderr, "flashd: %d prompt (%d from the cache, snapshot at %d) + %d generated tokens, %.1f ms (first %.1f), %s, pp %.1f decode %.1f tok/s\n", np, resumed, snapped, ngen, run_ms, first_ms, fin, pp, dec);
	if (stream) {
		if (!g.gone) {
			sbuf b = {0};
			sb_printf(&b, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion.chunk\",\"usage\":{\"completion_tokens\":%d,\"prompt_tokens\":%d,\"total_tokens\":%d},\"timings\":%s}\n\n", fin, g.created, NAME, ngen, np, ngen + np, t.s);
			chunk(fd, b.s, b.n); sb_free(&b);
			chunk(fd, "data: [DONE]\n\n", 14); wrs(fd, "0\r\n\r\n");
		}
	} else {
		sbuf b = {0};
		sb_puts(&b, "{\"choices\":[{\"index\":0,\"finish_reason\":\""); sb_puts(&b, fin); sb_puts(&b, "\",\"message\":{\"role\":\"assistant\",\"content\":");
		sb_json_str(&b, g.content.s ? g.content.s : "", g.content.n);
		if (g.reasoning.n) { sb_puts(&b, ",\"reasoning_content\":"); sb_json_str(&b, g.reasoning.s, g.reasoning.n); }
		sb_printf(&b, "}}],\"created\":%ld,\"id\":\"chatcmpl-flash\",\"model\":\"%s\",\"object\":\"chat.completion\",\"usage\":{\"completion_tokens\":%d,\"prompt_tokens\":%d,\"total_tokens\":%d},\"timings\":%s}", g.created, NAME, ngen, np, ngen + np, t.s);
		reply(fd, 200, "application/json", b.s, b.n); sb_free(&b);
	}
	sb_free(&t); sb_free(&g.content); sb_free(&g.reasoning); arena_free(a);
}

/* ---- HTTP ---- */
static void *serve(void *arg) {
	int fd = (int)(long)arg; struct timeval tv = { 30, 0 };
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
	char *req = malloc(1 << 16); size_t n = 0; char *hend = NULL;
	while (!hend && n < (1 << 16) - 1) {
		ssize_t k = recv(fd, req + n, (1 << 16) - 1 - n, 0); if (k <= 0) { free(req); close(fd); return NULL; }
		n += (size_t)k; req[n] = 0; hend = strstr(req, "\r\n\r\n");
	}
	if (!hend) { error(fd, 400, "bad request"); free(req); close(fd); return NULL; }
	size_t hl = (size_t)(hend - req) + 4, clen = 0;
	for (char *p = req; p < hend; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : hend)
		if (!strncasecmp(p, "content-length:", 15)) clen = (size_t)strtoull(p + 15, NULL, 10);
	if (clen > (128u << 20)) { error(fd, 413, "request too large"); free(req); close(fd); return NULL; }
	char *body = malloc(clen + 1); size_t have = n - hl > clen ? clen : n - hl; memcpy(body, req + hl, have);
	while (have < clen) { ssize_t k = recv(fd, body + have, clen - have, 0); if (k <= 0) break; have += (size_t)k; }
	body[have] = 0;
	char method[16] = "", path[256] = ""; sscanf(req, "%15s %255s", method, path);
	char *q = strchr(path, '?'); if (q) *q = 0;
	if (!strcmp(method, "GET") && !strcmp(path, "/health")) reply(fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
	else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
		sbuf b = {0}; sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":"); sb_json_str(&b, NAME, strlen(NAME)); sb_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"flashd\"}]}");
		reply(fd, 200, "application/json", b.s, b.n); sb_free(&b);
	} else if (!strcmp(method, "GET") && !strcmp(path, "/props")) {
		sbuf b = {0}; const char *t = tok_template(V);
		sb_printf(&b, "{\"default_generation_settings\":{\"n_ctx\":%d,\"temperature\":%g,\"top_k\":%d,\"top_p\":%g,\"presence_penalty\":0.0,\"repeat_penalty\":1.0,\"max_reasoning_tokens\":%ld,\"max_answer_tokens\":%ld},\"modalities\":{\"vision\":%s},\"model_alias\":", POSITIONS, DEF_TEMP, DEF_TOPK, DEF_TOPP, REASONING_MAX, ANSWER_MAX, VIS_SOCK ? "true" : "false"); sb_json_str(&b, NAME, strlen(NAME));
		sb_puts(&b, ",\"chat_template\":"); sb_json_str(&b, t ? t : "", t ? strlen(t) : 0); sb_putc(&b, '}');
		reply(fd, 200, "application/json", b.s, b.n); sb_free(&b);
	} else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) chat(fd, body, have);
	else error(fd, 404, "not found");
	free(body); free(req); close(fd);
	return NULL;
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: flashd <shard-1.gguf> <flash-socket> <host:port>\n"); return 2; }
	signal(SIGPIPE, SIG_IGN);
	SOCK = argv[2];
	if (getenv("FLASHD_NAME")) NAME = getenv("FLASHD_NAME");
	if (getenv("FLASHD_THINKING")) THINK_DEFAULT = atoi(getenv("FLASHD_THINKING"));
	V = tok_load(argv[1]); if (!V) return 1;
	VIS_SOCK = getenv("FLASHVIS_SOCK"); if (getenv("FLASHD_IMGDIR")) snprintf(IMGDIR, sizeof IMGDIR, "%s", getenv("FLASHD_IMGDIR"));
	ID_IMGPAD = tok_find(V, "<|image_pad|>");
	{ double x; if (!tok_meta_num(V, "general.sampling.temp", &x)) DEF_TEMP = x; if (!tok_meta_num(V, "general.sampling.top_p", &x)) DEF_TOPP = x; if (!tok_meta_num(V, "general.sampling.top_k", &x)) DEF_TOPK = (int)x;
	  if (getenv("FLASHD_GREEDY")) DEF_TEMP = 0.0; }
	const char *ts = tok_template(V); char err[512];
	if (!ts) { fprintf(stderr, "%s has no chat template\n", argv[1]); return 1; }
	TPL = jinja_compile(ts, strlen(ts), err, sizeof err);
	if (!TPL) { fprintf(stderr, "chat template: %s\n", err); return 1; }
	int e = tok_find(V, "<|im_end|>"), te = tok_find(V, "</think>");
	if (e < 0) { fprintf(stderr, "no <|im_end|> token\n"); return 1; }
	ID_END = (uint32_t)e; ID_THINK_END = te < 0 ? 0xffffffffu : (uint32_t)te;
	FILE *out = NULL, *in = engine_open(&out);
	if (!in) { fprintf(stderr, "cannot reach the engine at %s\n", SOCK); return 1; }
	fprintf(out, "info\n"); fflush(out);
	char line[128]; if (fgets(line, sizeof line, in) && sscanf(line, "info %d %u", &POSITIONS, &IMG_ROWS) >= 1) {} else { fprintf(stderr, "engine did not answer info\n"); return 1; }
	fclose(in); fclose(out);

	char host[128]; int port;
	if (sscanf(argv[3], "%127[^:]:%d", host, &port) != 2) { fprintf(stderr, "listen address must be host:port\n"); return 2; }
	int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) }; inet_pton(AF_INET, host, &sa.sin_addr);
	if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 64)) { perror(argv[3]); return 1; }
	fprintf(stderr, "flashd: %s, %d positions, sampling defaults from the GGUF: temp %g top_k %d top_p %g%s, listening on %s\n", NAME, POSITIONS, DEF_TEMP, DEF_TOPK, DEF_TOPP, DEF_TEMP > 0 ? "" : " (greedy)", argv[3]);
	for (;;) {
		int c = accept(ls, NULL, NULL); if (c < 0) { if (errno != EINTR) perror("accept"); continue; }
		pthread_t th; if (pthread_create(&th, NULL, serve, (void *)(long)c)) { close(c); continue; }
		pthread_detach(th);
	}
}
