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
#include <time.h>
#include <unistd.h>

static tok_vocab *V;
static jinja *TPL;
static const char *SOCK, *NAME = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-sass";
static int POSITIONS = 2048, THINK_DEFAULT = 1;
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
	char piece[256]; int n = tok_piece(V, id, piece, sizeof piece);
	if (n > 0) emit_text(g, piece, (size_t)n);
}

/* ---- POST /v1/chat/completions ---- */
static void chat(int fd, const char *body, size_t bn) {
	arena *a = arena_new(); const char *je = NULL;
	jv *req = json_parse(a, body, bn, &je);
	if (!req || req->k != J_DICT) { error(fd, 400, "the body is not a JSON object"); arena_free(a); return; }
	jv *msgs = jv_getz(req, "messages");
	if (!msgs || msgs->k != J_LIST || !msgs->l.n) { error(fd, 400, "messages must be a non-empty list"); arena_free(a); return; }
	jv *vars = jv_new(a, J_DICT);
	jv_set(a, vars, "messages", 8, msgs);
	jv_set(a, vars, "add_generation_prompt", 21, jv_bool(a, 1));
	jv *tools = jv_getz(req, "tools"); if (tools && tools->k == J_LIST && tools->l.n) jv_set(a, vars, "tools", 5, tools);
	jv *kw = jv_getz(req, "chat_template_kwargs");
	jv *think = kw ? jv_getz(kw, "enable_thinking") : NULL;
	if (think && think->k == J_BOOL) jv_set(a, vars, "enable_thinking", 15, think);
	else if (!THINK_DEFAULT) jv_set(a, vars, "enable_thinking", 15, jv_bool(a, 0));
	jv *eff = kw ? jv_getz(kw, "reasoning_effort") : NULL;
	if (eff && eff->k == J_STR) jv_set(a, vars, "reasoning_effort", 16, eff);
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
	long n_new = max_tokens < POSITIONS - np ? max_tokens : POSITIONS - np;
	FILE *out = NULL, *in = engine_open(&out);
	if (!in) { pthread_mutex_unlock(&ENGINE); free(ids); free(prompt); arena_free(a); error(fd, 500, "the engine is not running"); return; }
	fprintf(out, "stop %u %u\n", ID_END, tok_eos(V) == ID_END ? 0xffffffffu : tok_eos(V));
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
	char line[256]; int ngen = 0, ended = 0; double run_ms = 0, first_ms = 0;
	while (fgets(line, sizeof line, in)) {
		unsigned id; int e, npm; double r, f;
		if (sscanf(line, "tok %u", &id) == 1) on_token(&g, id);
		else if (sscanf(line, "done %d %d %d %lf %lf", &ngen, &e, &npm, &r, &f) == 5) { ended = e; run_ms = r; first_ms = f; break; }
		else if (!strncmp(line, "error", 5)) { fprintf(stderr, "flashd: engine: %s", line); break; }
	}
	fclose(in); fclose(out);
	pthread_mutex_unlock(&ENGINE);
	free(prompt);

	double pp = first_ms > 0 ? np / (first_ms / 1e3) : 0, dec_ms = run_ms - first_ms, dec = (ngen > 1 && dec_ms > 0) ? (ngen - 1) / (dec_ms / 1e3) : 0;
	sbuf t = {0};
	sb_printf(&t, "{\"prompt_n\":%d,\"prompt_ms\":%.1f,\"prompt_per_second\":%.2f,\"predicted_n\":%d,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f}", np, first_ms, pp, ngen, dec_ms > 0 ? dec_ms : 0, dec);
	const char *fin = ended ? "stop" : "length";
	fprintf(stderr, "flashd: %d prompt + %d generated tokens, %.1f ms (first %.1f), %s, pp %.1f decode %.1f tok/s\n", np, ngen, run_ms, first_ms, fin, pp, dec);
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
	if (clen > (4u << 20)) { error(fd, 413, "request too large"); free(req); close(fd); return NULL; }
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
		sb_printf(&b, "{\"default_generation_settings\":{\"n_ctx\":%d},\"model_alias\":", POSITIONS); sb_json_str(&b, NAME, strlen(NAME));
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
	char line[128]; if (fgets(line, sizeof line, in) && sscanf(line, "info %d", &POSITIONS) == 1) {} else { fprintf(stderr, "engine did not answer info\n"); return 1; }
	fclose(in); fclose(out);

	char host[128]; int port;
	if (sscanf(argv[3], "%127[^:]:%d", host, &port) != 2) { fprintf(stderr, "listen address must be host:port\n"); return 2; }
	int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) }; inet_pton(AF_INET, host, &sa.sin_addr);
	if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 64)) { perror(argv[3]); return 1; }
	fprintf(stderr, "flashd: %s, %d positions, listening on %s\n", NAME, POSITIONS, argv[3]);
	for (;;) {
		int c = accept(ls, NULL, NULL); if (c < 0) { if (errno != EINTR) perror("accept"); continue; }
		pthread_t th; if (pthread_create(&th, NULL, serve, (void *)(long)c)) { close(c); continue; }
		pthread_detach(th);
	}
}
