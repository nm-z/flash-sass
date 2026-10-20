/* CPU-only regression: real flashd HTTP handler and loopback TCP, with a
 * simulated Unix-socket core and tokenizer. No model or GPU is used. */
#define main flashd_program_main
#include "flashd.c"
#undef main
#include <stdint.h>

static const char *test_detail, *test_name;
static int test_tokens, test_delayed_eof, test_listener, test_core_ok;
static void test_fail(const char *message) {
    fprintf(stderr, "FAIL [%s]: %s\n", test_name ? test_name : "setup", message);
    exit(1);
}
#define REQUIRE(c,m) do { if (!(c)) test_fail(m); } while (0)

/* The fixed prompt and two decoded pieces isolate transport behavior. */
tok_vocab *tok_load(const char *p) { (void)p; return (tok_vocab *)(uintptr_t)1; }
int tok_count(const tok_vocab *v) { (void)v; return 1024; }
int tok_encode(const tok_vocab *v, const char *s, size_t n, int special,
               uint32_t *ids, int capacity) {
    (void)v; (void)s; (void)n; (void)special;
    if (capacity < 32) return -1;
    for (int i = 0; i < 32; i++) ids[i] = (uint32_t)(100 + i);
    return 32;
}
int tok_piece(const tok_vocab *v, uint32_t id, char *out, int capacity) {
    (void)v;
    const char *piece = id == 1 ? "hello" : id == 2 ? " world" : "";
    size_t n = strlen(piece);
    if (n > (size_t)capacity) return -1;
    memcpy(out, piece, n); return (int)n;
}
int tok_find(const tok_vocab *v, const char *s) { (void)v; (void)s; return -1; }
uint32_t tok_eos(const tok_vocab *v) { (void)v; return 999; }
int tok_add_bos(const tok_vocab *v) { (void)v; return 0; }
uint32_t tok_bos(const tok_vocab *v) { (void)v; return 998; }
const char *tok_template(const tok_vocab *v) { (void)v; return "a prompt that has enough tokens"; }
const char *tok_text(const tok_vocab *v, uint32_t id, size_t *n) {
    (void)v; const char *s = id == 1 ? "hello" : id == 2 ? " world" : "";
    if (n) *n = strlen(s);
    return s;
}
int tok_is_special(const tok_vocab *v, uint32_t id) { (void)v; return id == 999; }
int tok_is_hidden(const tok_vocab *v, uint32_t id) { (void)v; return id == 999; }
int tok_meta_num(const tok_vocab *v, const char *k, double *x) { (void)v; (void)k; (void)x; return -1; }

static void *test_core(void *unused) {
    (void)unused;
    int fd = accept(test_listener, NULL, NULL);
    if (fd < 0) return NULL;
    FILE *in = fdopen(fd, "r"), *out = fdopen(dup(fd), "w");
    if (!in || !out) test_fail("opening simulated core streams");
    char *line = NULL; size_t capacity = 0;
    while (getline(&line, &capacity, in) >= 0) {
        if (!strcmp(line, "timings 1\n")) fputs("ok: timings 1\n", out);
        else if (!strcmp(line, "cancel_on_eof 1\n")) fputs("ok: cancel_on_eof 1\n", out);
        else if (!strncmp(line, "stop ", 5)) fputs("ok: stop 999 4294967295\n", out);
        else if (!strncmp(line, "sample ", 7)) fputs("ok: sample\n", out);
        else if (!strncmp(line, "snap ", 5)) fputs("ok: snap 31\n", out);
        else if (!strncmp(line, "chat ", 5)) {
            if (test_tokens) fputs("tok 1\ntok 2\n", out);
            fflush(out);
            if (test_delayed_eof) {
                /* Exceed flashd's two-second read timeout, then close cleanly. */
                struct timespec delay = {3, 0};
                while (nanosleep(&delay, &delay) < 0 && errno == EINTR) {}
            } else {
                fprintf(out, "error: %s\n", test_detail); fflush(out);
            }
            test_core_ok = 1; break;
        } else break;
        if (fflush(out)) break;
    }
    free(line); fclose(in); fclose(out); return NULL;
}
static void *test_http_server(void *arg) {
    int fd = accept(*(int *)arg, NULL, NULL);
    if (fd < 0) return NULL;
    return serve((void *)(intptr_t)fd);
}
static size_t occurrences(const char *s, const char *needle) {
    size_t n = 0, len = strlen(needle);
    while ((s = strstr(s, needle))) { n++; s += len; }
    return n;
}
static void send_all(int fd, const char *s, size_t len) {
    while (len) {
        ssize_t n = send(fd, s, len, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        REQUIRE(n > 0, "sending HTTP request"); s += n; len -= (size_t)n;
    }
}
static char *read_response(int fd) {
    size_t used = 0, capacity = 8192;
    char *b = malloc(capacity); REQUIRE(b, "allocating response");
    for (;;) {
        if (used + 4097 > capacity) {
            capacity *= 2; REQUIRE(capacity <= 1024 * 1024, "response size bound");
            char *b2 = realloc(b, capacity); REQUIRE(b2, "growing response"); b = b2;
        }
        ssize_t n = recv(fd, b + used, capacity - used - 1, 0);
        if (n < 0 && errno == EINTR) continue;
        REQUIRE(n >= 0, "reading HTTP response");
        if (!n) break;
        used += (size_t)n;
    }
    b[used] = 0; return b;
}
static void check_sse(const char *body, int expect_tokens) {
    const char *p = body;
    int errors = 0, done = 0, got_content = 0, failed = 0;
    while (*p) {
        const char *end = strchr(p, '\n');
        size_t n = end ? (size_t)(end - p) : strlen(p);
        char *line = strndup(p, n); REQUIRE(line, "allocating SSE line");
        if (!strncmp(line, "data:", 5)) {
            const char *payload = line + 5;
            while (*payload == ' ') payload++;
            if (!strncmp(payload, "[DONE]", 6)) {
                done++; REQUIRE(failed, "[DONE] precedes error");
            } else if (strstr(payload, "\"error\"")) {
                errors++; failed = 1;
            } else {
                REQUIRE(!failed, "SSE data after error");
                if (strstr(payload, "hello") || strstr(payload, " world")) got_content = 1;
                REQUIRE(!strstr(payload, "\"stop\"") && !strstr(payload, "\"length\""),
                        "successful finish reason");
                REQUIRE(!strstr(payload, "\"usage\"") && !strstr(payload, "\"timings\""),
                        "success metadata");
            }
        }
        free(line); if (!end) break; p = end + 1;
    }
    REQUIRE(errors == 1 && done == 1, "exactly one SSE error and [DONE] required");
    REQUIRE(got_content == !!expect_tokens, "nonempty decoded text before error");
    if (test_detail) REQUIRE(strstr(body, test_detail), "SSE lost core metadata");
}
static void test_case(const char *name, int streaming, int tokens,
                      const char *detail, int delayed_eof) {
    test_name = name; test_detail = detail; test_tokens = tokens;
    test_delayed_eof = delayed_eof; test_core_ok = 0;
    alarm(12); /* bounds the full exchange, including both thread joins */
    double start = mono_ms();
    struct sockaddr_in address = {.sin_family = AF_INET};
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(listener >= 0, "creating TCP listener");
    REQUIRE(!bind(listener, (struct sockaddr *)&address, sizeof address), "binding TCP");
    REQUIRE(!listen(listener, 1), "listening on TCP");
    socklen_t length = sizeof address;
    REQUIRE(!getsockname(listener, (struct sockaddr *)&address, &length), "reading TCP port");
    pthread_t ct, ht;
    REQUIRE(!pthread_create(&ct, NULL, test_core, NULL), "starting simulated core");
    REQUIRE(!pthread_create(&ht, NULL, test_http_server, &listener), "starting HTTP handler");
    int client = socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(client >= 0, "creating TCP client");
    REQUIRE(!connect(client, (struct sockaddr *)&address, sizeof address), "connecting TCP");
    char json[512], request[1024];
    int jn = snprintf(json, sizeof json,
        "{\"stream\":%s,\"max_tokens\":16,\"messages\":[{\"role\":\"user\","
        "\"content\":\"Count carefully from one through sixteen.\"}]}",
        streaming ? "true" : "false");
    int rn = snprintf(request, sizeof request,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nContent-Length: %d\r\n"
        "Connection: close\r\n\r\n%s", jn, json);
    REQUIRE(rn > 0 && (size_t)rn < sizeof request, "request size");
    send_all(client, request, (size_t)rn);
    /* Do not shutdown the write half: flashd interprets POLLRDHUP as abandonment. */
    char *response = read_response(client); close(client);
    REQUIRE(!pthread_join(ht, NULL) && !pthread_join(ct, NULL), "joining owned threads");
    close(listener); REQUIRE(test_core_ok, "simulated core completed chat");
    char *body = strstr(response, "\r\n\r\n");
    REQUIRE(body, "HTTP header terminator"); body += 4;
    printf("\nCASE simulated core: %s\n%s\n", name, response);
    if (streaming) {
        REQUIRE(!strncmp(response, "HTTP/1.1 200", 12), "SSE HTTP 200");
        REQUIRE(strstr(response, "Content-Type: text/event-stream"), "SSE content type");
        check_sse(body, tokens);
    } else {
        REQUIRE(!strncmp(response, "HTTP/1.1 502", 12), "JSON HTTP 502");
        REQUIRE(strstr(response, "Content-Type: application/json"), "JSON content type");
        REQUIRE(occurrences(body, "\"error\"") == 1, "one JSON error");
        REQUIRE(!strstr(body, "\"choices\"") && !strstr(body, "\"usage\"")
                && !strstr(body, "\"finish_reason\""), "no success JSON fields");
        REQUIRE(strstr(body, "\"complete\":false"), "error timings must be incomplete");
        if (detail) REQUIRE(strstr(body, detail), "JSON lost core metadata");
    }
    free(response); alarm(0);
    printf("PASS simulated core %s: %.3f ms\n", name, mono_ms() - start);
}
int main(void) {
    char dir[] = "/tmp/flashd-http-stub-XXXXXX", path[108], err[512];
    signal(SIGPIPE, SIG_IGN);
    REQUIRE(mkdtemp(dir), "creating private socket directory");
    snprintf(path, sizeof path, "%s/core.sock", dir);
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", path);
    test_listener = socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(test_listener >= 0, "creating simulated core");
    REQUIRE(!bind(test_listener, (struct sockaddr *)&sa, sizeof sa), "binding simulated core");
    REQUIRE(!listen(test_listener, 8), "listening for simulated core");
    V = (tok_vocab *)(uintptr_t)1; SOCK = path; POSITIONS = 2048;
    THINK_DEFAULT = 0; ID_END = 999; ID_THINK_END = UINT32_MAX;
    const char *tpl = "a prompt that has enough tokens";
    TPL = jinja_compile(tpl, strlen(tpl), err, sizeof err);
    REQUIRE(TPL, "compiling template");
    for (int eio = 0; eio < 2; eio++) {
        const char *detail = eio
            ? "PLE I/O position 160 head 3 offset 123456 bytes 17/90: Input/output error"
            : "PLE I/O position 160 head 3 offset 123456 bytes 17/90: unexpected EOF";
        for (int stream = 0; stream < 2; stream++) for (int tokens = 0; tokens < 2; tokens++) {
            char name[80];
            snprintf(name, sizeof name, "%s %s %s", stream ? "SSE" : "JSON",
                     tokens ? "after decoded text" : "before tokens", eio ? "EIO" : "EOF");
            test_case(name, stream, tokens, detail, 0);
        }
    }
    test_case("JSON timeout then clean EOF", 0, 0, NULL, 1);
    test_case("SSE text then timeout and clean EOF", 1, 1, NULL, 1);
    jinja_free(TPL); close(test_listener);
    REQUIRE(!unlink(path) && !rmdir(dir), "removing owned socket");
    puts("PASS ten HTTP/SSE CPU regressions; real GPU propagation remains unmeasured");
    return 0;
}
