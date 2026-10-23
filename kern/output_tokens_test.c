/* Exercise flashd's actual token decoder and SSE writer with the published vocabulary. */
#define main flashd_program_main
#include "flashd.c"
#undef main
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: output_tokens_test <shard1.gguf>\n"); return 2; }
    V = tok_load(argv[1]); if (!V) return 1;
    ID_THINK_END = (uint32_t)tok_find(V, "</think>");
    const char *expected = "<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>";
    uint32_t ids[256]; int n = tok_encode(V, expected, strlen(expected), 0, ids, 256);
    int open = tok_find(V, "<tool_call>"), close_tag = tok_find(V, "</tool_call>");
    if (n < 1 || open != 248058 || close_tag != 248059 ||
        tok_is_hidden(V, (uint32_t)open) || tok_is_hidden(V, (uint32_t)close_tag) ||
        !tok_is_special(V, (uint32_t)open) || !tok_is_hidden(V, 248044) || !tok_is_hidden(V, 248046)) return 1;
    for (int stream = 0; stream <= 1; stream++) {
        int sv[2]; if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return 1;
        gen g = {.fd = sv[0], .stream = stream};
        on_token(&g, 248044);
        for (int i = 0; i < n; i++) on_token(&g, ids[i]);
        on_token(&g, 248046);
        if (!g.content.s || strcmp(g.content.s, expected) || g.gone || g.cap_hit) {
            fprintf(stderr, "stream=%d decoded: %s\n", stream, g.content.s ? g.content.s : "(empty)"); return 1;
        }
        shutdown(sv[0], SHUT_WR);
        char wire[16384]; size_t used = 0; ssize_t nr;
        while ((nr = read(sv[1], wire + used, sizeof wire - 1 - used)) > 0) used += (size_t)nr;
        wire[used] = 0;
        if (stream && (!strstr(wire, "<tool_call>") || !strstr(wire, "</tool_call>"))) return 1;
        close(sv[0]); close(sv[1]); sb_free(&g.content); sb_free(&g.reasoning);
        printf("PASS stream=%d: USER_DEFINED tool delimiters retained; CONTROL hidden\n", stream);
    }
    return 0;
}
