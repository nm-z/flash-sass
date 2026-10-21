/* CPU socket regression for flashd's generation reader.
 * This checks frontend disconnect detection with pending engine output;
 * whole-model cancellation and slot reuse are checked separately. */
#define main flashd_program_main
#include "flashd.c"
#undef main

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #expr); \
        goto cleanup; \
    } \
} while (0)

static int write_all(int fd, const char *data, size_t size) {
    while (size) {
        ssize_t n = send(fd, data, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        data += n; size -= (size_t)n;
    }
    return 0;
}

int main(void) {
    int core[2] = {-1, -1}, client[2] = {-1, -1};
    FILE *in = NULL;
    int gone = 0, result = 1;
    char line[256], input_buffer[4096];
    static const char tokens[] = "tok 101\ntok 102\ntok 103\n";
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, core) == 0);
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, client) == 0);
    in = fdopen(core[0], "r");
    REQUIRE(in != NULL);
    core[0] = -1; /* Owned by in. */
    REQUIRE(setvbuf(in, input_buffer, _IOFBF, sizeof input_buffer) == 0);
    REQUIRE(write_all(core[1], tokens, sizeof tokens - 1) == 0);

    REQUIRE(generation_line(line, sizeof line, in, client[0], &gone) == line);
    REQUIRE(strcmp(line, "tok 101\n") == 0);
    REQUIRE(gone == 0);

    /* Core remains connected and readable: neither EOF, a timeout nor an
     * HTTP response write can explain detecting this disconnected client. */
    REQUIRE(close(client[1]) == 0);
    client[1] = -1;
    errno = EAGAIN;
    REQUIRE(generation_line(line, sizeof line, in, client[0], &gone) == NULL);
    REQUIRE(gone == 1);
    REQUIRE(fgets(line, sizeof line, in) == line);
    REQUIRE(strcmp(line, "tok 102\n") == 0);
    REQUIRE(generation_line(line, sizeof line, in, client[0], &gone) == NULL);
    REQUIRE(fgets(line, sizeof line, in) == line);
    REQUIRE(strcmp(line, "tok 103\n") == 0);

    puts("PASS: live client reads tokens; disconnect stops before pending engine output is consumed");
    result = 0;
cleanup:
    if (in) fclose(in);
    for (int i = 0; i < 2; i++) {
        if (core[i] >= 0) close(core[i]);
        if (client[i] >= 0) close(client[i]);
    }
    return result;
}
