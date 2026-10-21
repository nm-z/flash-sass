#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <locale.h>
#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define LINE_CAP 8192
#define PROMPT_TOKENS 64
#define GENERATED_TOKENS 40
#define NEW_TOKENS (GENERATED_TOKENS - 1)
#define REPEATS 2
#define CONNECT_TIMEOUT_MS 5000
#define TRACE_TOKENS 3
#define TRACE_CAP (2u * 1024u * 1024u)

static const char *config_keys[] = {
    "kv_host", "ple_disk", "maxt", "layers", "nblk", "dies",
    "attn_dies", "split", "attn_map", "img_rows", "rope_owner",
    "trace_tokens", "two_position", "prefill_chunk", "attn_q_split",
    "attn_q_mask", "cubin", "cubin_fnv"
};

static char socket_path[PATH_MAX];
static char restore_path[PATH_MAX];
static pid_t core_pid;
static uid_t core_uid;
static unsigned long long core_start;
static char original_config[LINE_CAP];
static const char *trace_cubin;
static const char *custom_cubin;
static const char *base_cubin;
static const char *candidate_cubin;
static const char *load_cubin;
static int mutated;
static volatile sig_atomic_t interrupted;

typedef struct {
    unsigned ids[GENERATED_TOKENS];
    unsigned count;
    char timing[LINE_CAP];
} run_result;

static void on_signal(int sig) { (void)sig; interrupted = 1; }

static int fail(const char *what) {
    fprintf(stderr, "decode_ab: %s\n", what);
    return -1;
}

static int write_all(int fd, const char *buf, size_t len) {
    while (len) {
        ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            if (interrupted) return -1;
            continue;
        }
        if (n <= 0) return -1;
        buf += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_line(int fd, char *out, size_t cap) {
    size_t used = 0;
    while (used + 1 < cap) {
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        int ready = poll(&pfd, 1, 180000);
        if (ready < 0 && errno == EINTR) {
            if (interrupted) return -1;
            continue;
        }
        if (ready <= 0) return -1;
        char ch;
        ssize_t n = recv(fd, &ch, 1, 0);
        if (n < 0 && errno == EINTR) {
            if (interrupted) return -1;
            continue;
        }
        if (n <= 0) return -1;
        if (ch == '\n') {
            if (used && out[used - 1] == '\r') used--;
            out[used] = 0;
            return 0;
        }
        out[used++] = ch;
    }
    return -1;
}

static int connect_core(void) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof addr.sun_path) { close(fd); return -1; }
    strcpy(addr.sun_path, socket_path);
    /* A full server accept backlog must not hold the shared GPUQ lease indefinitely. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) { close(fd); return -1; }
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        if (errno != EINPROGRESS) { close(fd); return -1; }
        struct pollfd pfd = { .fd = fd, .events = POLLOUT };
        int ready;
        do ready = poll(&pfd, 1, CONNECT_TIMEOUT_MS);
        while (ready < 0 && errno == EINTR && !interrupted);
        if (!ready) {
            fprintf(stderr, "decode_ab: core connect timed out after %d ms\n", CONNECT_TIMEOUT_MS);
            errno = ETIMEDOUT; close(fd); return -1;
        }
        if (ready < 0) { close(fd); return -1; }
        int error = 0; socklen_t error_size = sizeof error;
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &error_size) < 0 || error) {
            if (error) errno = error;
            close(fd); return -1;
        }
    }
    if (fcntl(fd, F_SETFL, flags) < 0) { close(fd); return -1; }
    struct timeval tv = { .tv_sec = 180, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    struct ucred cred;
    socklen_t size = sizeof cred;
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &size) < 0 || cred.uid != getuid()) {
        close(fd);
        return -1;
    }
    if (!core_pid) { core_pid = cred.pid; core_uid = cred.uid; }
    if (cred.pid != core_pid || cred.uid != core_uid) { close(fd); return -1; }
    return fd;
}

static int test_connect_timeout(void) {
    int listener = -1, fillers[64], nf = 0, saw_full = 0, status = 1;
    for (size_t i = 0; i < sizeof fillers / sizeof fillers[0]; i++) fillers[i] = -1;
    snprintf(socket_path, sizeof socket_path, "/tmp/fs82-decode-ab-%ld.sock", (long)getpid());
    unlink(socket_path);
    listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) return fail("test listener socket"), 1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof addr.sun_path) { fail("test socket path too long"); goto cleanup; }
    strcpy(addr.sun_path, socket_path);
    if (bind(listener, (struct sockaddr *)&addr, sizeof addr) || listen(listener, 0)) {
        fail("test listener setup"); goto cleanup;
    }
    for (unsigned i = 0; i < sizeof fillers / sizeof fillers[0]; i++) {
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if (fd < 0) { fail("test filler socket"); goto cleanup; }
        if (!connect(fd, (struct sockaddr *)&addr, sizeof addr) || errno == EINPROGRESS) {
            fillers[nf++] = fd;
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            close(fd); saw_full = 1; break;
        }
        close(fd); fail("test backlog fill"); goto cleanup;
    }
    if (!saw_full) { fail("test could not fill the Unix listener backlog"); goto cleanup; }
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    int fd = connect_core();
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (fd >= 0) { close(fd); fail("connect unexpectedly passed a full listener backlog"); goto cleanup; }
    double elapsed_ms = (end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_nsec - begin.tv_nsec) / 1e6;
    if (elapsed_ms > CONNECT_TIMEOUT_MS + 1000.0) { fail("connect exceeded its bounded timeout"); goto cleanup; }
    printf("decode_ab connect timeout: full Unix backlog rejected in %.3f ms PASS\n", elapsed_ms);
    status = 0;
cleanup:
    for (int i = 0; i < nf; i++) close(fillers[i]);
    if (listener >= 0) close(listener);
    unlink(socket_path);
    return status;
}

static int send_command(int fd, const char *command) {
    if (strchr(command, '\n') || strchr(command, '\r')) return -1;
    return write_all(fd, command, strlen(command)) || write_all(fd, "\n", 1) ? -1 : 0;
}

static int control(const char *command, char *reply, size_t cap) {
    int fd = connect_core();
    if (fd < 0) return -1;
    int rc = send_command(fd, command);
    if (!rc) rc = shutdown(fd, SHUT_WR);
    if (!rc) rc = read_line(fd, reply, cap);
    close(fd);
    return rc ? -1 : 0;
}

static int read_to_eof(int fd, char *out, size_t cap) {
    size_t used = 0;
    while (used + 1 < cap) {
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        int ready = poll(&pfd, 1, 180000);
        if (ready < 0 && errno == EINTR) {
            if (interrupted) return -1;
            continue;
        }
        if (ready <= 0) return -1;
        ssize_t n = recv(fd, out + used, cap - used - 1, 0);
        if (n < 0 && errno == EINTR) {
            if (interrupted) return -1;
            continue;
        }
        if (n < 0) return -1;
        if (!n) {
            out[used] = 0;
            return used ? 0 : -1;
        }
        used += (size_t)n;
    }
    return -1;
}

static int dump_profile(const char *arm, int die) {
    char command[32], profile[LINE_CAP * 2];
    int fd = connect_core();
    if (fd < 0) return -1;
    snprintf(command, sizeof command, "prof %d all", die);
    int rc = send_command(fd, command);
    if (!rc) rc = shutdown(fd, SHUT_WR);
    if (!rc) rc = read_to_eof(fd, profile, sizeof profile);
    close(fd);
    if (rc) {
        fprintf(stderr, "decode_ab: failed to read die %d phase profile\n", die);
        return -1;
    }
    printf("PROFILE_BEGIN arm=%s die=%d\n%s", arm, die, profile);
    if (profile[strlen(profile) - 1] != '\n') putchar('\n');
    printf("PROFILE_END arm=%s die=%d\n", arm, die);
    fflush(stdout);
    return 0;
}

static int dump_trace(const char *arm, int die) {
    if (!trace_cubin) return 0;
    char command[32];
    char *trace = malloc(TRACE_CAP);
    if (!trace) return fail("trace buffer allocation");
    int fd = connect_core();
    if (fd < 0) { free(trace); return fail("trace socket connect"); }
    snprintf(command, sizeof command, "trace %d", die);
    int rc = send_command(fd, command);
    if (!rc) rc = shutdown(fd, SHUT_WR);
    if (!rc) rc = read_to_eof(fd, trace, TRACE_CAP);
    close(fd);
    if (rc || !strstr(trace, "trace_begin") || !strstr(trace, "trace_end") || strstr(trace, "error:")) {
        fprintf(stderr, "decode_ab: failed to read die %d trace\n", die);
        free(trace);
        return -1;
    }
    printf("TRACE_BEGIN arm=%s die=%d\n%s", arm, die, trace);
    if (trace[strlen(trace) - 1] != '\n') putchar('\n');
    printf("TRACE_END arm=%s die=%d\n", arm, die);
    fflush(stdout);
    free(trace);
    return 0;
}

static int read_start_tick(pid_t pid, unsigned long long *tick) {
    char path[64], buf[4096];
    snprintf(path, sizeof path, "/proc/%ld/stat", (long)pid);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return -1; }
    fclose(f);
    char *p = strrchr(buf, ')');
    if (!p || p[1] != ' ') return -1;
    p += 2;
    int field = 3;
    while (field < 22) {
        p = strchr(p, ' ');
        if (!p) return -1;
        p++; field++;
    }
    errno = 0;
    char *end;
    unsigned long long value = strtoull(p, &end, 10);
    if (errno || end == p) return -1;
    *tick = value;
    return 0;
}

static int core_argv_restore_path(char *out, size_t cap) {
    char path[64], data[8192];
    snprintf(path, sizeof path, "/proc/%ld/cmdline", (long)core_pid);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(data, 1, sizeof data - 1, f);
    fclose(f);
    if (!n) return -1;
    data[n] = 0;
    size_t i = 0;
    int arg = 0;
    while (i < n) {
        size_t len = strnlen(data + i, n - i);
        if (arg == 2) {
            char resolved[PATH_MAX];
            if (!realpath(data + i, resolved) || strlen(resolved) >= cap) return -1;
            strcpy(out, resolved);
            return 0;
        }
        i += len + 1;
        arg++;
    }
    return -1;
}

static int get_config(char *out, size_t cap) {
    if (control("config", out, cap)) return -1;
    return strncmp(out, "config ", 7) ? -1 : 0;
}

static int config_value(const char *config, const char *key, char *out, size_t cap) {
    size_t keylen = strlen(key);
    const char *p = config;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *end = strchr(p, ' ');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        if (len > keylen && !memcmp(p, key, keylen) && p[keylen] == '=') {
            size_t vlen = len - keylen - 1;
            if (vlen >= cap) return -1;
            memcpy(out, p + keylen + 1, vlen);
            out[vlen] = 0;
            return 0;
        }
        if (!end) break;
        p = end + 1;
    }
    return -1;
}

static int same_layout(const char *expected, const char *actual, int allow_trace_config) {
    char a[1024], b[1024];
    for (size_t i = 0; i < sizeof config_keys / sizeof config_keys[0]; i++) {
        if (allow_trace_config && (!strcmp(config_keys[i], "trace_tokens") ||
                                   !strcmp(config_keys[i], "cubin") ||
                                   !strcmp(config_keys[i], "cubin_fnv"))) continue;
        if (config_value(expected, config_keys[i], a, sizeof a) ||
            config_value(actual, config_keys[i], b, sizeof b) || strcmp(a, b)) {
            fprintf(stderr, "decode_ab: config mismatch at %s\n", config_keys[i]);
            return -1;
        }
    }
    return 0;
}

static int check_rev(const char *config, const char *expected) {
    char rev[128];
    if (config_value(config, "rev", rev, sizeof rev) || strcmp(rev, expected)) {
        fprintf(stderr, "decode_ab: expected rev=%s; got %s\n", expected,
                config_value(config, "rev", rev, sizeof rev) ? "<missing>" : rev);
        return -1;
    }
    return 0;
}

static int load_library(const char *path, const char *expected_rev,
                        const char *cubin_override) {
    char command[2 * PATH_MAX + 128], reply[LINE_CAP], config[LINE_CAP];
    const char *cubin = cubin_override ? cubin_override : load_cubin;
    int n = cubin
        ? snprintf(command, sizeof command, "reload %s FLASH_CUBIN=%s FLASH_TRACE=%d",
                   path, cubin, trace_cubin ? TRACE_TOKENS : 0)
        : snprintf(command, sizeof command, "reload %s", path);
    if (n < 0 || (size_t)n >= sizeof command) return -1;
    mutated = 1;
    if (control(command, reply, sizeof reply) || strncmp(reply, "ok: engine reloaded", 19)) {
        fprintf(stderr, "decode_ab: reload failed: %s\n", reply);
        return -1;
    }
    if (get_config(config, sizeof config) || check_rev(config, expected_rev) ||
        same_layout(original_config, config, cubin != NULL)) return -1;
    printf("CONFIG %s\n", config);
    return 0;
}

static int ack_command(int fd, const char *command, const char *prefix) {
    char line[LINE_CAP];
    if (send_command(fd, command) || read_line(fd, line, sizeof line)) return -1;
    return strncmp(line, prefix, strlen(prefix)) ? -1 : 0;
}

static int json_number(const char *json, const char *key, double *value) {
    char needle[128];
    snprintf(needle, sizeof needle, "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p || !(p = strchr(p, ':'))) return -1;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!strncmp(p, "null", 4)) return 1;
    errno = 0;
    char *end;
    double x = strtod(p, &end);
    if (errno || end == p) return -1;
    *value = x;
    return 0;
}

static void print_ids(const unsigned *ids, unsigned count) {
    putchar('[');
    for (unsigned i = 0; i < count; i++) printf("%s%u", i ? "," : "", ids[i]);
    putchar(']');
}

static int run_request(const char *arm, const char *phase, int rep, unsigned n_new,
                       run_result *result) {
    char line[LINE_CAP], command[2048];
    int fd = connect_core();
    if (fd < 0) return -1;
    if (ack_command(fd, "timings 1", "ok:") || ack_command(fd, "stop", "ok:") ||
        ack_command(fd, "sample 0 0 1 1", "ok:")) { close(fd); return -1; }
    int used = snprintf(command, sizeof command, "chat %u", n_new);
    for (unsigned i = 0; i < PROMPT_TOKENS; i++) {
        int n = snprintf(command + used, sizeof command - (size_t)used,
                         " %u", 10u + i);
        if (n < 0 || (size_t)n >= sizeof command - (size_t)used) { close(fd); return -1; }
        used += n;
    }
    if (send_command(fd, command)) { close(fd); return -1; }
    memset(result, 0, sizeof *result);
    int terminal = 0;
    while (!terminal && !interrupted) {
        if (read_line(fd, line, sizeof line)) { close(fd); return -1; }
        if (!strncmp(line, "tok ", 4)) {
            unsigned token;
            if (sscanf(line + 4, "%u", &token) != 1 || result->count >= GENERATED_TOKENS) {
                close(fd); return -1;
            }
            result->ids[result->count++] = token;
        } else if (!strncmp(line, "timings ", 8)) {
            if (strlen(line + 8) >= sizeof result->timing) { close(fd); return -1; }
            strcpy(result->timing, line + 8);
        } else if (!strncmp(line, "done ", 5)) {
            unsigned generated;
            if (sscanf(line + 5, "%u", &generated) != 1 || generated != n_new + 1 ||
                result->count != generated || !result->timing[0]) { close(fd); return -1; }
            terminal = 1;
        } else if (!strncmp(line, "error", 5)) {
            fprintf(stderr, "decode_ab: core returned %s\n", line);
            close(fd); return -1;
        }
    }
    close(fd);
    if (!terminal) return -1;
    double rate = 0.0;
    int rate_state = json_number(result->timing, "decode_tokens_per_second", &rate);
    if (n_new && (rate_state || rate <= 0.0)) return -1;
    if (n_new) {
        printf("{\"gpuq_metric\":\"decode.engine.tokens_s\",\"value\":%.9f,"
               "\"unit\":\"tokens/s\",\"dimensions\":{\"arm\":\"%s\","
               "\"phase\":\"%s\",\"rep\":%d,\"prompt_tokens\":%d,"
               "\"generated_tokens\":%u},\"token_ids\":",
               rate, arm, phase, rep, PROMPT_TOKENS, result->count);
        print_ids(result->ids, result->count);
        printf(",\"engine_timing\":%s}\n", result->timing);
    } else {
        printf("{\"event\":\"warmup\",\"arm\":\"%s\",\"phase\":\"%s\","
               "\"prompt_tokens\":%d,\"generated_tokens\":%u,\"token_ids\":",
               arm, phase, PROMPT_TOKENS, result->count);
        print_ids(result->ids, result->count);
        printf(",\"engine_timing\":%s}\n", result->timing);
    }
    fflush(stdout);
    return 0;
}

static int same_tokens(const run_result *a, const run_result *b) {
    return a->count == b->count && !memcmp(a->ids, b->ids, a->count * sizeof a->ids[0]);
}

static int restore_live(void) {
    for (unsigned attempt = 0; attempt < 20; attempt++) {
        char reply[LINE_CAP], config[LINE_CAP];
        char command[2 * PATH_MAX + 128], original_cubin[PATH_MAX];
        int n;
        if (load_cubin || base_cubin || candidate_cubin) {
            if (config_value(original_config, "cubin", original_cubin, sizeof original_cubin)) return -1;
            n = snprintf(command, sizeof command, "reload %s FLASH_CUBIN=%s FLASH_TRACE=0",
                         restore_path, original_cubin);
        } else n = snprintf(command, sizeof command, "reload %s", restore_path);
        if (n < 0 || (size_t)n >= sizeof command)
            return -1;
        if (!control(command, reply, sizeof reply) && !strncmp(reply, "ok: engine reloaded", 19) &&
            !get_config(config, sizeof config) && !strcmp(config, original_config)) {
            mutated = 0;
            printf("RESTORED %s\n", config);
            fflush(stdout);
            return 0;
        }
        fprintf(stderr, "decode_ab: restore attempt %u failed; retaining GPUQ job and retrying\n", attempt + 1);
        sleep(2);
    }
    return -1;
}

int main(int argc, char **argv) {
    setlocale(LC_NUMERIC, "C");
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGHUP, on_signal);
    if (argc == 2 && !strcmp(argv[1], "--test-connect")) return test_connect_timeout();
    if (argc != 7) {
        fprintf(stderr, "usage: %s SOCKET RESTORE_LIB BASE_LIB BASE_REV CANDIDATE_LIB CANDIDATE_REV\n", argv[0]);
        return 2;
    }
    const char *job_id = getenv("GPUQ_JOB_ID");
    if (!job_id || !*job_id) return fail("must run inside gpuq run"), 2;
    char host[256];
    if (gethostname(host, sizeof host) || strcmp(host, "archy")) return fail("must run on archy"), 2;
    if (strlen(argv[1]) >= sizeof socket_path || strlen(argv[2]) >= sizeof restore_path) return 2;
    strcpy(socket_path, argv[1]);
    trace_cubin = getenv("DECODE_AB_TRACE_CUBIN");
    custom_cubin = getenv("DECODE_AB_CUBIN");
    /* Use separate paths to compare SASS-only variants with the same library. */
    base_cubin = getenv("DECODE_AB_BASE_CUBIN");
    candidate_cubin = getenv("DECODE_AB_CANDIDATE_CUBIN");
    if (!!base_cubin != !!candidate_cubin)
        return fail("set both DECODE_AB_BASE_CUBIN and DECODE_AB_CANDIDATE_CUBIN"), 2;
    if ((trace_cubin && custom_cubin) ||
        (base_cubin && (trace_cubin || custom_cubin)))
        return fail("choose a single-cubin mode or separate base/candidate cubins"), 2;
    load_cubin = trace_cubin ? trace_cubin : custom_cubin;
    if (load_cubin && (!*load_cubin || access(load_cubin, R_OK) ||
                       strpbrk(load_cubin, " \t\r\n")))
        return fail("invalid custom cubin path"), 2;
    if (base_cubin && (!*base_cubin || access(base_cubin, R_OK) ||
                       strpbrk(base_cubin, " \t\r\n") || !*candidate_cubin ||
                       access(candidate_cubin, R_OK) ||
                       strpbrk(candidate_cubin, " \t\r\n")))
        return fail("invalid base/candidate cubin path"), 2;
    char resolved[PATH_MAX];
    if (!realpath(argv[2], resolved) || strlen(resolved) >= sizeof restore_path) return fail("missing restore library"), 2;
    strcpy(restore_path, resolved);
    if (access(argv[3], R_OK) || access(argv[5], R_OK)) return fail("missing baseline or candidate library"), 2;
    if (readlink(socket_path, resolved, sizeof resolved) < 0) {
        struct stat st; if (stat(socket_path, &st) || !S_ISSOCK(st.st_mode)) return fail("core socket is unavailable"), 2;
    }
    char config[LINE_CAP], core_restore[PATH_MAX];
    if (get_config(config, sizeof config)) return fail("cannot read core config"), 1;
    strcpy(original_config, config);
    if (config_value(config, "layers", resolved, sizeof resolved) || strcmp(resolved, "48/48") ||
        config_value(config, "nblk", resolved, sizeof resolved) || strcmp(resolved, "512") ||
        config_value(config, "dies", resolved, sizeof resolved) || strcmp(resolved, "6") ||
        config_value(config, "kv_host", resolved, sizeof resolved) || strcmp(resolved, "0") ||
        config_value(config, "ple_disk", resolved, sizeof resolved) || strcmp(resolved, "1") ||
        config_value(config, "trace_tokens", resolved, sizeof resolved) || strcmp(resolved, "0"))
        return fail("resident engine does not match the untraced six-die validation layout"), 1;
    if (core_argv_restore_path(core_restore, sizeof core_restore) || strcmp(core_restore, restore_path))
        return fail("restore library is not the resident core's original command-line library"), 1;
    if (read_start_tick(core_pid, &core_start)) return fail("cannot capture resident core start tick"), 1;
    if (check_rev(config, "attention-q-release")) return fail("unexpected live baseline revision"), 1;
    unsigned new_tokens = trace_cubin ? TRACE_TOKENS - 1u : NEW_TOKENS;
    printf("GPUQ_JOB_ID=%s CORE_PID=%ld CORE_START=%llu\n", job_id, (long)core_pid, core_start);
    printf("WORKLOAD prompt_tokens=%d prompt_ids=10..73 generated_tokens=%u sampler=greedy prefix_reuse=none\n",
           PROMPT_TOKENS, new_tokens + 1u);
    if (trace_cubin) printf("TRACE_PROFILE cubin=%s trace_tokens=%d\n", trace_cubin, TRACE_TOKENS);
    else if (custom_cubin) printf("CUSTOM_CUBIN cubin=%s trace_tokens=0\n", custom_cubin);
    else if (base_cubin) printf("CUBIN_AB base=%s candidate=%s trace_tokens=0\n", base_cubin, candidate_cubin);
    printf("LIVE_CONFIG %s\n", config);
    fflush(stdout);

    run_result base_ref[REPEATS], result;
    int status = 1;
    mutated = 1;
    if (load_library(argv[3], argv[4], base_cubin)) goto cleanup;
    if (run_request("baseline", "warmup", 0, 0, &result)) goto cleanup;
    for (int rep = 0; rep < REPEATS; rep++) {
        if (run_request("baseline", "pre", rep + 1, new_tokens, &base_ref[rep])) goto cleanup;
        if (rep && !same_tokens(&base_ref[0], &base_ref[rep])) { fail("baseline output IDs changed between repeats"); goto cleanup; }
    }
    for (int die = 0; die < 6; die++)
        if (dump_profile("baseline", die)) goto cleanup;
    for (int die = 0; die < 6; die++)
        if (dump_trace("baseline", die)) goto cleanup;

    if (load_library(argv[5], argv[6], candidate_cubin)) goto cleanup;
    if (run_request("candidate", "warmup", 0, 0, &result)) goto cleanup;
    for (int rep = 0; rep < REPEATS; rep++) {
        if (run_request("candidate", "measure", rep + 1, new_tokens, &result)) goto cleanup;
        if (!same_tokens(&base_ref[0], &result)) { fail("candidate output IDs differ from baseline"); goto cleanup; }
    }
    for (int die = 0; die < 6; die++)
        if (dump_profile("candidate", die)) goto cleanup;
    for (int die = 0; die < 6; die++)
        if (dump_trace("candidate", die)) goto cleanup;

    if (load_library(argv[3], argv[4], base_cubin)) goto cleanup;
    if (run_request("baseline", "warmup-after", 0, 0, &result)) goto cleanup;
    for (int rep = 0; rep < REPEATS; rep++) {
        if (run_request("baseline", "post", rep + 1, new_tokens, &result)) goto cleanup;
        if (!same_tokens(&base_ref[0], &result)) { fail("post-candidate baseline output IDs differ"); goto cleanup; }
    }
    status = 0;

cleanup:
    if (mutated && restore_live()) {
        fprintf(stderr, "decode_ab: CRITICAL: could not restore live original library %s\n", restore_path);
        return 75;
    }
    return status;
}
