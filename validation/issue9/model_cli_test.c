#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_MANIFESTS 6

typedef struct {
    const char *budget;
    const char *placement;
    const char *ids_path;
    const char *inject;
    const char *notwin;
    const char *nph;
    const char *attn_map;
    const char *capture;
    int clear_ids;
} test_env_t;

typedef struct {
    const char *model_binary;
    const char *probe_library;
    char scratch[PATH_MAX];
    char *loader_path;
    char ids[5][PATH_MAX];
    char manifests[MAX_MANIFESTS][PATH_MAX];
    unsigned checks;
    unsigned failures;
} harness_t;

static char cleanup_path[PATH_MAX];

static void cleanup_scratch(void) {
    DIR *dir;
    struct dirent *entry;
    if (!cleanup_path[0] || !(dir = opendir(cleanup_path))) return;
    while ((entry = readdir(dir)) != NULL) {
        char path[PATH_MAX];
        int n;
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        n = snprintf(path, sizeof path, "%s/%s", cleanup_path, entry->d_name);
        if (n > 0 && (size_t)n < sizeof path) unlink(path);
    }
    closedir(dir);
    rmdir(cleanup_path);
}

static int make_path(char *dst, size_t capacity, const char *dir, const char *name) {
    int n = snprintf(dst, capacity, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= capacity) { errno = ENAMETOOLONG; return -1; }
    return 0;
}

static int write_text(const char *path, const char *text) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    size_t left = strlen(text);
    const char *p = text;
    if (fd < 0) return -1;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { int saved = errno ? errno : EIO; close(fd); errno = saved; return -1; }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) < 0) return -1;
    return 0;
}

static int set_or_unset(const char *name, const char *value) {
    return value ? setenv(name, value, 1) : unsetenv(name);
}

static int prepare_environment(const harness_t *h, const test_env_t *overrides) {
    const char *ids = h->ids[0];
    if (unsetenv("MODEL_NOTWIN") || unsetenv("MODEL_NPH") || unsetenv("MODEL_INJECT") ||
        unsetenv("ATTN_DIE") || unsetenv("MODEL_CAPTURE_QSA")) return -1;
    if (overrides && overrides->clear_ids) ids = NULL;
    else if (overrides && overrides->ids_path) ids = overrides->ids_path;
    if (setenv("MODEL_ATTN", "qsa", 1) ||
        setenv("FLASH_NBLK", overrides && overrides->budget ? overrides->budget : "1", 1) ||
        setenv("FLASH_KV_HOST", overrides && overrides->placement ? overrides->placement : "1", 1) ||
        setenv("ATTN_MAP", overrides && overrides->attn_map ? overrides->attn_map : "0,0,0,0,0,0,0,0,0,0,0,0", 1) ||
        setenv("SPLIT_SHARES", "1", 1) ||
        set_or_unset("MODEL_IDS", ids) ||
        set_or_unset("MODEL_INJECT", overrides ? overrides->inject : NULL) ||
        set_or_unset("MODEL_NOTWIN", overrides ? overrides->notwin : NULL) ||
        set_or_unset("MODEL_NPH", overrides ? overrides->nph : NULL) ||
        set_or_unset("MODEL_CAPTURE_QSA", overrides ? overrides->capture : NULL) ||
        setenv("LD_PRELOAD", h->probe_library, 1) ||
        setenv("LD_LIBRARY_PATH", h->loader_path, 1)) return -1;
    return 0;
}

static int marker_state(const char *path, int *exact_line) {
    FILE *f = fopen(path, "r");
    char line[512];
    int any = 0;
    *exact_line = 0;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (strstr(line, "CUDA_BOUNDARY_REACHED")) any = 1;
        if (!strcmp(line, "CUDA_BOUNDARY_REACHED")) *exact_line = 1;
    }
    if (ferror(f)) { fclose(f); return -1; }
    fclose(f);
    return any;
}

static void dump_file(const char *path) {
    FILE *f = fopen(path, "r");
    char buffer[2048];
    size_t n;
    if (!f) return;
    while ((n = fread(buffer, 1, sizeof buffer, f)) != 0) fwrite(buffer, 1, n, stderr);
    fclose(f);
}

static int run_case(harness_t *h, const char *name, int expected,
                    const char *b0, const char *b1, const char *tokens,
                    const char *flags, const test_env_t *overrides,
                    int manifest_count, const int *manifest_ids) {
    char stdout_path[PATH_MAX], stderr_path[PATH_MAX];
    char *child_argv[6 + MAX_MANIFESTS + 1];
    pid_t pid;
    int status = 0, exact_marker = 0, any_marker;
    int nout, nerr;
    h->checks++;
    nout = snprintf(stdout_path, sizeof stdout_path, "%s/%s.stdout", h->scratch, name);
    nerr = snprintf(stderr_path, sizeof stderr_path, "%s/%s.stderr", h->scratch, name);
    if (nout < 0 || (size_t)nout >= sizeof stdout_path || nerr < 0 || (size_t)nerr >= sizeof stderr_path) {
        fprintf(stderr, "FAIL %s: output path too long\n", name); h->failures++; return -1;
    }
    if (manifest_count < 1 || manifest_count > MAX_MANIFESTS) {
        fprintf(stderr, "FAIL %s: invalid manifest fixture count\n", name); h->failures++; return -1;
    }
    pid = fork();
    if (pid < 0) { perror("fork"); h->failures++; return -1; }
    if (pid == 0) {
        int out_fd = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err_fd = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out_fd < 0 || err_fd < 0 || dup2(out_fd, STDOUT_FILENO) < 0 || dup2(err_fd, STDERR_FILENO) < 0) _Exit(125);
        close(out_fd); close(err_fd);
        if (prepare_environment(h, overrides) < 0) { perror("test environment"); _Exit(125); }
        child_argv[0] = (char *)h->model_binary;
        child_argv[1] = "/unused-cache";
        child_argv[2] = (char *)b0;
        child_argv[3] = (char *)b1;
        child_argv[4] = (char *)tokens;
        child_argv[5] = (char *)flags;
        for (int i = 0; i < manifest_count; i++) {
            if (manifest_ids[i] < 0 || manifest_ids[i] >= MAX_MANIFESTS) _Exit(125);
            child_argv[6 + i] = h->manifests[manifest_ids[i]];
        }
        child_argv[6 + manifest_count] = NULL;
        execv(h->model_binary, child_argv);
        perror("execv model_test");
        _Exit(127);
    }
    for (;;) {
        pid_t waited = waitpid(pid, &status, 0);
        if (waited == pid) break;
        if (waited < 0 && errno == EINTR) continue;
        perror("waitpid"); h->failures++; return -1;
    }
    if (!WIFEXITED(status)) status = 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    else status = WEXITSTATUS(status);
    any_marker = marker_state(stderr_path, &exact_marker);
    if (status == expected && any_marker >= 0 &&
        ((expected == 99 && exact_marker) || (expected == 2 && !any_marker) ||
         (expected != 99 && expected != 2))) {
        printf("PASS %s exit=%d\n", name, status);
        return 0;
    }
    fprintf(stderr, "FAIL %s: exit %d, expected %d; CUDA marker %s\n",
            name, status, expected, any_marker < 0 ? "unreadable" : any_marker ? "present" : "absent");
    dump_file(stdout_path);
    dump_file(stderr_path);
    h->failures++;
    return -1;
}

static int prepare_fixtures(harness_t *h) {
    static const char *const ids_text[5] = {
        "10\n11\n12\n13\n14\n15\n16\n17\n18\n",
        "10\n11\n",
        "10\n248320\n",
        "10\n11junk\n",
        "10\n11\n12\n13\n14\n15\n16\n17\n18\nignored-tail\n"
    };
    for (int i = 0; i < 5; i++) {
        char name[32];
        int n = snprintf(name, sizeof name, "ids-%d.txt", i);
        if (n < 0 || (size_t)n >= sizeof name || make_path(h->ids[i], sizeof h->ids[i], h->scratch, name) < 0 ||
            write_text(h->ids[i], ids_text[i]) < 0) return -1;
    }
    for (int die = 0; die < MAX_MANIFESTS; die++) {
        char name[32], header[96];
        int n = snprintf(name, sizeof name, "manifest-%d.txt", die);
        int m = snprintf(header, sizeof header, "arena 4096 die %d\n", die);
        if (n < 0 || (size_t)n >= sizeof name || m < 0 || (size_t)m >= sizeof header ||
            make_path(h->manifests[die], sizeof h->manifests[die], h->scratch, name) < 0 ||
            write_text(h->manifests[die], header) < 0) return -1;
    }
    return 0;
}

static int initialize_harness(harness_t *h, const char *binary, const char *probe) {
    char template[] = "/tmp/flash-model-cli-c.XXXXXX";
    const char *old_loader;
    size_t loader_bytes;
    char *real_binary = realpath(binary, NULL);
    char *real_probe = realpath(probe, NULL);
    char *scratch;
    memset(h, 0, sizeof *h);
    if (!real_binary || !real_probe) {
        perror("realpath model_test/probe"); free(real_binary); free(real_probe); return -1;
    }
    h->model_binary = real_binary;
    h->probe_library = real_probe;
    scratch = mkdtemp(template);
    if (!scratch) { perror("mkdtemp"); return -1; }
    if (snprintf(h->scratch, sizeof h->scratch, "%s", scratch) >= (int)sizeof h->scratch ||
        snprintf(cleanup_path, sizeof cleanup_path, "%s", scratch) >= (int)sizeof cleanup_path) {
        errno = ENAMETOOLONG; perror("scratch path"); return -1;
    }
    if (atexit(cleanup_scratch)) { fprintf(stderr, "cannot register scratch cleanup\n"); return -1; }
    old_loader = getenv("LD_LIBRARY_PATH");
    loader_bytes = strlen(h->scratch) + (old_loader ? strlen(old_loader) + 1 : 0) + 1;
    h->loader_path = malloc(loader_bytes);
    if (!h->loader_path) { perror("malloc loader path"); return -1; }
    if (snprintf(h->loader_path, loader_bytes, "%s%s%s", h->scratch,
                 old_loader ? ":" : "", old_loader ? old_loader : "") >= (int)loader_bytes) {
        errno = ENAMETOOLONG; perror("LD_LIBRARY_PATH"); return -1;
    }
    if (prepare_fixtures(h) < 0) { perror("create test fixtures"); return -1; }
    if (getenv("CUDA_STUB_LIBRARY") && *getenv("CUDA_STUB_LIBRARY")) {
        char *stub = realpath(getenv("CUDA_STUB_LIBRARY"), NULL);
        char link_path[PATH_MAX];
        if (!stub || make_path(link_path, sizeof link_path, h->scratch, "libcuda.so.1") < 0 || symlink(stub, link_path) < 0) {
            perror("create CUDA stub link"); free(stub); return -1;
        }
        free(stub);
    }
    return 0;
}

int main(int argc, char **argv) {
    harness_t h;
    static const test_env_t no_overrides = {0};
    static const test_env_t invalid_budget = { .budget = "1junk" };
    static const test_env_t zero_budget = { .budget = "0" };
    static const test_env_t invalid_placement = { .placement = "2" };
    static const test_env_t selection_injection = { .inject = "selection" };
    static const test_env_t logits_injection = { .inject = "logits" };
    static const test_env_t dn_injection = { .inject = "dn" };
    static const test_env_t conv_injection = { .inject = "conv" };
    static const test_env_t ple_injection = { .inject = "ple" };
    static const test_env_t unknown_injection = { .inject = "unknown" };
    static const test_env_t timing_injection = { .inject = "layer", .notwin = "1" };
    static const test_env_t truncated_program = { .nph = "1" };
    static const test_env_t absent_owner = { .attn_map = "1,1,1,1,1,1,1,1,1,1,1,1" };
    static const test_env_t capture_on = { .capture = "1" };
    static const test_env_t capture_no_ids = { .capture = "1", .clear_ids = 1 };
    static const test_env_t capture_no_head = { .capture = "1" };
    static const test_env_t capture_no_embed = { .capture = "1" };
    static const test_env_t capture_free_running = { .capture = "1", .notwin = "1" };
    static const test_env_t capture_injected = { .capture = "1", .inject = "layer" };
    static const test_env_t capture_invalid_flag = { .capture = "2" };
    static const test_env_t capture_under_exclusion = { .capture = "1" };
    static const int die0[] = {0};
    static const int die00[] = {0, 0};
    static const int dies012345[] = {0, 1, 2, 3, 4, 5};
    test_env_t short_ids, bad_ids, junk_ids, capture_short_ids;
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_TEST_BINARY CUINIT_PROBE_SO\n", argv[0]);
        return 2;
    }
    if (initialize_harness(&h, argv[1], argv[2]) < 0) return 1;
    short_ids = (test_env_t){ .ids_path = h.ids[1] };
    bad_ids = (test_env_t){ .ids_path = h.ids[2] };
    junk_ids = (test_env_t){ .ids_path = h.ids[3] };
    capture_short_ids = (test_env_t){ .capture = "1", .ids_path = h.ids[1] };

    run_case(&h, "positive-control", 99, "0", "8", "9", "eh", &no_overrides, 1, die0);
    run_case(&h, "zero-length", 2, "0", "8", "0", "eh", &no_overrides, 1, die0);
    run_case(&h, "negative-length", 2, "0", "8", "-1", "eh", &no_overrides, 1, die0);
    run_case(&h, "beyond-capacity", 2, "0", "8", "262145", "eh", &no_overrides, 1, die0);
    run_case(&h, "malformed-length", 2, "0", "8", "9junk", "eh", &no_overrides, 1, die0);
    run_case(&h, "malformed-block", 2, "0junk", "8", "9", "eh", &no_overrides, 1, die0);
    run_case(&h, "invalid-flags", 2, "0", "8", "9", "ehx", &no_overrides, 1, die0);
    run_case(&h, "empty-flags", 2, "0", "8", "9", "", &no_overrides, 1, die0);
    run_case(&h, "invalid-budget", 2, "0", "8", "9", "eh", &invalid_budget, 1, die0);
    run_case(&h, "zero-budget", 2, "0", "8", "9", "eh", &zero_budget, 1, die0);
    run_case(&h, "invalid-placement", 2, "0", "8", "9", "eh", &invalid_placement, 1, die0);
    run_case(&h, "missing-teacher-ids", 2, "0", "8", "9", "eh", &short_ids, 1, die0);
    run_case(&h, "out-of-vocabulary-id", 2, "0", "8", "2", "eh", &bad_ids, 1, die0);
    run_case(&h, "malformed-id", 2, "0", "8", "2", "eh", &junk_ids, 1, die0);
    run_case(&h, "valid-id-prefix", 99, "0", "8", "9", "eh", &(test_env_t){ .ids_path = h.ids[4] }, 1, die0);
    run_case(&h, "no-attention-block", 2, "0", "3", "9", "eh", &no_overrides, 1, die0);
    run_case(&h, "unreachable-selection", 2, "0", "8", "4", "eh", &selection_injection, 1, die0);
    run_case(&h, "reachable-selection", 99, "0", "8", "9", "eh", &selection_injection, 1, die0);
    run_case(&h, "unreachable-logits", 2, "0", "8", "9", "e", &logits_injection, 1, die0);
    run_case(&h, "unreachable-dn", 2, "3", "4", "9", "eh", &dn_injection, 1, die0);
    run_case(&h, "unreachable-conv", 2, "3", "4", "9", "eh", &conv_injection, 1, die0);
    run_case(&h, "unreachable-ple", 2, "3", "4", "9", "eh", &ple_injection, 1, die0);
    run_case(&h, "unknown-injection", 2, "0", "8", "9", "eh", &unknown_injection, 1, die0);
    run_case(&h, "timing-injection", 2, "0", "8", "9", "eh", &timing_injection, 1, die0);
    run_case(&h, "truncated-program", 2, "0", "8", "9", "eh", &truncated_program, 1, die0);
    run_case(&h, "absent-attention-owner", 2, "0", "8", "9", "eh", &absent_owner, 1, die0);

    run_case(&h, "capture-partial-block-scope", 2, "0", "8", "9", "eh", &capture_on, 1, die0);
    run_case(&h, "capture-one-die-scope", 2, "0", "48", "9", "eh", &capture_on, 1, die0);
    run_case(&h, "capture-synthetic-ids", 2, "0", "48", "9", "eh", &capture_no_ids, 6, dies012345);
    run_case(&h, "capture-short-teacher-ids", 2, "0", "48", "9", "eh", &capture_short_ids, 6, dies012345);
    run_case(&h, "capture-missing-head", 2, "0", "48", "9", "e", &capture_no_head, 6, dies012345);
    run_case(&h, "capture-missing-embedding", 2, "0", "48", "9", "h", &capture_no_embed, 6, dies012345);
    run_case(&h, "capture-free-running", 2, "0", "48", "9", "eh", &capture_free_running, 6, dies012345);
    run_case(&h, "capture-injected", 2, "0", "48", "9", "eh", &capture_injected, 6, dies012345);
    run_case(&h, "capture-under-exclusion", 2, "0", "48", "7", "eh", &capture_under_exclusion, 6, dies012345);
    run_case(&h, "capture-invalid-flag", 2, "0", "48", "9", "eh", &capture_invalid_flag, 6, dies012345);
    run_case(&h, "duplicate-physical-manifest", 2, "0", "8", "9", "eh", &no_overrides, 2, die00);
    run_case(&h, "full-capture-positive-control", 99, "0", "48", "9", "eh", &capture_on, 6, dies012345);

    free((void *)h.model_binary);
    free((void *)h.probe_library);
    free(h.loader_path);
    printf("%s: %u CPU boundary checks; cuInit intercepted, no GPU validation\n",
           h.failures ? "FAIL" : "OK", h.checks);
    return h.failures ? 1 : 0;
}
