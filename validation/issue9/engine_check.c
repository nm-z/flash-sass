#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <openssl/evp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define EC_DIES 6u
#define EC_LAYERS 48u
#define EC_ATTN 12u
#define EC_MAX_SNAPSHOTS 64u
#define EC_MAX_RUNS 4u
#define EC_VOCAB 248320u
#define EC_TOKEN_LIMIT 262142u
#define EC_MAX_RECEIPT_BYTES (16u * 1024u * 1024u)

typedef struct {
    char *repo;
    char *out;
    char *ple;
    char *revision;
    char *manifests[EC_DIES];
    pid_t holder_pids[EC_DIES];
    size_t manifest_count;
    size_t holder_count;
    char *weight_manifests[EC_DIES];
    size_t weight_manifest_count;
    char ple_receipt_path[4096];
    char ple_receipt_label[4096];
    char ple_receipt_sha256[65];
    char ple_declared_sha256[65];
    uint64_t ple_live_size;
    uint64_t ple_live_mtime_ns;
    uint64_t ple_live_ctime_ns;
    uint64_t ple_live_device;
    uint64_t ple_live_inode;
    unsigned physical[EC_DIES];
    unsigned visible[EC_DIES];
    unsigned shares[EC_DIES];
    unsigned attn_map[EC_ATTN];
    unsigned snapshots[EC_MAX_SNAPSHOTS];
    size_t snapshot_count;
    unsigned tokens;
    unsigned budget;
    unsigned maxt;
    unsigned ple_off;
    int lock_fd;
    unsigned timeout_seconds;
    unsigned min_free_mib;
    unsigned min_host_mib;
    char *visible_csv;
    char *shares_csv;
    char *attn_csv;
    char *kv_csv;
    char *snapshots_csv;
    int modes_scalar;
    int modes_chunk3;
    int capture_stages;
    char *ld_library_path;
    char *cuda_device_order;
} options_t;

typedef struct {
    char layers[EC_DIES][EC_LAYERS][17];
    unsigned char layer_present[EC_DIES][EC_LAYERS];
    char ple[EC_DIES][17];
    char logits[EC_DIES][17];
    unsigned char die_present[EC_DIES];
} fingerprint_t;

typedef struct {
    int valid;
    size_t token_count;
    uint32_t *tokens;
    fingerprint_t fingerprint;
} reference_t;

typedef struct {
    char name[64];
    char prompt;
    unsigned resume;
    unsigned snap;
    unsigned n_new;
} request_t;

typedef struct {
    char name[64];
    char mode[16];
    unsigned kv_host;
    pid_t child_pid;
    int exit_code;
    int signal_number;
    int timed_out;
    int interrupted;
    unsigned elapsed_ms;
    int passed;
    char stdin_path[4096];
    char stdout_path[4096];
    char stderr_path[4096];
    size_t requests_checked;
} run_result_t;

typedef struct {
    char path[4096];
    uint64_t size;
    uint64_t mtime_ns;
    char sha256[65];
} file_identity_t;

static uint64_t stat_mtime_ns(const struct stat *st);
static uint64_t stat_ctime_ns(const struct stat *st);

static volatile sig_atomic_t ec_interrupted;
static char ec_error[2048];

static void ec_signal(int signo) {
    (void)signo;
    ec_interrupted = 1;
}

static int ec_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ec_error, sizeof ec_error, fmt, ap);
    va_end(ap);
    return -1;
}

static double monotime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *string_dup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static char *absolute_path(const char *path) {
    char cwd[4096];
    char *out;
    size_t n;
    if (path[0] == '/') return string_dup(path);
    if (!getcwd(cwd, sizeof cwd)) return NULL;
    n = strlen(cwd) + 1 + strlen(path) + 1;
    out = malloc(n);
    if (!out) return NULL;
    snprintf(out, n, "%s/%s", cwd, path);
    return out;
}

static int parse_uint(const char *s, unsigned max, unsigned *out) {
    uint64_t v = 0;
    if (!s || !*s) return -1;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        unsigned digit = (unsigned)(*p - '0');
        if (digit > max || v > ((uint64_t)max - digit) / 10u) return -1;
        v = v * 10u + digit;
    }
    *out = (unsigned)v;
    return 0;
}

static int parse_pid(const char *s, pid_t *out) {
    unsigned v;
    if (parse_uint(s, 2147483647u, &v) < 0 || !v) return -1;
    *out = (pid_t)v;
    return 0;
}

static int parse_csv(const char *s, unsigned *values, size_t expected,
                     unsigned maximum, const char *what) {
    char *copy = string_dup(s);
    char *part;
    size_t count = 0;
    if (!copy) return ec_fail("allocate %s list", what);
    part = copy;
    for (;;) {
        char *comma = strchr(part, ',');
        if (comma) *comma = '\0';
        if (!*part) { free(copy); return ec_fail("empty item in %s list", what); }
        if (count >= expected || parse_uint(part, maximum, &values[count]) < 0) {
            free(copy);
            return ec_fail("invalid %s list", what);
        }
        count++;
        if (!comma) break;
        part = comma + 1;
    }
    free(copy);
    if (count != expected) return ec_fail("%s list has %zu items, expected %zu", what, count, expected);
    return 0;
}

static int parse_snapshots(const char *s, options_t *o) {
    char *copy = string_dup(s), *part;
    if (!copy) return ec_fail("allocate snapshot list");
    part = copy;
    for (;;) {
        char *comma = strchr(part, ',');
        unsigned v;
        if (comma) *comma = '\0';
        if (!*part) { free(copy); return ec_fail("empty snapshot position"); }
        if (o->snapshot_count >= EC_MAX_SNAPSHOTS ||
            parse_uint(part, o->tokens - 1, &v) < 0 || !v) {
            free(copy);
            return ec_fail("invalid snapshot position list");
        }
        for (size_t i = 0; i < o->snapshot_count; i++) if (o->snapshots[i] == v) {
            free(copy);
            return ec_fail("duplicate snapshot position %u", v);
        }
        o->snapshots[o->snapshot_count++] = v;
        if (!comma) break;
        part = comma + 1;
    }
    free(copy);
    if (!o->snapshot_count) return ec_fail("at least one snapshot is required");
    return 0;
}

static int format_csv(const unsigned *values, size_t count, char **out) {
    size_t cap = count * 12 + 1;
    size_t used = 0;
    char *s = malloc(cap);
    if (!s) return ec_fail("allocate normalized list");
    s[0] = '\0';
    for (size_t i = 0; i < count; i++) {
        int n = snprintf(s + used, cap - used, "%s%u", i ? "," : "", values[i]);
        if (n < 0 || (size_t)n >= cap - used) { free(s); return ec_fail("format normalized list"); }
        used += (size_t)n;
    }
    *out = s;
    return 0;
}

static int revision_valid(const char *s) {
    size_t n = strlen(s);
    if (n >= 7 && n <= 40) {
        for (size_t i = 0; i < n; i++) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return 0;
        return 1;
    }
    if (n >= 13 && n <= 46 && !strcmp(s + n - 6, "-dirty")) {
        for (size_t i = 0; i < n - 6; i++) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return 0;
        return 1;
    }
    return 0;
}

static void usage(FILE *f, const char *program) {
    fprintf(f,
        "usage: %s --repo DIR --out NEW_DIR --ple FILE --revision HEX "
        "--manifest FILE x6 --holder-pid PID x6 --cuda-visible CSV "
        "--split-shares CSV --attn-map CSV --tokens N [--snapshots CSV] "
        "--nblk N --lock-fd FD [--ple-off N] [--timeout SEC] "
        "[--weight-manifest FILE ...] [--capture-stages] [--self-test]\n", program);
}

static int add_arg_string(char **slot, const char *value, const char *name) {
    if (*slot) return ec_fail("duplicate %s option", name);
    *slot = string_dup(value);
    return *slot ? 0 : ec_fail("allocate %s option", name);
}

static int add_manifest(options_t *o, const char *path) {
    if (o->manifest_count >= EC_DIES) return ec_fail("more than six manifests");
    o->manifests[o->manifest_count] = absolute_path(path);
    if (!o->manifests[o->manifest_count]) return ec_fail("resolve manifest path");
    o->manifest_count++;
    return 0;
}

static int add_holder(options_t *o, const char *pid_text) {
    if (o->holder_count >= EC_DIES) return ec_fail("more than six holder PIDs");
    if (parse_pid(pid_text, &o->holder_pids[o->holder_count]) < 0)
        return ec_fail("invalid holder PID");
    o->holder_count++;
    return 0;
}

static int add_weight_manifest(options_t *o, const char *path) {
    if (o->weight_manifest_count >= EC_DIES) return ec_fail("more than six weight manifests");
    o->weight_manifests[o->weight_manifest_count] = absolute_path(path);
    if (!o->weight_manifests[o->weight_manifest_count]) return ec_fail("resolve weight manifest path");
    o->weight_manifest_count++;
    return 0;
}

static int parse_options(int argc, char **argv, options_t *o) {
    enum {
        OPT_REPO = 1000, OPT_OUT, OPT_PLE, OPT_REVISION, OPT_MANIFEST, OPT_PID,
        OPT_WEIGHT_MANIFEST, OPT_VISIBLE, OPT_SHARES, OPT_ATTN, OPT_TOKENS,
        OPT_SNAPSHOTS, OPT_NBLK, OPT_LOCK, OPT_PLE_OFF, OPT_TIMEOUT, OPT_KV,
        OPT_MIN_FREE, OPT_MIN_HOST, OPT_CAPTURE_STAGES
    };
    static const struct option opts[] = {
        {"repo", required_argument, NULL, OPT_REPO},
        {"out", required_argument, NULL, OPT_OUT},
        {"ple", required_argument, NULL, OPT_PLE},
        {"revision", required_argument, NULL, OPT_REVISION},
        {"manifest", required_argument, NULL, OPT_MANIFEST},
        {"holder-pid", required_argument, NULL, OPT_PID},
        {"weight-manifest", required_argument, NULL, OPT_WEIGHT_MANIFEST},
        {"cuda-visible", required_argument, NULL, OPT_VISIBLE},
        {"split-shares", required_argument, NULL, OPT_SHARES},
        {"attn-map", required_argument, NULL, OPT_ATTN},
        {"tokens", required_argument, NULL, OPT_TOKENS},
        {"snapshots", required_argument, NULL, OPT_SNAPSHOTS},
        {"nblk", required_argument, NULL, OPT_NBLK},
        {"lock-fd", required_argument, NULL, OPT_LOCK},
        {"ple-off", required_argument, NULL, OPT_PLE_OFF},
        {"timeout", required_argument, NULL, OPT_TIMEOUT},
        {"kv-hosts", required_argument, NULL, OPT_KV},
        {"min-free-mib", required_argument, NULL, OPT_MIN_FREE},
        {"min-host-mib", required_argument, NULL, OPT_MIN_HOST},
        {"capture-stages", no_argument, NULL, OPT_CAPTURE_STAGES},
        {NULL, 0, NULL, 0}
    };
    unsigned tmp;
    int c;
    memset(o, 0, sizeof *o);
    o->lock_fd = 9;
    o->timeout_seconds = 600;
    o->min_free_mib = 1024;
    o->min_host_mib = 1024;
    o->budget = 512;
    o->tokens = 9;
    o->ple_off = 192;
    o->modes_scalar = 1;
    o->modes_chunk3 = 1;
    while ((c = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        switch (c) {
        case OPT_REPO: if (add_arg_string(&o->repo, optarg, "repo") < 0) return -1; break;
        case OPT_OUT: if (add_arg_string(&o->out, optarg, "out") < 0) return -1; break;
        case OPT_PLE: if (add_arg_string(&o->ple, optarg, "ple") < 0) return -1; break;
        case OPT_REVISION: if (add_arg_string(&o->revision, optarg, "revision") < 0) return -1; break;
        case OPT_MANIFEST: if (add_manifest(o, optarg) < 0) return -1; break;
        case OPT_PID: if (add_holder(o, optarg) < 0) return -1; break;
        case OPT_WEIGHT_MANIFEST: if (add_weight_manifest(o, optarg) < 0) return -1; break;
        case OPT_VISIBLE: if (add_arg_string(&o->visible_csv, optarg, "cuda-visible") < 0) return -1; break;
        case OPT_SHARES: if (add_arg_string(&o->shares_csv, optarg, "split-shares") < 0) return -1; break;
        case OPT_ATTN: if (add_arg_string(&o->attn_csv, optarg, "attn-map") < 0) return -1; break;
        case OPT_TOKENS:
            if (parse_uint(optarg, EC_TOKEN_LIMIT, &o->tokens) < 0) return ec_fail("invalid --tokens");
            break;
        case OPT_SNAPSHOTS: if (add_arg_string(&o->snapshots_csv, optarg, "snapshots") < 0) return -1; break;
        case OPT_NBLK:
            if (parse_uint(optarg, 4096, &o->budget) < 0) return ec_fail("invalid --nblk");
            break;
        case OPT_LOCK:
            if (parse_uint(optarg, 1048576u, &tmp) < 0 || tmp < 3) return ec_fail("invalid --lock-fd");
            o->lock_fd = (int)tmp;
            break;
        case OPT_PLE_OFF:
            if (parse_uint(optarg, UINT32_MAX, &o->ple_off) < 0) return ec_fail("invalid --ple-off");
            break;
        case OPT_TIMEOUT:
            if (parse_uint(optarg, 86400, &o->timeout_seconds) < 0 || !o->timeout_seconds)
                return ec_fail("invalid --timeout");
            break;
        case OPT_KV:
            if (add_arg_string(&o->kv_csv, optarg, "kv-hosts") < 0) return -1;
            break;
        case OPT_MIN_FREE:
            if (parse_uint(optarg, UINT32_MAX, &o->min_free_mib) < 0 || !o->min_free_mib) return ec_fail("invalid --min-free-mib");
            break;
        case OPT_MIN_HOST:
            if (parse_uint(optarg, UINT32_MAX, &o->min_host_mib) < 0 || !o->min_host_mib) return ec_fail("invalid --min-host-mib");
            break;
        case OPT_CAPTURE_STAGES:
            if (o->capture_stages) return ec_fail("duplicate --capture-stages");
            o->capture_stages = 1;
            break;
        default: return ec_fail("unknown command-line option");
        }
    }
    if (optind != argc) return ec_fail("unexpected positional arguments");
    if (!o->snapshots_csv) o->snapshots_csv = string_dup("1,2,3,5");
    if (!o->kv_csv) o->kv_csv = string_dup("0,1");
    if (!o->repo || !o->out || !o->ple || !o->revision || !o->visible_csv ||
        !o->shares_csv || !o->attn_csv || !o->snapshots_csv || !o->kv_csv ||
        o->manifest_count != EC_DIES || o->holder_count != EC_DIES || !o->weight_manifest_count)
        return ec_fail("missing required paths, topology, holders, or request scope");
    if (o->tokens < 6 || o->tokens > EC_TOKEN_LIMIT || !o->budget || !revision_valid(o->revision))
        return ec_fail("invalid full-model scope");
    if (o->capture_stages && o->tokens > 15)
        return ec_fail("--capture-stages supports at most 15 prompt tokens plus one generated position");
    if (o->tokens > UINT32_MAX - 129u) return ec_fail("token count is too large for engine capacity");
    o->maxt = 128u * ((o->tokens + 2u + 127u) / 128u);
    if (parse_csv(o->visible_csv, o->visible, EC_DIES, 255, "CUDA visibility") < 0 ||
        parse_csv(o->shares_csv, o->shares, EC_DIES, 1000000u, "split shares") < 0 ||
        parse_csv(o->attn_csv, o->attn_map, EC_ATTN, EC_DIES - 1u, "attention map") < 0 ||
        parse_snapshots(o->snapshots_csv, o) < 0) return -1;
    {
        unsigned kvs[2];
        if (parse_csv(o->kv_csv, kvs, 2, 1, "KV host placement") < 0 ||
            kvs[0] != 0 || kvs[1] != 1) return ec_fail("--kv-hosts must request both placements in order: 0,1");
    }
    {
        uint64_t sum = 0;
        for (unsigned d = 0; d < EC_DIES; d++) {
            if (!o->shares[d]) return ec_fail("split shares must all be positive");
            sum += o->shares[d];
            for (unsigned e = 0; e < d; e++) if (o->visible[e] == o->visible[d])
                return ec_fail("CUDA visibility contains a duplicate physical die");
        }
        if (sum != 600) return ec_fail("production split shares must sum to 600");
    }
    if (!o->modes_scalar || !o->modes_chunk3) return ec_fail("both scalar and chunk3 modes are required");
    {
        char *normalized = NULL;
        if (format_csv(o->visible, EC_DIES, &normalized) < 0) return -1;
        free(o->visible_csv); o->visible_csv = normalized;
        if (format_csv(o->shares, EC_DIES, &normalized) < 0) return -1;
        free(o->shares_csv); o->shares_csv = normalized;
        if (format_csv(o->attn_map, EC_ATTN, &normalized) < 0) return -1;
        free(o->attn_csv); o->attn_csv = normalized;
        if (format_csv((const unsigned[]){0, 1}, 2, &normalized) < 0) return -1;
        free(o->kv_csv); o->kv_csv = normalized;
        if (format_csv(o->snapshots, o->snapshot_count, &normalized) < 0) return -1;
        free(o->snapshots_csv); o->snapshots_csv = normalized;
    }
    {
        const char *v = getenv("LD_LIBRARY_PATH");
        if (v && !(o->ld_library_path = string_dup(v))) return ec_fail("copy LD_LIBRARY_PATH");
        v = getenv("CUDA_DEVICE_ORDER");
        if (v && !(o->cuda_device_order = string_dup(v))) return ec_fail("copy CUDA_DEVICE_ORDER");
    }
    return 0;
}

static int path_join(char *out, size_t cap, const char *dir, const char *leaf) {
    int n = snprintf(out, cap, "%s/%s", dir, leaf);
    return n >= 0 && (size_t)n < cap ? 0 : ec_fail("path is too long");
}

static int directory_is_new(const char *path) {
    struct stat st;
    if (lstat(path, &st) == 0) return ec_fail("output directory already exists: %s", path);
    if (errno != ENOENT) return ec_fail("stat output directory %s: %s", path, strerror(errno));
    if (mkdir(path, 0755) < 0) return ec_fail("create output directory %s: %s", path, strerror(errno));
    return 0;
}

static int manifest_die(const char *path, unsigned *die) {
    FILE *f = fopen(path, "r");
    size_t bytes;
    int physical;
    if (!f) return ec_fail("open manifest %s: %s", path, strerror(errno));
    if (fscanf(f, "arena %zu die %d", &bytes, &physical) != 2 || !bytes || physical < 0 || physical > 255) {
        fclose(f);
        return ec_fail("invalid holder manifest header: %s", path);
    }
    if (fclose(f) != 0) return ec_fail("close manifest %s: %s", path, strerror(errno));
    *die = (unsigned)physical;
    return 0;
}

static int validate_process_inputs(options_t *o) {
    struct stat lock_stat, known_stat;
    struct stat ple_stat;
    int fd_flags;
    for (unsigned i = 0; i < EC_DIES; i++) {
        if (kill(o->holder_pids[i], 0) < 0 && errno != EPERM)
            return ec_fail("holder PID %ld is not alive", (long)o->holder_pids[i]);
        if (manifest_die(o->manifests[i], &o->physical[i]) < 0) return -1;
        for (unsigned j = 0; j < i; j++) {
            if (!strcmp(o->manifests[i], o->manifests[j])) return ec_fail("duplicate manifest path");
            if (o->holder_pids[i] == o->holder_pids[j]) return ec_fail("duplicate holder PID");
            if (o->physical[i] == o->physical[j]) return ec_fail("manifests do not name six distinct physical dies");
        }
    }
    if (!strcmp(o->ple, "-") || stat(o->ple, &ple_stat) < 0 || !S_ISREG(ple_stat.st_mode) || ple_stat.st_size < 0)
        return ec_fail("full-model request checks require a regular PLE shard file");
    if ((uint64_t)ple_stat.st_size != o->ple_live_size || stat_mtime_ns(&ple_stat) != o->ple_live_mtime_ns ||
        stat_ctime_ns(&ple_stat) != o->ple_live_ctime_ns ||
        (uint64_t)ple_stat.st_dev != o->ple_live_device || (uint64_t)ple_stat.st_ino != o->ple_live_inode)
        return ec_fail("live PLE shard metadata changed after receipt binding");
    {
        const uint64_t ple_bytes = UINT64_C(320001536) * 90u;
        uint64_t ple_end = (uint64_t)o->ple_off + ple_bytes;
        if ((uint64_t)ple_stat.st_size < ple_end)
            return ec_fail("PLE shard does not cover the production table at the requested offset");
    }
    if (fstat(o->lock_fd, &lock_stat) < 0 ||
        stat("/tmp/flash-sass-engine-validation.lock", &known_stat) < 0 ||
        lock_stat.st_dev != known_stat.st_dev || lock_stat.st_ino != known_stat.st_ino)
        return ec_fail("--lock-fd is not the engine validation reservation");
    if (flock(o->lock_fd, LOCK_EX | LOCK_NB) < 0)
        return ec_fail("engine validation reservation is not held: %s", strerror(errno));
    fd_flags = fcntl(o->lock_fd, F_GETFD);
    if (fd_flags < 0 || fcntl(o->lock_fd, F_SETFD, fd_flags | FD_CLOEXEC) < 0)
        return ec_fail("set close-on-exec on validation reservation: %s", strerror(errno));
    return 0;
}

static int make_absolute_options(options_t *o) {
    char *p;
    char **paths[] = {&o->repo, &o->out, &o->ple};
    for (size_t i = 0; i < sizeof paths / sizeof paths[0]; i++) {
        p = absolute_path(*paths[i]);
        if (!p) return ec_fail("resolve path: %s", strerror(errno));
        free(*paths[i]); *paths[i] = p;
    }
    return 0;
}

static int available_host_mib(uint64_t *mib) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    if (!f) return ec_fail("open /proc/meminfo: %s", strerror(errno));
    while (fgets(line, sizeof line, f)) {
        unsigned long long kb;
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            fclose(f);
            *mib = (uint64_t)kb / 1024u;
            return 0;
        }
    }
    fclose(f);
    return ec_fail("MemAvailable is absent from /proc/meminfo");
}

static int check_resource_floors(const options_t *o) {
    char parent[4096];
    char *slash;
    struct statvfs vfs;
    uint64_t free_bytes, host_mib;
    if (strlen(o->out) >= sizeof parent) return ec_fail("output path is too long");
    strcpy(parent, o->out);
    slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) parent[1] = '\0';
    else *slash = '\0';
    if (statvfs(parent, &vfs) < 0) return ec_fail("statvfs %s: %s", parent, strerror(errno));
    if (vfs.f_bavail && vfs.f_frsize > UINT64_MAX / vfs.f_bavail) free_bytes = UINT64_MAX;
    else free_bytes = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
    if (free_bytes / (1024u * 1024u) < o->min_free_mib)
        return ec_fail("output filesystem has less than %u MiB available", o->min_free_mib);
    if (available_host_mib(&host_mib) < 0) return -1;
    if (host_mib < o->min_host_mib)
        return ec_fail("host has %" PRIu64 " MiB available, below %u MiB", host_mib, o->min_host_mib);
    return 0;
}

static uint64_t stat_mtime_ns(const struct stat *st) {
    return (uint64_t)st->st_mtim.tv_sec * UINT64_C(1000000000) + (uint64_t)st->st_mtim.tv_nsec;
}

static uint64_t stat_ctime_ns(const struct stat *st) {
    return (uint64_t)st->st_ctim.tv_sec * UINT64_C(1000000000) + (uint64_t)st->st_ctim.tv_nsec;
}

static int sha256_file(const char *path, char out[65]) {
    FILE *f = fopen(path, "rb");
    EVP_MD_CTX *ctx = NULL;
    unsigned char buf[1u << 20], digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    int rc = -1;
    if (!f) return ec_fail("open identity input %s: %s", path, strerror(errno));
    ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) {
        ec_fail("initialize SHA-256 identity for %s", path); goto done;
    }
    for (;;) {
        size_t n = fread(buf, 1, sizeof buf, f);
        if (n && EVP_DigestUpdate(ctx, buf, n) != 1) { ec_fail("hash identity input %s", path); goto done; }
        if (n < sizeof buf) {
            if (ferror(f)) { ec_fail("read identity input %s: %s", path, strerror(errno)); goto done; }
            break;
        }
    }
    if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1 || digest_len != 32) {
        ec_fail("finish SHA-256 identity for %s", path); goto done;
    }
    for (unsigned i = 0; i < digest_len; i++) snprintf(out + 2u * i, 3, "%02x", digest[i]);
    out[64] = '\0';
    rc = 0;
done:
    if (ctx) EVP_MD_CTX_free(ctx);
    if (fclose(f) != 0 && rc == 0) rc = ec_fail("close identity input %s: %s", path, strerror(errno));
    return rc;
}

static int stat_version_equal(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_mode == b->st_mode && a->st_size == b->st_size &&
           a->st_mtim.tv_sec == b->st_mtim.tv_sec && a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
           a->st_ctim.tv_sec == b->st_ctim.tv_sec && a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
}

static int read_sha_receipt_snapshot(const char *path, char **contents_out,
                                     size_t *length_out, char sha256[65],
                                     struct stat *snapshot_stat_out) {
    int fd = -1;
    char *contents = NULL;
    struct stat before, after;
    size_t length, used = 0;
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    int rc = -1;

    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return ec_fail("open SHA receipt %s: %s", path, strerror(errno));
    if (fstat(fd, &before) < 0) { ec_fail("stat SHA receipt %s: %s", path, strerror(errno)); goto done; }
    if (!S_ISREG(before.st_mode) || before.st_size < 0 || (uint64_t)before.st_size > EC_MAX_RECEIPT_BYTES) {
        ec_fail("SHA receipt is not a regular file of at most %u bytes: %s",
                EC_MAX_RECEIPT_BYTES, path);
        goto done;
    }
    length = (size_t)before.st_size;
    contents = malloc(length + 1u);
    if (!contents) { ec_fail("allocate bounded SHA receipt snapshot"); goto done; }
    while (used < length) {
        ssize_t n = read(fd, contents + used, length - used);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { ec_fail("read SHA receipt %s: %s", path, strerror(errno)); goto done; }
        if (n == 0) { ec_fail("SHA receipt changed or ended during read: %s", path); goto done; }
        used += (size_t)n;
    }
    if (fstat(fd, &after) < 0) { ec_fail("restat SHA receipt %s: %s", path, strerror(errno)); goto done; }
    if (!stat_version_equal(&before, &after)) {
        ec_fail("SHA receipt metadata changed during bounded snapshot: %s", path);
        goto done;
    }
    if (EVP_Digest(contents, length, digest, &digest_length, EVP_sha256(), NULL) != 1 || digest_length != 32) {
        ec_fail("hash bounded SHA receipt snapshot: %s", path);
        goto done;
    }
    for (unsigned i = 0; i < digest_length; i++) snprintf(sha256 + 2u * i, 3, "%02x", digest[i]);
    sha256[64] = '\0';
    contents[length] = '\0';
    if (close(fd) < 0) { fd = -1; ec_fail("close SHA receipt %s: %s", path, strerror(errno)); goto done; }
    fd = -1;
    *contents_out = contents;
    *length_out = length;
    if (snapshot_stat_out) *snapshot_stat_out = after;
    contents = NULL;
    rc = 0;
done:
    if (fd >= 0) close(fd);
    free(contents);
    return rc;
}

static int identify_file(const char *path, file_identity_t *identity) {
    struct stat st;
    memset(identity, 0, sizeof *identity);
    if (strlen(path) >= sizeof identity->path) return ec_fail("identity path too long");
    strcpy(identity->path, path);
    if (stat(path, &st) < 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
        return ec_fail("invalid identity file %s: %s", path, strerror(errno));
    identity->size = (uint64_t)st.st_size;
    identity->mtime_ns = stat_mtime_ns(&st);
    return sha256_file(path, identity->sha256);
}

static int hex_sha256_valid(const char *s) {
    if (strlen(s) != 64) return 0;
    for (unsigned i = 0; i < 64; i++) {
        char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return 0;
    }
    return 1;
}

static void lower_hex_sha256(char out[65], const char *in) {
    for (unsigned i = 0; i < 64; i++) {
        char c = in[i];
        out[i] = c >= 'A' && c <= 'F' ? (char)(c - 'A' + 'a') : c;
    }
    out[64] = '\0';
}

static const char *path_basename_const(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static int decode_sha256sum_path(const char *src, int escaped, char **out) {
    size_t cap = strlen(src) + 1, used = 0;
    char *decoded = malloc(cap);
    if (!decoded) return ec_fail("allocate checkpoint receipt label");
    for (size_t i = 0; src[i]; i++) {
        if (escaped && src[i] == '\\') {
            if (!src[i + 1]) { free(decoded); return ec_fail("malformed escaped checkpoint label"); }
            char next = src[++i];
            if (next == '\\') decoded[used++] = '\\';
            else if (next == 'n') decoded[used++] = '\n';
            else if (next == 'r') decoded[used++] = '\r';
            else { free(decoded); return ec_fail("unsupported escape in checkpoint receipt label"); }
        } else decoded[used++] = src[i];
    }
    decoded[used] = '\0';
    *out = decoded;
    return 0;
}

static char *canonical_entry_path(const char *label, const char *manifest_path) {
    char *candidate, *canonical;
    if (label[0] == '/') return realpath(label, NULL);
    candidate = absolute_path(label);
    if (!candidate) return NULL;
    canonical = realpath(candidate, NULL);
    free(candidate);
    if (canonical) return canonical;
    {
        const char *slash = strrchr(manifest_path, '/');
        size_t dir_len = slash ? (size_t)(slash - manifest_path) : 1u;
        const char *dir = slash ? manifest_path : ".";
        size_t label_len = strlen(label);
        candidate = malloc(dir_len + 1u + label_len + 1u);
        if (!candidate) return NULL;
        memcpy(candidate, dir, dir_len);
        candidate[dir_len] = '/';
        memcpy(candidate + dir_len + 1u, label, label_len + 1u);
        canonical = realpath(candidate, NULL);
        free(candidate);
    }
    return canonical;
}

static int bind_ple_receipt(options_t *o) {
    char *canonical_ple = realpath(o->ple, NULL);
    size_t basename_matches = 0, exact_matches = 0;
    char declared[65] = {0}, manifest_sha[65] = {0};
    char label_copy[4096] = {0}, receipt_path[4096] = {0};
    struct stat ple_stat, ple_stat_after;
    if (!canonical_ple) return ec_fail("resolve live PLE shard path: %s", strerror(errno));
    if (stat(canonical_ple, &ple_stat) < 0 || !S_ISREG(ple_stat.st_mode) || ple_stat.st_size < 0) {
        free(canonical_ple);
        return ec_fail("stat live PLE shard: %s", strerror(errno));
    }
    if (o->weight_manifest_count == 0) { free(canonical_ple); return ec_fail("--weight-manifest must provide offline PLE SHA provenance"); }
    for (size_t m = 0; m < o->weight_manifest_count; m++) {
        char *snapshot = NULL;
        size_t snapshot_length = 0, offset = 0;
        char receipt_sha[65];
        if (read_sha_receipt_snapshot(o->weight_manifests[m], &snapshot,
                                      &snapshot_length, receipt_sha, NULL) < 0) {
            free(canonical_ple);
            return -1;
        }
        if (memchr(snapshot, '\0', snapshot_length)) {
            free(snapshot); free(canonical_ple);
            return ec_fail("weight manifest contains a NUL byte: %s", o->weight_manifests[m]);
        }
        while (offset < snapshot_length) {
            size_t end = offset, line_length;
            char *p, *label, digest[65];
            char *canonical_label;
            int escaped = 0;
            while (end < snapshot_length && snapshot[end] != '\n') end++;
            line_length = end - offset;
            if (line_length && snapshot[offset + line_length - 1u] == '\r') line_length--;
            snapshot[offset + line_length] = '\0';
            p = snapshot + offset;
            offset = end < snapshot_length ? end + 1u : snapshot_length;
            if (!line_length) continue;
            if (*p == '\\') { escaped = 1; p++; }
            if (strlen(p) < 65) {
                free(snapshot); free(canonical_ple);
                return ec_fail("malformed SHA-256 entry in %s", o->weight_manifests[m]);
            }
            memcpy(digest, p, 64); digest[64] = '\0';
            if (!hex_sha256_valid(digest)) {
                free(snapshot); free(canonical_ple);
                return ec_fail("malformed SHA-256 digest in %s", o->weight_manifests[m]);
            }
            p += 64;
            if (*p != ' ' && *p != '\t') {
                free(snapshot); free(canonical_ple);
                return ec_fail("missing filename in weight manifest %s", o->weight_manifests[m]);
            }
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '*') p++;
            if (!*p) {
                free(snapshot); free(canonical_ple);
                return ec_fail("invalid filename in weight manifest %s", o->weight_manifests[m]);
            }
            if (decode_sha256sum_path(p, escaped, &label) < 0) {
                free(snapshot); free(canonical_ple);
                return -1;
            }
            if (strcmp(path_basename_const(label), path_basename_const(canonical_ple))) { free(label); continue; }
            basename_matches++;
            canonical_label = canonical_entry_path(label, o->weight_manifests[m]);
            if (canonical_label && !strcmp(canonical_label, canonical_ple)) {
                exact_matches++;
                if (strlen(o->weight_manifests[m]) >= sizeof receipt_path || strlen(label) >= sizeof label_copy) {
                    free(canonical_label); free(label); free(snapshot); free(canonical_ple);
                    return ec_fail("PLE receipt path exceeds report limit");
                }
                strcpy(receipt_path, o->weight_manifests[m]);
                strcpy(label_copy, label);
                lower_hex_sha256(declared, digest);
                memcpy(manifest_sha, receipt_sha, sizeof manifest_sha);
            }
            free(canonical_label);
            free(label);
        }
        free(snapshot);
    }
    if (stat(canonical_ple, &ple_stat_after) < 0 || !S_ISREG(ple_stat_after.st_mode) || ple_stat_after.st_size < 0) {
        free(canonical_ple);
        return ec_fail("restat live PLE shard after receipt read: %s", strerror(errno));
    }
    if (!stat_version_equal(&ple_stat, &ple_stat_after)) {
        free(canonical_ple);
        return ec_fail("live PLE metadata changed while binding the offline receipt");
    }
    free(canonical_ple);
    if (basename_matches != 1 || exact_matches != 1)
        return ec_fail("PLE checkpoint must have one unique basename and exact-path entry in --weight-manifest");
    strcpy(o->ple_receipt_path, receipt_path);
    strcpy(o->ple_receipt_label, label_copy);
    strcpy(o->ple_receipt_sha256, manifest_sha);
    strcpy(o->ple_declared_sha256, declared);
    o->ple_live_size = (uint64_t)ple_stat_after.st_size;
    o->ple_live_mtime_ns = stat_mtime_ns(&ple_stat_after);
    o->ple_live_ctime_ns = stat_ctime_ns(&ple_stat_after);
    o->ple_live_device = (uint64_t)ple_stat_after.st_dev;
    o->ple_live_inode = (uint64_t)ple_stat_after.st_ino;
    return 0;
}

static int verify_ple_provenance(const options_t *o, const char *phase) {
    struct stat ple_stat;
    char *receipt_contents = NULL;
    size_t receipt_length = 0;
    char receipt_sha[65];
    if (read_sha_receipt_snapshot(o->ple_receipt_path, &receipt_contents,
                                  &receipt_length, receipt_sha, NULL) < 0) {
        char cause[sizeof ec_error];
        snprintf(cause, sizeof cause, "%s", ec_error);
        return ec_fail("%s PLE receipt recheck failed: %s", phase, cause);
    }
    free(receipt_contents);
    (void)receipt_length;
    if (strcmp(receipt_sha, o->ple_receipt_sha256))
        return ec_fail("%s PLE receipt SHA-256 changed after initial snapshot", phase);
    if (stat(o->ple, &ple_stat) < 0)
        return ec_fail("%s live PLE stat failed: %s", phase, strerror(errno));
    if (!S_ISREG(ple_stat.st_mode) || ple_stat.st_size < 0)
        return ec_fail("%s live PLE path is no longer a regular shard", phase);
    if ((uint64_t)ple_stat.st_size != o->ple_live_size ||
        stat_mtime_ns(&ple_stat) != o->ple_live_mtime_ns ||
        stat_ctime_ns(&ple_stat) != o->ple_live_ctime_ns ||
        (uint64_t)ple_stat.st_dev != o->ple_live_device ||
        (uint64_t)ple_stat.st_ino != o->ple_live_inode)
        return ec_fail("%s live PLE metadata differs from receipt-binding snapshot", phase);
    return 0;
}

static int build_plan(const options_t *o, request_t **plan_out, size_t *count_out) {
    size_t count = 2 + 3 * o->snapshot_count + 2;
    request_t *plan = calloc(count, sizeof *plan);
    size_t at = 0;
    if (!plan) return ec_fail("allocate request plan");
#define PLAN(NAME, PROMPT, RESUME, SNAP, NEWTOK) do { \
    snprintf(plan[at].name, sizeof plan[at].name, "%s", NAME); \
    plan[at].prompt = PROMPT; plan[at].resume = RESUME; \
    plan[at].snap = SNAP; plan[at].n_new = NEWTOK; at++; \
} while (0)
    PLAN("fresh-A", 'A', 0, 0, 0);
    PLAN("fresh-B", 'B', 0, 0, 0);
    for (size_t i = 0; i < o->snapshot_count; i++) {
        char name[64];
        snprintf(name, sizeof name, "save-%u-A", o->snapshots[i]); PLAN(name, 'A', 0, o->snapshots[i], 0);
        snprintf(name, sizeof name, "resume-%u-A", o->snapshots[i]); PLAN(name, 'A', o->snapshots[i], 0, 0);
        snprintf(name, sizeof name, "reset-%u-B", o->snapshots[i]); PLAN(name, 'B', 0, 0, 0);
    }
    PLAN("final-A", 'A', 0, 0, 0);
    PLAN("generation-A", 'G', 0, 0, 1);
#undef PLAN
    if (at != count) { free(plan); return ec_fail("internal request plan length mismatch"); }
    *plan_out = plan; *count_out = count;
    return 0;
}

static unsigned prompt_base(char prompt) {
    return prompt == 'B' ? 30u : 10u;
}

static int write_commands(const options_t *o, const request_t *plan, size_t count,
                          const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return ec_fail("create command input %s: %s", path, strerror(errno));
    if (fprintf(f, "config\ngen 0\n") < 0) goto io_error;
    for (size_t r = 0; r < count; r++) {
        if (plan[r].snap && fprintf(f, "snap %u\n", plan[r].snap) < 0) goto io_error;
        if (fprintf(f, "gen %u", plan[r].n_new) < 0) goto io_error;
        for (unsigned i = 0; i < o->tokens; i++) {
            unsigned id = (prompt_base(plan[r].prompt) + i) % EC_VOCAB;
            if (fprintf(f, " %u", id) < 0) goto io_error;
        }
        if (fprintf(f, "\nstatehash\n") < 0) goto io_error;
    }
    if (fprintf(f, "quit\n") < 0 || fflush(f) == EOF || fsync(fileno(f)) < 0) goto io_error;
    if (fclose(f) != 0) return ec_fail("close command input %s: %s", path, strerror(errno));
    return 0;
io_error:
    {
        int saved = errno ? errno : EIO;
        fclose(f);
        return ec_fail("write command input %s: %s", path, strerror(saved));
    }
}

static int write_all_json_string(FILE *f, const char *s) {
    if (fputc('"', f) == EOF) return -1;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': if (fputs("\\\"", f) == EOF) return -1; break;
        case '\\': if (fputs("\\\\", f) == EOF) return -1; break;
        case '\n': if (fputs("\\n", f) == EOF) return -1; break;
        case '\r': if (fputs("\\r", f) == EOF) return -1; break;
        case '\t': if (fputs("\\t", f) == EOF) return -1; break;
        default:
            if (*p < 0x20) { if (fprintf(f, "\\u%04x", *p) < 0) return -1; }
            else if (fputc(*p, f) == EOF) return -1;
        }
    }
    return fputc('"', f) == EOF ? -1 : 0;
}

static int write_report(const options_t *o, const run_result_t *runs, size_t nruns,
                        const file_identity_t *ids, size_t nids,
                        const char *status, const char *error) {
    char path[4096];
    FILE *f;
    if (path_join(path, sizeof path, o->out, "results.json") < 0) return -1;
    f = fopen(path, "wb");
    if (!f) return ec_fail("write report %s: %s", path, strerror(errno));
    if (fprintf(f, "{\n  \"schema\":\"flash-sass.issue9.engine-check.v1\",\n  \"status\":") < 0 ||
        write_all_json_string(f, status) < 0 ||
        fprintf(f, ",\n  \"scope\":\"full48-six-die GPU-vs-GPU request fingerprints\",\n  \"revision\":") < 0 ||
        write_all_json_string(f, o->revision) < 0 ||
        fprintf(f, ",\n  \"engine_stage_capture\":%s,\n  \"ple_path\":", o->capture_stages ? "true" : "false") < 0 || write_all_json_string(f, o->ple) < 0 ||
        fprintf(f, ",\n  \"ple_offset\":%u,\n  \"ple_hash_provenance\":\"declared from offline checkpoints.sha256; live shard not rehashed\",\n  \"ple_receipt_manifest\":",
                o->ple_off) < 0 || write_all_json_string(f, o->ple_receipt_path) < 0 ||
        fprintf(f, ",\n  \"ple_receipt_identity_sha256\":") < 0 ||
        write_all_json_string(f, o->ple_receipt_sha256) < 0 ||
        fprintf(f, ",\n  \"ple_receipt_label\":") < 0 ||
        write_all_json_string(f, o->ple_receipt_label) < 0 ||
        fprintf(f, ",\n  \"ple_declared_sha256\":") < 0 ||
        write_all_json_string(f, o->ple_declared_sha256) < 0 ||
        fprintf(f, ",\n  \"ple_live_stat\":{\"size\":%" PRIu64 ",\"mtime_ns\":%" PRIu64 ",\"ctime_ns\":%" PRIu64 ",\"device\":%" PRIu64 ",\"inode\":%" PRIu64 "},\n  \"ple_live_stat_scope\":\"size/mtime/ctime/device/inode sampled during receipt binding and compared to current stat before and after each child; current-vs-hash-time metadata only, not live shard content verification\",\n  \"tokens\":%u,\n  \"capacity\":%u,\n  \"layers\":48,\n  \"dies\":6,\n  \"FLASH_NBLK\":%u,\n  \"kv_hosts\":\"0,1\",\n  \"pair_modes\":\"scalar,chunk3\",\n  \"selection_exclusion_threshold\":%u,\n  \"selection_exclusion_condition_met\":%s,\n  \"cuda_visible_devices\":",
                o->ple_live_size, o->ple_live_mtime_ns, o->ple_live_ctime_ns, o->ple_live_device, o->ple_live_inode,
                o->tokens, o->maxt, o->budget,
                4u * (o->budget + 1u),
                o->tokens >= 4u * (o->budget + 1u) ? "true" : "false") < 0 ||
        write_all_json_string(f, o->visible_csv) < 0 ||
        fprintf(f, ",\n  \"split_shares\":") < 0 || write_all_json_string(f, o->shares_csv) < 0 ||
        fprintf(f, ",\n  \"attn_map\":") < 0 || write_all_json_string(f, o->attn_csv) < 0 ||
        fprintf(f, ",\n  \"snapshot_positions\":") < 0 || write_all_json_string(f, o->snapshots_csv) < 0 ||
        fprintf(f, ",\n  \"physical_dies\":[") < 0) goto io_error;
    for (unsigned d = 0; d < EC_DIES; d++) if (fprintf(f, "%s%u", d ? "," : "", o->physical[d]) < 0) goto io_error;
    if (fprintf(f, "],\n  \"holder_pids\":[") < 0) goto io_error;
    for (unsigned d = 0; d < EC_DIES; d++) if (fprintf(f, "%s%ld", d ? "," : "", (long)o->holder_pids[d]) < 0) goto io_error;
    if (fprintf(f, "],\n  \"inputs\":[") < 0) goto io_error;
    for (size_t i = 0; i < nids; i++) {
        if (i && fputc(',', f) == EOF) goto io_error;
        if (fprintf(f, "\n    {\"path\":") < 0 || write_all_json_string(f, ids[i].path) < 0 ||
            fprintf(f, ",\"size\":%" PRIu64 ",\"mtime_ns\":%" PRIu64 ",\"sha256\":", ids[i].size, ids[i].mtime_ns) < 0 ||
            write_all_json_string(f, ids[i].sha256) < 0 || fprintf(f, "}") < 0) goto io_error;
    }
    if (fprintf(f, "\n  ],\n  \"runs\":[") < 0) goto io_error;
    for (size_t i = 0; i < nruns; i++) {
        const run_result_t *r = &runs[i];
        if (i && fputc(',', f) == EOF) goto io_error;
        if (fprintf(f, "\n    {\"name\":") < 0 || write_all_json_string(f, r->name) < 0 ||
            fprintf(f, ",\"mode\":") < 0 || write_all_json_string(f, r->mode) < 0 ||
            fprintf(f, ",\"kv_host\":%u,\"pid\":%ld,\"exit_code\":%d,\"signal\":%d,\"timed_out\":%s,\"interrupted\":%s,\"elapsed_ms\":%u,\"passed\":%s,\"requests_checked\":%zu,\"stdin\":",
                    r->kv_host, (long)r->child_pid, r->exit_code, r->signal_number,
                    r->timed_out ? "true" : "false", r->interrupted ? "true" : "false",
                    r->elapsed_ms, r->passed ? "true" : "false", r->requests_checked) < 0 ||
            write_all_json_string(f, r->stdin_path) < 0 || fprintf(f, ",\"stdout\":") < 0 ||
            write_all_json_string(f, r->stdout_path) < 0 || fprintf(f, ",\"stderr\":") < 0 ||
            write_all_json_string(f, r->stderr_path) < 0 || fprintf(f, "}") < 0) goto io_error;
    }
    if (fprintf(f, "\n  ],\n  \"error\":") < 0 || write_all_json_string(f, error ? error : "") < 0 ||
        fprintf(f, "\n}\n") < 0 || fflush(f) == EOF || fsync(fileno(f)) < 0) goto io_error;
    if (fclose(f) != 0) return ec_fail("close report %s: %s", path, strerror(errno));
    return 0;
io_error:
    {
        int saved = errno ? errno : EIO;
        fclose(f);
        return ec_fail("write report %s: %s", path, strerror(saved));
    }
}

static int remove_inherited_engine_overrides(void) {
    for (;;) {
        char *key = NULL;
        for (char **entry = environ; entry && *entry; entry++) {
            const char *value = *entry;
            if (strncmp(value, "FLASH_", 6) && strncmp(value, "MODEL_", 6)) continue;
            const char *equal = strchr(value, '=');
            if (!equal) continue;
            size_t n = (size_t)(equal - value);
            key = malloc(n + 1);
            if (!key) return ec_fail("allocate environment key");
            memcpy(key, value, n); key[n] = '\0';
            break;
        }
        if (!key) return 0;
        int rc = unsetenv(key);
        int saved = errno;
        free(key);
        if (rc < 0) return ec_fail("remove inherited engine setting: %s", strerror(saved));
    }
}

static void stop_private_core(pid_t child, int *wait_status) {
    pid_t got;
    if (kill(child, SIGTERM) < 0 && errno == ESRCH) {
        while (waitpid(child, wait_status, 0) < 0 && errno == EINTR) {}
        return;
    }
    double grace = monotime() + 2.0;
    for (;;) {
        got = waitpid(child, wait_status, WNOHANG);
        if (got == child) return;
        if (got < 0 && errno != EINTR) break;
        if (monotime() >= grace) break;
        struct timespec nap = {0, 50000000}; nanosleep(&nap, NULL);
    }
    if (kill(child, SIGKILL) < 0 && errno != ESRCH) return;
    while (waitpid(child, wait_status, 0) < 0 && errno == EINTR) {}
}

static int run_private_core(const options_t *o, const char *mode, unsigned kv_host,
                            const char *input_path, const char *stdout_path,
                            const char *stderr_path, double deadline,
                            run_result_t *result) {
    char core_path[4096], engine_path[4096], cwd[4096];
    char max_text[32], nblk_text[32], pleoff_text[32], kv_text[8], pair_text[8], chunk_text[8];
    char *child_argv[EC_DIES + 5];
    pid_t child;
    int wait_status = 0, timed_out = 0;
    double started = monotime();
    if (path_join(cwd, sizeof cwd, o->repo, "kern") < 0 ||
        path_join(core_path, sizeof core_path, cwd, "flashcore") < 0 ||
        path_join(engine_path, sizeof engine_path, cwd, "libengine.so") < 0)
        return -1;
    if (access(core_path, X_OK) < 0) return ec_fail("flashcore is not executable: %s", core_path);
    if (access(engine_path, R_OK) < 0) return ec_fail("engine library is unreadable: %s", engine_path);
    snprintf(max_text, sizeof max_text, "%u", o->maxt);
    snprintf(nblk_text, sizeof nblk_text, "%u", o->budget);
    snprintf(pleoff_text, sizeof pleoff_text, "%u", o->ple_off);
    snprintf(kv_text, sizeof kv_text, "%u", kv_host);
    snprintf(pair_text, sizeof pair_text, "%d", !strcmp(mode, "chunk3"));
    snprintf(chunk_text, sizeof chunk_text, "%d", !strcmp(mode, "chunk3") ? 3 : 0);
    child_argv[0] = core_path;
    child_argv[1] = o->ple;
    child_argv[2] = engine_path;
    for (unsigned d = 0; d < EC_DIES; d++) child_argv[3 + d] = o->manifests[d];
    child_argv[3 + EC_DIES] = NULL;
    child = fork();
    if (child < 0) return ec_fail("fork private flashcore: %s", strerror(errno));
    if (child == 0) {
        int in_fd, out_fd, err_fd;
        struct sigaction sa;
        memset(&sa, 0, sizeof sa); sa.sa_handler = SIG_DFL; sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL); sigaction(SIGHUP, &sa, NULL);
        close(o->lock_fd);
        if (chdir(cwd) < 0) _exit(126);
        in_fd = open(input_path, O_RDONLY);
        out_fd = open(stdout_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
        err_fd = open(stderr_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (in_fd < 0 || out_fd < 0 || err_fd < 0) _exit(126);
        if (dup2(in_fd, STDIN_FILENO) < 0 || dup2(out_fd, STDOUT_FILENO) < 0 ||
            dup2(err_fd, STDERR_FILENO) < 0) _exit(126);
        if (in_fd != STDIN_FILENO) close(in_fd);
        if (out_fd != STDOUT_FILENO) close(out_fd);
        if (err_fd != STDERR_FILENO) close(err_fd);
        if (remove_inherited_engine_overrides() < 0) _exit(126);
        if (setenv("FLASH_ENGINE_CAPTURE", o->capture_stages ? "1" : "0", 1)) _exit(126);
        if ((o->ld_library_path && setenv("LD_LIBRARY_PATH", o->ld_library_path, 1)) ||
            (o->cuda_device_order && setenv("CUDA_DEVICE_ORDER", o->cuda_device_order, 1)) ||
            setenv("CUDA_VISIBLE_DEVICES", o->visible_csv, 1) ||
            setenv("SPLIT_SHARES", o->shares_csv, 1) || setenv("ATTN_MAP", o->attn_csv, 1) ||
            setenv("FLASH_MAXT", max_text, 1) || setenv("FLASH_ENGINE_MAXT", max_text, 1) ||
            setenv("FLASH_LAYERS", "48", 1) || setenv("FLASH_NBLK", nblk_text, 1) ||
            setenv("FLASH_MAXIMG", "0", 1) || setenv("FLASH_PLE_DISK", "1", 1) ||
            setenv("FLASH_PLE_OFF", pleoff_text, 1) || setenv("FLASH_KV_HOST", kv_text, 1) ||
            setenv("FLASH_TWO_POSITION", pair_text, 1) || setenv("FLASH_PREFILL_CHUNK", chunk_text, 1)) _exit(126);
        execv(core_path, child_argv);
        _exit(127);
    }
    result->child_pid = child;
    for (;;) {
        pid_t got = waitpid(child, &wait_status, WNOHANG);
        if (got == child) break;
        if (got < 0 && errno != EINTR) {
            int saved = errno;
            stop_private_core(child, &wait_status);
            if (WIFSIGNALED(wait_status)) result->signal_number = WTERMSIG(wait_status);
            return ec_fail("wait for private flashcore: %s", strerror(saved));
        }
        if (ec_interrupted || monotime() >= deadline) {
            timed_out = !ec_interrupted;
            result->timed_out = timed_out;
            result->interrupted = ec_interrupted != 0;
            stop_private_core(child, &wait_status);
            if (WIFSIGNALED(wait_status)) result->signal_number = WTERMSIG(wait_status);
            else if (WIFEXITED(wait_status)) result->exit_code = WEXITSTATUS(wait_status);
            result->elapsed_ms = (unsigned)((monotime() - started) * 1000.0);
            if (timed_out) return ec_fail("private flashcore exceeded run deadline");
            return ec_fail("private flashcore interrupted");
        }
        struct timespec nap = {0, 50000000}; nanosleep(&nap, NULL);
    }
    result->elapsed_ms = (unsigned)((monotime() - started) * 1000.0);
    if (WIFEXITED(wait_status)) result->exit_code = WEXITSTATUS(wait_status);
    else if (WIFSIGNALED(wait_status)) result->signal_number = WTERMSIG(wait_status);
    if (result->exit_code || result->signal_number)
        return ec_fail("private core %s exited code=%d signal=%d", result->name,
                       result->exit_code, result->signal_number);
    return 0;
}

static int add_identity(file_identity_t *ids, size_t capacity, size_t *count,
                        const char *path) {
    if (*count >= capacity) return ec_fail("too many source identities");
    if (identify_file(path, &ids[*count]) < 0) return -1;
    (*count)++;
    return 0;
}

static int add_weight_manifest_identity(file_identity_t *ids, size_t capacity,
                                        size_t *count, const char *path) {
    file_identity_t *identity;
    char *snapshot = NULL;
    size_t length = 0;
    struct stat st;
    if (*count >= capacity) return ec_fail("too many source identities");
    if (strlen(path) >= sizeof ids[*count].path) return ec_fail("identity path too long");
    identity = &ids[*count];
    memset(identity, 0, sizeof *identity);
    if (read_sha_receipt_snapshot(path, &snapshot, &length, identity->sha256, &st) < 0) return -1;
    free(snapshot);
    (void)length;
    strcpy(identity->path, path);
    identity->size = (uint64_t)st.st_size;
    identity->mtime_ns = stat_mtime_ns(&st);
    (*count)++;
    return 0;
}

static int collect_identities(const options_t *o, file_identity_t *ids,
                              size_t capacity, size_t *count) {
    char path[4096];
    *count = 0;
    if (path_join(path, sizeof path, o->repo, "kern/flashcore") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "kern/libengine.so") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "kern/die.cubin") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    for (unsigned d = 0; d < EC_DIES; d++) if (add_identity(ids, capacity, count, o->manifests[d]) < 0) return -1;
    for (size_t i = 0; i < o->weight_manifest_count; i++)
        if (add_weight_manifest_identity(ids, capacity, count, o->weight_manifests[i]) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "kern/engine.c") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "kern/flashcore.c") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "kern/prog.c") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    if (path_join(path, sizeof path, o->repo, "validation/issue9/engine_check.c") < 0 || add_identity(ids, capacity, count, path) < 0) return -1;
    return 0;
}

static int verify_collected_receipt_identity(const options_t *o,
                                             const file_identity_t *ids,
                                             size_t count) {
    const file_identity_t *receipt = NULL;
    size_t matches = 0;
    for (size_t i = 0; i < count; i++) {
        if (!strcmp(ids[i].path, o->ple_receipt_path)) {
            receipt = &ids[i];
            matches++;
        }
    }
    if (matches != 1 || !receipt)
        return ec_fail("selected PLE receipt has no unique collect_identities record");
    if (strcmp(receipt->sha256, o->ple_receipt_sha256))
        return ec_fail("selected PLE receipt SHA differs from collect_identities snapshot");
    return 0;
}

static int make_run_paths(const options_t *o, run_result_t *run) {
    char leaf[128];
    snprintf(leaf, sizeof leaf, "%s.stdin", run->name);
    if (path_join(run->stdin_path, sizeof run->stdin_path, o->out, leaf) < 0) return -1;
    snprintf(leaf, sizeof leaf, "%s.stdout", run->name);
    if (path_join(run->stdout_path, sizeof run->stdout_path, o->out, leaf) < 0) return -1;
    snprintf(leaf, sizeof leaf, "%s.stderr", run->name);
    if (path_join(run->stderr_path, sizeof run->stderr_path, o->out, leaf) < 0) return -1;
    return 0;
}

static void options_free(options_t *o) {
    free(o->repo); free(o->out); free(o->ple); free(o->revision);
    free(o->visible_csv); free(o->shares_csv); free(o->attn_csv);
    free(o->kv_csv); free(o->snapshots_csv);
    free(o->ld_library_path); free(o->cuda_device_order);
    for (unsigned d = 0; d < EC_DIES; d++) free(o->manifests[d]);
    for (size_t i = 0; i < o->weight_manifest_count; i++) free(o->weight_manifests[i]);
}

static void references_free(reference_t references[3]) {
    for (unsigned i = 0; i < 3; i++) free(references[i].tokens);
}

static int parse_run_output(const options_t *o, const request_t *plan, size_t plan_count,
                            const char *mode, unsigned kv_host,
                            const char *stdout_path, reference_t references[3],
                            run_result_t *run);
static int engine_check_self_test(void);

int main(int argc, char **argv) {
    options_t o;
    request_t *plan = NULL;
    size_t plan_count = 0;
    reference_t references[3] = {{0}};
    run_result_t runs[EC_MAX_RUNS] = {0};
    file_identity_t identities[32];
    size_t identity_count = 0, run_count = 0;
    char command_path[4096] = {0};
    char result_path[4096] = {0};
    char status[16] = "FAILED";
    double deadline;
    int output_created = 0;
    int rc = 1;
    struct sigaction sa;
    if (argc == 2 && !strcmp(argv[1], "--self-test")) return engine_check_self_test();
    if (parse_options(argc, argv, &o) < 0) {
        fprintf(stderr, "engine_check: %s\n", ec_error);
        usage(stderr, argv[0]);
        options_free(&o);
        return 2;
    }
    memset(&sa, 0, sizeof sa); sa.sa_handler = ec_signal; sigemptyset(&sa.sa_mask);
    if (sigaction(SIGINT, &sa, NULL) < 0 || sigaction(SIGTERM, &sa, NULL) < 0 || sigaction(SIGHUP, &sa, NULL) < 0) {
        ec_fail("install signal handlers: %s", strerror(errno)); goto done;
    }
    if (make_absolute_options(&o) < 0 || bind_ple_receipt(&o) < 0 || check_resource_floors(&o) < 0 ||
        validate_process_inputs(&o) < 0) goto done;
    if (directory_is_new(o.out) < 0) goto done;
    output_created = 1;
    if (path_join(command_path, sizeof command_path, o.out, "commands.txt") < 0 ||
        path_join(result_path, sizeof result_path, o.out, "results.json") < 0) goto done;
    if (collect_identities(&o, identities, sizeof identities / sizeof identities[0], &identity_count) < 0 ||
        verify_collected_receipt_identity(&o, identities, identity_count) < 0 ||
        build_plan(&o, &plan, &plan_count) < 0) goto done;
    if (write_commands(&o, plan, plan_count, command_path) < 0 ||
        add_identity(identities, sizeof identities / sizeof identities[0], &identity_count, command_path) < 0) goto done;
    if (write_report(&o, runs, 0, identities, identity_count, "RUNNING", "") < 0) goto done;
    deadline = monotime() + (double)o.timeout_seconds;
    for (unsigned kv = 0; kv < 2; kv++) {
        for (unsigned mode_index = 0; mode_index < 2; mode_index++) {
            if (ec_interrupted) { ec_fail("engine request runner interrupted"); goto done; }
            run_result_t *run = &runs[run_count];
            const char *mode = mode_index ? "chunk3" : "scalar";
            memset(run, 0, sizeof *run);
            snprintf(run->name, sizeof run->name, "kv%u-%s", kv, mode);
            snprintf(run->mode, sizeof run->mode, "%s", mode);
            run->kv_host = kv;
            if (make_run_paths(&o, run) < 0 ||
                write_commands(&o, plan, plan_count, run->stdin_path) < 0) goto done;
            ec_error[0] = '\0';
            if (verify_ple_provenance(&o, "before child") < 0) {
                run_count++;
                goto done;
            }
            {
                int child_rc, after_rc;
                char child_error[sizeof ec_error] = {0};
                char after_error[sizeof ec_error] = {0};
                child_rc = run_private_core(&o, mode, kv, run->stdin_path, run->stdout_path,
                                            run->stderr_path, deadline, run);
                if (child_rc < 0) snprintf(child_error, sizeof child_error, "%s", ec_error);
                after_rc = verify_ple_provenance(&o, "after child");
                if (after_rc < 0) snprintf(after_error, sizeof after_error, "%s", ec_error);
                if (child_rc < 0 || after_rc < 0) {
                    if (child_rc < 0 && after_rc < 0)
                        ec_fail("%s; %s", child_error, after_error);
                    else if (child_rc < 0)
                        snprintf(ec_error, sizeof ec_error, "%s", child_error);
                    run_count++;
                    goto done;
                }
            }
            if (ec_interrupted) { ec_fail("engine request runner interrupted"); run_count++; goto done; }
            if (parse_run_output(&o, plan, plan_count, mode, kv, run->stdout_path,
                                 references, run) < 0) {
                run_count++;
                goto done;
            }
            run_count++;
            if (write_report(&o, runs, run_count, identities, identity_count, "RUNNING", "") < 0) goto done;
        }
    }
    if (ec_interrupted) { ec_fail("engine request runner interrupted"); goto done; }
    snprintf(status, sizeof status, "PASSED");
    if (write_report(&o, runs, run_count, identities, identity_count, status, "") < 0) goto done;
    printf("ENGINE_CHECK_PASSED results=%s\n", result_path);
    rc = 0;
done:
    if (rc && output_created) {
        const char *message = ec_error[0] ? ec_error : "engine request validation failed";
        write_report(&o, runs, run_count, identities, identity_count, "FAILED", message);
        fprintf(stderr, "engine_check: %s\n", message);
    } else if (rc) {
        fprintf(stderr, "engine_check: %s\n", ec_error[0] ? ec_error : "initialization failed");
    }
    if (!rc) printf("engine_check: checked %zu requests across %zu private cores\n", plan_count * run_count, run_count);
    free(plan);
    references_free(references);
    options_free(&o);
    return rc;
}

static void trim_line(char *line) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
}

static int line_is_error(const char *line) {
    return !strncmp(line, "error:", 6) || strstr(line, "MISMATCH") != NULL;
}

static int config_matches(const options_t *o, const char *mode, unsigned kv, const char *line) {
    char *copy = string_dup(line + 7), *save = NULL;
    char seen_names[64][32];
    size_t seen_count = 0;
    int found_rev = 0, found_kv = 0, found_disk = 0, found_maxt = 0;
    int found_layers = 0, found_nblk = 0, found_dies = 0, found_split = 0;
    int found_map = 0, found_attn = 0, found_pair = 0, found_chunk = 0;
    int found_capture = !o->capture_stages;
    unsigned active[EC_DIES] = {0};
    char attn_dies[64] = {0};
    size_t used = 0;
    if (!copy) return ec_fail("copy config response");
    for (unsigned i = 0; i < EC_ATTN; i++) active[o->attn_map[i]] = 1;
    for (unsigned d = 0; d < EC_DIES; d++) if (active[d]) {
        int n = snprintf(attn_dies + used, sizeof attn_dies - used, "%s%u", used ? "," : "", d);
        if (n < 0 || (size_t)n >= sizeof attn_dies - used) { free(copy); return ec_fail("format active attention dies"); }
        used += (size_t)n;
    }
    for (char *field = strtok_r(copy, " \t", &save); field; field = strtok_r(NULL, " \t", &save)) {
        char *eq = strchr(field, '=');
        if (!eq) { free(copy); return ec_fail("malformed config field"); }
        *eq++ = '\0';
        if (strlen(field) >= sizeof seen_names[0] || seen_count >= 64) {
            free(copy); return ec_fail("config field name is too long or too numerous");
        }
        for (size_t i = 0; i < seen_count; i++) if (!strcmp(seen_names[i], field)) {
            ec_fail("duplicate config field %s", field);
            free(copy);
            return -1;
        }
        strcpy(seen_names[seen_count++], field);
        if (!strcmp(field, "rev")) { found_rev = !strcmp(eq, o->revision); }
        else if (!strcmp(field, "kv_host")) { char expected[8]; snprintf(expected, sizeof expected, "%u", kv); found_kv = !strcmp(eq, expected); }
        else if (!strcmp(field, "ple_disk")) found_disk = !strcmp(eq, "1");
        else if (!strcmp(field, "maxt")) { char expected[32]; snprintf(expected, sizeof expected, "%u", o->maxt); found_maxt = !strcmp(eq, expected); }
        else if (!strcmp(field, "layers")) found_layers = !strcmp(eq, "48/48");
        else if (!strcmp(field, "nblk")) { char expected[32]; snprintf(expected, sizeof expected, "%u", o->budget); found_nblk = !strcmp(eq, expected); }
        else if (!strcmp(field, "dies")) found_dies = !strcmp(eq, "6");
        else if (!strcmp(field, "split")) found_split = !strcmp(eq, o->shares_csv);
        else if (!strcmp(field, "attn_map")) found_map = !strcmp(eq, o->attn_csv);
        else if (!strcmp(field, "attn_dies")) found_attn = !strcmp(eq, attn_dies);
        else if (!strcmp(field, "two_position")) { char expected[8]; snprintf(expected, sizeof expected, "%d", !strcmp(mode, "chunk3")); found_pair = !strcmp(eq, expected); }
        else if (!strcmp(field, "prefill_chunk")) { char expected[8]; snprintf(expected, sizeof expected, "%d", !strcmp(mode, "chunk3") ? 3 : 0); found_chunk = !strcmp(eq, expected); }
        else if (!strcmp(field, "engine_capture")) found_capture = !strcmp(eq, o->capture_stages ? "1" : "0");
        else continue;
    }
    free(copy);
    if (!(found_rev && found_kv && found_disk && found_maxt && found_layers && found_nblk &&
          found_dies && found_split && found_map && found_attn && found_pair && found_chunk && found_capture))
        return ec_fail("effective engine config differs from required full48/six-die request scope");
    return 0;
}

static int config_fixture(char *out, size_t out_cap,
                          const char *revision_fields, const char *kv_field,
                          const char *layers_field, const char *dies_field,
                          const char *attn_dies_field, const char *shares_field,
                          const char *attn_map_field, const char *pair_field,
                          const char *chunk_field) {
    int n = snprintf(out, out_cap,
                     "config %s %s ple_disk=1 maxt=2176 %s nblk=512 %s %s split=%s attn_map=%s %s %s",
                     revision_fields, kv_field, layers_field, dies_field,
                     attn_dies_field, shares_field, attn_map_field,
                     pair_field, chunk_field);
    return n >= 0 && (size_t)n < out_cap ? 0 : -1;
}

static int config_fixture_check(const options_t *o, const char *name,
                                const char *mode, unsigned kv,
                                const char *line, int should_accept) {
    int rc;
    ec_error[0] = '\0';
    rc = config_matches(o, mode, kv, line);
    if ((should_accept && rc == 0) || (!should_accept && rc < 0)) {
        printf("PASS %s\n", name);
        ec_error[0] = '\0';
        return 0;
    }
    fprintf(stderr, "FAIL %s: expected %s, got %s (%s)\n", name,
            should_accept ? "accept" : "reject", rc == 0 ? "accept" : "reject",
            ec_error[0] ? ec_error : "no parser error");
    ec_error[0] = '\0';
    return -1;
}

static int engine_check_self_test(void) {
    static const char revision40[] = "0123456789abcdef0123456789abcdef01234567";
    static const char *const shares = "112,97,81,116,97,97";
    static const char *const attn = "1,1,1,1,4,4,4,4,5,5,5,5";
    options_t fixture = {0};
    char line[2048];
    char revision_field[64];
    unsigned checks = 0;
    int failed = 0;
    fixture.revision = (char *)revision40;
    fixture.maxt = 2176;
    fixture.budget = 512;
    fixture.shares_csv = (char *)shares;
    fixture.attn_csv = (char *)attn;
    {
        static const unsigned owners[EC_ATTN] = {1, 1, 1, 1, 4, 4, 4, 4, 5, 5, 5, 5};
        memcpy(fixture.attn_map, owners, sizeof owners);
    }
    snprintf(revision_field, sizeof revision_field, "rev=%s", revision40);
#define FIXTURE(REV, KV, LAYERS, DIES, AD, SHARES, ATTN, PAIR, CHUNK) \
    config_fixture(line, sizeof line, REV, KV, LAYERS, DIES, AD, SHARES, ATTN, PAIR, CHUNK)
#define CHECK(NAME, MODE, KV, ACCEPT) do { \
    if (FIXTURE(revision_field, "kv_host=0", "layers=48/48", "dies=6", \
                "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0") < 0 || \
        config_fixture_check(&fixture, NAME, MODE, KV, line, ACCEPT) < 0) failed++; \
    checks++; \
} while (0)
    CHECK("full 40-hex revision accepted", "scalar", 0, 1);
    CHECK("scalar pair/chunk flags accepted", "scalar", 0, 1);

    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567 rev=deadbeef",
            "kv_host=0", "layers=48/48", "dies=6", "attn_dies=1,4,5",
            shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "contradictory duplicate revision, good first",
                                   "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=deadbeef rev=0123456789abcdef0123456789abcdef01234567",
            "kv_host=0", "layers=48/48", "dies=6", "attn_dies=1,4,5",
            shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "contradictory duplicate revision, bad first",
                                   "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567 rev=0123456789abcdef0123456789abcdef01234567",
            "kv_host=0", "layers=48/48", "dies=6", "attn_dies=1,4,5",
            shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "duplicate identical revision key rejected",
                                   "scalar", 0, line, 0) < 0; checks++;

    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "missing layers rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=47/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched layers rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "missing dies rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=5",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched die count rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", "", attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "missing split topology rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", "112,97,81,116,96,98", attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched split topology rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, "", "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "missing attention map rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, "1,1,1,1,4,4,4,4,5,5,5,4", "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched attention map rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched attention die set rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=1", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "mismatched KV host rejected", "scalar", 0, line, 0) < 0; checks++;

    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=1", "prefill_chunk=0");
    failed += config_fixture_check(&fixture, "scalar paired flag mismatch rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=3");
    failed += config_fixture_check(&fixture, "scalar chunk flag mismatch rejected", "scalar", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=1", "prefill_chunk=3");
    failed += config_fixture_check(&fixture, "chunk3 paired/chunk flags accepted", "chunk3", 0, line, 1) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=3");
    failed += config_fixture_check(&fixture, "chunk3 paired flag mismatch rejected", "chunk3", 0, line, 0) < 0; checks++;
    FIXTURE("rev=0123456789abcdef0123456789abcdef01234567", "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=1", "prefill_chunk=2");
    failed += config_fixture_check(&fixture, "chunk3 chunk-size mismatch rejected", "chunk3", 0, line, 0) < 0; checks++;
    FIXTURE(revision_field, "kv_host=0", "layers=48/48", "dies=6",
            "attn_dies=1,4,5", shares, attn, "two_position=0", "prefill_chunk=0");
    fixture.capture_stages = 1;
    failed += config_fixture_check(&fixture, "capture missing effective flag rejected", "scalar", 0, line, 0) < 0; checks++;
    size_t capture_at = strlen(line);
    snprintf(line + capture_at, sizeof line - capture_at, " engine_capture=0");
    failed += config_fixture_check(&fixture, "capture disabled effective flag rejected", "scalar", 0, line, 0) < 0; checks++;
    snprintf(line + capture_at, sizeof line - capture_at, " engine_capture=1");
    failed += config_fixture_check(&fixture, "capture enabled effective flag accepted", "scalar", 0, line, 1) < 0; checks++;
    fixture.capture_stages = 0;
    failed += config_fixture_check(&fixture, "unexpected enabled capture rejected", "scalar", 0, line, 0) < 0; checks++;
#undef CHECK
#undef FIXTURE
    if (failed) {
        fprintf(stderr, "ENGINE_CHECK_SELF_TEST_FAILED checks=%u failures=%d scope=config-parser-only gpu=not-run\n",
                checks, failed);
        return 1;
    }
    printf("ENGINE_CHECK_SELF_TEST_PASSED checks=%u scope=config-parser-only gpu=not-run numerical=not-claimed\n", checks);
    return 0;
}

static int parse_tokens_line(const char *line, size_t expected_count, uint32_t **tokens_out) {
    char *copy, *save = NULL, *p;
    uint32_t *tokens;
    size_t count = 0;
    if (strncmp(line, "tokens:", 7)) return ec_fail("expected tokens response, got: %.100s", line);
    copy = string_dup(line + 7);
    tokens = calloc(expected_count ? expected_count : 1, sizeof *tokens);
    if (!copy || !tokens) { free(copy); free(tokens); return ec_fail("allocate token response"); }
    for (p = strtok_r(copy, " \t", &save); p; p = strtok_r(NULL, " \t", &save)) {
        unsigned token;
        if (count >= expected_count || parse_uint(p, EC_VOCAB - 1u, &token) < 0) {
            free(copy); free(tokens); return ec_fail("invalid or excess token in response");
        }
        tokens[count++] = token;
    }
    free(copy);
    if (count != expected_count) { free(tokens); return ec_fail("received %zu tokens, expected %zu", count, expected_count); }
    *tokens_out = tokens;
    return 0;
}

typedef struct {
    double run_ms;
    int positions_run;
    int resumed;
    int paired_positions;
    int prefill_chunks;
} time_result_t;

static int parse_time_line(const char *line, time_result_t *time_result) {
    double per_position, first_token, programs;
    unsigned phases;
    int parsed;
    parsed = sscanf(line,
        "time: %lf ms for %d positions run (%d from the snapshot; %lf ms per position run; first token %lf ms; %u phases; programs %lf ms; %d paired positions; %d prefill chunks)",
        &time_result->run_ms, &time_result->positions_run, &time_result->resumed,
        &per_position, &first_token, &phases, &programs,
        &time_result->paired_positions, &time_result->prefill_chunks);
    if (parsed != 9) return ec_fail("incomplete or malformed generation timing response");
    (void)per_position; (void)first_token; (void)programs; (void)phases;
    return 0;
}

static int parse_statehash_header(const char *line, int *positions, int *paired, int *chunks) {
    if (sscanf(line, "statehash positions=%d paired_positions=%d prefill_chunks=%d", positions, paired, chunks) != 3)
        return ec_fail("invalid statehash header");
    return 0;
}

static int trailing_space_only(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return !*p;
}

static void lower_hash(char dst[17], const char *src) {
    for (unsigned i = 0; i < 16; i++) {
        char c = src[i];
        dst[i] = c >= 'A' && c <= 'F' ? (char)(c - 'A' + 'a') : c;
    }
    dst[16] = '\0';
}

static int parse_statehash_line(const options_t *o, const char *line, fingerprint_t *fp) {
    unsigned die, layer;
    char state[17], ple[17], logits[17];
    int used = 0;
    if (sscanf(line, "statehash die=%u layer=%u state=%16[0-9a-fA-F]%n", &die, &layer, state, &used) == 3) {
        if (!trailing_space_only(line + used) || die >= EC_DIES || layer >= EC_LAYERS || strlen(state) != 16)
            return ec_fail("invalid layer statehash record");
        if (layer % 4u == 3u && o->attn_map[layer / 4u] != die)
            return ec_fail("attention layer %u reported by nonowner die %u", layer, die);
        if (fp->layer_present[die][layer]) return ec_fail("duplicate statehash die=%u layer=%u", die, layer);
        lower_hash(fp->layers[die][layer], state);
        fp->layer_present[die][layer] = 1;
        return 1;
    }
    used = 0;
    if (sscanf(line, "statehash die=%u ple=%16[0-9a-fA-F] logits=%16[0-9a-fA-F]%n", &die, ple, logits, &used) == 3) {
        if (!trailing_space_only(line + used) || die >= EC_DIES || strlen(ple) != 16 || strlen(logits) != 16)
            return ec_fail("invalid die PLE/logit statehash record");
        if (fp->die_present[die]) return ec_fail("duplicate PLE/logit statehash for die %u", die);
        lower_hash(fp->ple[die], ple);
        lower_hash(fp->logits[die], logits);
        fp->die_present[die] = 1;
        return 1;
    }
    return ec_fail("unrecognized statehash response: %.120s", line);
}

static int fingerprint_complete(const options_t *o, const fingerprint_t *fp) {
    for (unsigned d = 0; d < EC_DIES; d++) {
        if (!fp->die_present[d]) return ec_fail("statehash lacks PLE/logits for die %u", d);
        for (unsigned layer = 0; layer < EC_LAYERS; layer++) {
            int owned = layer % 4u != 3u || o->attn_map[layer / 4u] == d;
            if ((int)fp->layer_present[d][layer] != owned)
                return ec_fail("statehash layer coverage mismatch for die %u layer %u", d, layer);
        }
    }
    return 0;
}

static int fingerprint_equal(const fingerprint_t *a, const fingerprint_t *b) {
    for (unsigned d = 0; d < EC_DIES; d++) {
        if (strcmp(a->ple[d], b->ple[d]) || strcmp(a->logits[d], b->logits[d])) return 0;
        for (unsigned l = 0; l < EC_LAYERS; l++) {
            if (a->layer_present[d][l] != b->layer_present[d][l]) return 0;
            if (a->layer_present[d][l] && strcmp(a->layers[d][l], b->layers[d][l])) return 0;
        }
    }
    return 1;
}

static int expected_pairs(const options_t *o, const request_t *request, int chunk_mode) {
    unsigned prefix = o->tokens - 1u;
    if (!chunk_mode) return 0;
    if (request->n_new) return (int)(2u * (prefix / 3u + (prefix % 3u) / 2u));
    if (request->snap)
        return (int)(2u * ((request->snap - request->resume) / 2u) +
                     2u * ((prefix - request->snap) / 2u));
    return (int)(2u * ((prefix - request->resume) / 2u));
}

static int expected_chunks(const options_t *o, const request_t *request, int chunk_mode) {
    if (!chunk_mode || !request->n_new) return 0;
    return (int)((o->tokens - 1u + 2u) / 3u);
}

static int compare_request(const options_t *o, const request_t *request,
                           const uint32_t *tokens, size_t token_count,
                           const fingerprint_t *fingerprint, reference_t references[3]) {
    unsigned reference_index = request->prompt == 'A' ? 0u : request->prompt == 'B' ? 1u : 2u;
    reference_t *reference = &references[reference_index];
    if (!reference->valid) {
        if (request->resume || request->snap)
            return ec_fail("first %c reference must be a fresh request", request->prompt);
        reference->tokens = calloc(token_count ? token_count : 1, sizeof *reference->tokens);
        if (!reference->tokens) return ec_fail("allocate baseline token reference");
        memcpy(reference->tokens, tokens, token_count * sizeof *tokens);
        reference->token_count = token_count;
        reference->fingerprint = *fingerprint;
        reference->valid = 1;
        return 0;
    }
    if (reference->token_count != token_count)
        return ec_fail("%s token output length differs from its reference", request->name);
    if (!fingerprint_equal(&reference->fingerprint, fingerprint))
        return ec_fail("%s GPU state fingerprint differs from its %c reference", request->name, request->prompt);
    {
        size_t first = request->n_new ? 0 : request->resume;
        for (size_t i = first; i < token_count; i++) if (tokens[i] != reference->tokens[i])
            return ec_fail("%s token differs from its GPU reference at position %zu", request->name, i);
    }
    (void)o;
    return 0;
}

static int read_protocol_line(FILE *f, char **line, size_t *capacity, const char *what) {
    ssize_t n = getline(line, capacity, f);
    if (n < 0) return ec_fail("private core ended before %s", what);
    trim_line(*line);
    return 0;
}

static int parse_run_output(const options_t *o, const request_t *plan, size_t plan_count,
                            const char *mode, unsigned kv_host,
                            const char *stdout_path, reference_t references[3],
                            run_result_t *run) {
    FILE *f = fopen(stdout_path, "r");
    char *line = NULL;
    size_t cap = 0;
    int chunk_mode = !strcmp(mode, "chunk3");
    int rc = -1;
    if (!f) return ec_fail("open private core output %s: %s", stdout_path, strerror(errno));
    if (read_protocol_line(f, &line, &cap, "ready") < 0) goto done;
    if (strcmp(line, "ready")) { ec_fail("private core did not announce ready: %.100s", line); goto done; }
    if (read_protocol_line(f, &line, &cap, "config") < 0) goto done;
    if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto done; }
    if (strncmp(line, "config ", 7)) { ec_fail("expected config response, got: %.100s", line); goto done; }
    if (config_matches(o, mode, kv_host, line) < 0) goto done;
    if (read_protocol_line(f, &line, &cap, "empty-request rejection") < 0) goto done;
    {
        char expected[128];
        snprintf(expected, sizeof expected, "error: 0 prompt ids + 0 new > %u positions", o->maxt);
        if (strcmp(line, expected)) { ec_fail("empty request was not rejected as an argument error"); goto done; }
    }
    for (size_t r = 0; r < plan_count; r++) {
        const request_t *request = &plan[r];
        uint32_t *tokens = NULL;
        size_t expected_token_count = request->n_new ? request->n_new + 1u : o->tokens;
        time_result_t time_result;
        fingerprint_t fingerprint;
        int positions = -1, paired = -1, chunks = -1;
        unsigned total_positions = o->tokens + request->n_new;
        memset(&fingerprint, 0, sizeof fingerprint);
        if (request->snap) {
            char expected[64];
            snprintf(expected, sizeof expected, "ok: snap %u", request->snap);
            if (read_protocol_line(f, &line, &cap, "snapshot acknowledgement") < 0) goto request_done;
            if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto request_done; }
            if (strcmp(line, expected)) { ec_fail("%s snapshot acknowledgement mismatch", request->name); goto request_done; }
        }
        if (read_protocol_line(f, &line, &cap, "tokens response") < 0) goto request_done;
        if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto request_done; }
        if (parse_tokens_line(line, expected_token_count, &tokens) < 0) goto request_done;
        if (read_protocol_line(f, &line, &cap, "time response") < 0) goto request_done;
        if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto request_done; }
        if (parse_time_line(line, &time_result) < 0) goto request_done;
        if (time_result.resumed != (int)request->resume ||
            time_result.positions_run != (int)(total_positions - request->resume)) {
            ec_fail("%s resumed/executed position counts mismatch", request->name); goto request_done;
        }
        if (time_result.paired_positions != expected_pairs(o, request, chunk_mode)) {
            ec_fail("%s paired-position count mismatch", request->name); goto request_done;
        }
        if (time_result.prefill_chunks != expected_chunks(o, request, chunk_mode)) {
            ec_fail("%s prefill-chunk count mismatch", request->name); goto request_done;
        }
        if (read_protocol_line(f, &line, &cap, "statehash header") < 0) goto request_done;
        if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto request_done; }
        if (parse_statehash_header(line, &positions, &paired, &chunks) < 0) goto request_done;
        if (positions != (int)total_positions || paired != time_result.paired_positions ||
            chunks != time_result.prefill_chunks) {
            ec_fail("%s statehash counters disagree with timing response", request->name); goto request_done;
        }
        for (;;) {
            if (read_protocol_line(f, &line, &cap, "statehash completion") < 0) goto request_done;
            if (line_is_error(line)) { ec_fail("engine reported failure: %.120s", line); goto request_done; }
            if (!strcmp(line, "statehash done")) break;
            if (parse_statehash_line(o, line, &fingerprint) < 0) goto request_done;
        }
        if (fingerprint_complete(o, &fingerprint) < 0 ||
            compare_request(o, request, tokens, expected_token_count, &fingerprint, references) < 0)
            goto request_done;
        run->requests_checked++;
        free(tokens);
        continue;
request_done:
        free(tokens);
        goto done;
    }
    while (getline(&line, &cap, f) >= 0) {
        trim_line(line);
        if (*line) { ec_fail("unexpected trailing private core output: %.100s", line); goto done; }
    }
    if (ferror(f)) { ec_fail("read private core output %s: %s", stdout_path, strerror(errno)); goto done; }
    run->passed = 1;
    rc = 0;
done:
    free(line);
    fclose(f);
    return rc;
}






