#define _GNU_SOURCE
#include "die_balance_capture.h"
#include "die_balance_common.h"

#include <curl/curl.h>
#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define DC_IO_TIMEOUT 120
#define DC_MAX_LINE (4 * 1024 * 1024)

static const char *dc_frontend_env[] = {
    "FLASHD_NAME", "FLASHD_THINKING", "FLASHD_QUEUE_MAX", "FLASHD_QUEUE_WAIT_S",
    "FLASHD_REQUEST_MAX_S", "FLASHD_IMGDIR", "FLASHD_GREEDY", "FLASHVIS_SOCK"
};
static const char *dc_root_ports[] = {"00:01.0", "00:02.0", "00:03.0"};

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
} dc_buffer;

struct dc_relay {
    char *path;
    char *upstream;
    char *log_path;
    pid_t core_pid;
    int server_fd;
    int client_fd;
    int core_fd;
    int log_fd;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int stopping;
    int has_result;
    json_object *result;
    char *failure;
};

typedef struct {
    pid_t pid;
    char *start_ticks;
    char *exe;
    char *exe_sha256;
    char *cwd;
    json_object *argv;
    char *comm;
} dc_process;

typedef struct {
    char *path;
    char *realpath;
    long long device;
    long long inode;
    long long size;
    long long mtime_ns;
} dc_model;

static int dc_fail(char **error, const char *format, ...) {
    if (error) {
        va_list args;
        va_start(args, format);
        if (vasprintf(error, format, args) < 0) *error = NULL;
        va_end(args);
    }
    return 0;
}

static int dc_digits(const char *text, int positive);

static int dc_append(dc_buffer *buffer, const char *text, size_t length) {
    if (buffer->length + length + 1 > buffer->capacity) {
        size_t next = buffer->capacity ? buffer->capacity : 4096;
        while (next < buffer->length + length + 1) next *= 2;
        char *value = realloc(buffer->text, next);
        if (!value) return 0;
        buffer->text = value;
        buffer->capacity = next;
    }
    memcpy(buffer->text + buffer->length, text, length);
    buffer->length += length;
    buffer->text[buffer->length] = '\0';
    return 1;
}

static char *dc_read_file(const char *path, size_t *length, char **error) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        dc_fail(error, "%s", strerror(errno));
        return NULL;
    }
    dc_buffer buffer = {0};
    char chunk[65536];
    for (;;) {
        ssize_t got = read(fd, chunk, sizeof(chunk));
        if (got == 0) break;
        if (got < 0) {
            if (errno == EINTR) continue;
            int saved = errno;
            close(fd);
            free(buffer.text);
            dc_fail(error, "%s", strerror(saved));
            return NULL;
        }
        if (!dc_append(&buffer, chunk, (size_t)got)) {
            close(fd);
            free(buffer.text);
            dc_fail(error, "out of memory");
            return NULL;
        }
    }
    close(fd);
    if (!buffer.text) buffer.text = strdup("");
    if (length) *length = buffer.length;
    return buffer.text;
}

static int dc_write_file(const char *path, const void *data, size_t length, int exclusive,
                         mode_t mode, char **error) {
    int flags = O_WRONLY | O_CREAT | O_CLOEXEC | (exclusive ? O_EXCL : O_TRUNC);
    int fd = open(path, flags, mode);
    if (fd < 0) return dc_fail(error, "%s", strerror(errno));
    const char *bytes = data;
    size_t offset = 0;
    while (offset < length) {
        ssize_t written = write(fd, bytes + offset, length - offset);
        if (written < 0) {
            if (errno == EINTR) continue;
            int saved = errno;
            close(fd);
            return dc_fail(error, "%s", strerror(saved));
        }
        offset += (size_t)written;
    }
    if (close(fd) != 0) return dc_fail(error, "%s", strerror(errno));
    return 1;
}

static double dc_monotonic(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec / 1e9;
}

static char *dc_sha256_path(const char *path, char **error) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        dc_fail(error, "%s", strerror(errno));
        return NULL;
    }
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) {
        if (ctx) EVP_MD_CTX_free(ctx);
        close(fd);
        dc_fail(error, "cannot initialize SHA-256");
        return NULL;
    }
    unsigned char buffer[65536], digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    int ok = 1;
    for (;;) {
        ssize_t got = read(fd, buffer, sizeof(buffer));
        if (got == 0) break;
        if (got < 0) {
            if (errno == EINTR) continue;
            ok = 0;
            break;
        }
        if (EVP_DigestUpdate(ctx, buffer, (size_t)got) != 1) { ok = 0; break; }
    }
    close(fd);
    if (!ok || EVP_DigestFinal_ex(ctx, digest, &digest_length) != 1) {
        EVP_MD_CTX_free(ctx);
        dc_fail(error, "cannot hash file");
        return NULL;
    }
    EVP_MD_CTX_free(ctx);
    char *hex = malloc((size_t)digest_length * 2 + 1);
    if (!hex) {
        dc_fail(error, "out of memory");
        return NULL;
    }
    for (unsigned int i = 0; i < digest_length; i++) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    hex[digest_length * 2] = '\0';
    return hex;
}

static int dc_time_ns(const struct stat *stat_value, long long *out) {
    *out = (long long)stat_value->st_mtim.tv_sec * 1000000000LL + stat_value->st_mtim.tv_nsec;
    return 1;
}

int dc_validation_lock(const char *path, const char *job_id, const char *descriptor,
                       int *lease_fd, char **error) {
    *lease_fd = -1;
    if (job_id || descriptor) {
        if (!job_id || !*job_id || !descriptor) return dc_fail(error, "incomplete inherited gpuq lease");
        char *end = NULL;
        errno = 0;
        long parsed_fd = strtol(descriptor, &end, 10);
        if (errno || end == descriptor || *end || parsed_fd < 0 || parsed_fd > INT32_MAX)
            return dc_fail(error, "invalid gpuq lock descriptor");
        struct stat expected, actual;
        if (stat(path, &expected) != 0 || fstat((int)parsed_fd, &actual) != 0) {
            const char *message = strerror(errno);
            return dc_fail(error, "cannot acquire validation lease: %s", message);
        }
        if (!S_ISREG(actual.st_mode) || expected.st_dev != actual.st_dev || expected.st_ino != actual.st_ino)
            return dc_fail(error, "inherited gpuq lease does not match %s", path);
        int duplicate = fcntl((int)parsed_fd, F_DUPFD_CLOEXEC, 3);
        if (duplicate < 0) return dc_fail(error, "cannot acquire validation lease: %s", strerror(errno));
        if (flock(duplicate, LOCK_EX | LOCK_NB) != 0) {
            close(duplicate);
            return dc_fail(error, "inherited gpuq descriptor does not own the validation lock");
        }
        *lease_fd = duplicate;
        return 1;
    }
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return dc_fail(error, "cannot acquire validation lease: %s", strerror(errno));
    while (flock(fd, LOCK_EX) != 0) {
        if (errno == EINTR) continue;
        int saved = errno;
        close(fd);
        return dc_fail(error, "cannot acquire validation lease: %s", strerror(saved));
    }
    *lease_fd = fd;
    return 1;
}

static int dc_is_uuid(const char *value) {
    if (!value || strncmp(value, "GPU-", 4) != 0 || strlen(value) != 40) return 0;
    for (size_t i = 4; i < 40; i++) {
        if (i == 12 || i == 17 || i == 22 || i == 27) {
            if (value[i] != '-') return 0;
        } else if (!isxdigit((unsigned char)value[i])) return 0;
    }
    return 1;
}

static int dc_gpu_util(const char *value) {
    if (!value || !*value) return 0;
    if (!strcmp(value, "0")) return 1;
    if (value[0] == '0') return 0;
    char *end = NULL;
    long n = strtol(value, &end, 10);
    return end && !*end && end != value && n >= 0 && n <= 100;
}

int dc_nvidia_rows(const char *raw, int mode, json_object **rows, char **error) {
    *rows = json_object_new_array();
    if (!raw || !*raw) return 1;
    char *copy = strdup(raw);
    if (!copy) return dc_fail(error, "out of memory");
    char *line = copy;
    while (line && *line) {
        char *next = strchr(line, '\n');
        if (next) *next++ = '\0';
        size_t line_len = strlen(line);
        if (line_len && line[line_len - 1] == '\r') line[--line_len] = '\0';
        char *raw_line = strdup(line);
        if (!raw_line) { free(copy); json_object_put(*rows); *rows = NULL; return dc_fail(error, "out of memory"); }
        char *values[4] = {0};
        int count = 0;
        char *part = line;
        for (;;) {
            char *comma = strchr(part, ',');
            if (comma) *comma = '\0';
            while (isspace((unsigned char)*part)) part++;
            size_t n = strlen(part);
            while (n && isspace((unsigned char)part[n - 1])) part[--n] = '\0';
            if (count < 4) values[count] = part;
            count++;
            if (!comma) break;
            part = comma + 1;
        }
        int valid = 0;
        if (mode == 3 && count == 3 && values[0]) {
            valid = dc_digits(values[0], 0) && dc_is_uuid(values[1]) && dc_gpu_util(values[2]);
        } else if (mode == 2 && count == 2 && values[0]) {
            valid = dc_digits(values[0], 1) && dc_is_uuid(values[1]);
        }
        if (!valid) {
            json_object *row_text = json_object_new_string(raw_line);
            char *bad_row = db_json_serialize(row_text, 0);
            json_object_put(row_text);
            free(copy);
            int failed = dc_fail(error, "malformed NVIDIA numeric/UUID row: %s", bad_row ? bad_row : "\"\"");
            free(bad_row);
            free(raw_line);
            json_object_put(*rows);
            *rows = NULL;
            return failed;
        }
        json_object *row = json_object_new_array();
        for (int i = 0; i < count; i++) json_object_array_add(row, json_object_new_string(values[i]));
        json_object_array_add(*rows, row);
        free(raw_line);
        line = next && *next ? next : NULL;
    }
    free(copy);
    return 1;
}

static int dc_match_engine_path(pid_t pid, const char *path) {
    char snapshot[256];
    snprintf(snapshot, sizeof(snapshot), "/tmp/flashcore-%d-gen", (int)pid);
    size_t prefix = strlen(snapshot);
    if (strncmp(path, snapshot, prefix) == 0) {
        const char *p = path + prefix;
        if (*p < '1' || *p > '9') return 0;
        while (isdigit((unsigned char)*p)) p++;
        if (strncmp(p, ".so", 3) != 0) return 0;
        p += 3;
        return *p == '\0' || strcmp(p, " (deleted)") == 0;
    }
    if (strstr(path, "/libengine") == path || strstr(path, "/libengine") != NULL) {
        const char *base = strrchr(path, '/');
        if (!base || strncmp(base, "/libengine", 10) != 0) return 0;
        const char *end = strstr(base, ".so");
        if (!end) return 0;
        end += 3;
        return *end == '\0' || strcmp(end, " (deleted)") == 0;
    }
    return 0;
}

int dc_engine_mapping(pid_t pid, const char *maps, char ***fields, char **error) {
    char *copy = strdup(maps ? maps : "");
    if (!copy) return dc_fail(error, "out of memory");
    char **selected = NULL;
    int matches = 0;
    char *save = NULL;
    for (char *line = strtok_r(copy, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        char *tokens[6] = {0};
        char *cursor = line;
        for (int i = 0; i < 5; i++) {
            while (isspace((unsigned char)*cursor)) cursor++;
            if (!*cursor) break;
            tokens[i] = cursor;
            while (*cursor && !isspace((unsigned char)*cursor)) cursor++;
            if (*cursor) *cursor++ = '\0';
        }
        while (isspace((unsigned char)*cursor)) cursor++;
        tokens[5] = cursor;
        if (!tokens[1] || !tokens[4] || !*tokens[4] || !tokens[5] || !*tokens[5] || !strchr(tokens[1], 'x') ||
            !dc_match_engine_path(pid, tokens[5])) continue;
        char **row = calloc(6, sizeof(*row));
        if (!row) { free(copy); return dc_fail(error, "out of memory"); }
        for (int i = 0; i < 6; i++) row[i] = strdup(tokens[i]);
        matches++;
        if (matches == 1) selected = row;
        else {
            for (int i = 0; i < 6; i++) free(row[i]);
            free(row);
        }
    }
    free(copy);
    if (matches != 1) {
        if (selected) { for (int i = 0; i < 6; i++) free(selected[i]); free(selected); }
        return dc_fail(error, "expected one identifiable mapped engine generation");
    }
    *fields = selected;
    return 1;
}

int dc_stop_owned_child(pid_t pid) {
    if (pid <= 0) return 1;
    int status;
    pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid || (waited < 0 && errno == ECHILD)) return 1;
    if (waited < 0) return 1;
    if (kill(pid, SIGTERM) != 0 && errno != ESRCH) return 0;
    double deadline = dc_monotonic() + 3.0;
    while (dc_monotonic() < deadline) {
        waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid || (waited < 0 && errno == ECHILD)) return 1;
        struct timespec pause = {.tv_sec = 0, .tv_nsec = 50000000};
        nanosleep(&pause, NULL);
    }
    if (kill(pid, SIGKILL) != 0 && errno != ESRCH) return 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return 1;
}

static void dc_process_free(dc_process *process) {
    free(process->start_ticks); free(process->exe); free(process->exe_sha256);
    free(process->cwd); free(process->comm);
    if (process->argv) json_object_put(process->argv);
    memset(process, 0, sizeof(*process));
}

static char *dc_proc_path(pid_t pid, const char *name) {
    char *path = NULL;
    asprintf(&path, "/proc/%d/%s", (int)pid, name);
    return path;
}

static int dc_readlink(const char *path, char **result, char **error) {
    size_t size = 256;
    for (;;) {
        char *buffer = malloc(size + 1);
        if (!buffer) return dc_fail(error, "out of memory");
        ssize_t length = readlink(path, buffer, size);
        if (length < 0) {
            int saved = errno;
            free(buffer);
            return dc_fail(error, "%s", strerror(saved));
        }
        if ((size_t)length < size) {
            buffer[length] = '\0';
            *result = buffer;
            return 1;
        }
        free(buffer);
        size *= 2;
    }
}

static int dc_process_identity(pid_t pid, dc_process *process, char **error) {
    memset(process, 0, sizeof(*process));
    process->pid = pid;
    char *path = dc_proc_path(pid, "stat");
    size_t length = 0;
    char *stat_text = dc_read_file(path, &length, error);
    free(path);
    if (!stat_text) return 0;
    char *suffix = NULL;
    for (char *p = stat_text; p[0] && p[1]; p++) if (p[0] == ')' && p[1] == ' ') suffix = p + 2;
    if (!suffix) { free(stat_text); return dc_fail(error, "bad proc stat"); }
    char *save = NULL;
    char *token = strtok_r(suffix, " ", &save);
    for (int i = 0; token && i < 19; i++) token = strtok_r(NULL, " ", &save);
    if (!token) { free(stat_text); return dc_fail(error, "bad proc stat"); }
    process->start_ticks = strdup(token);
    free(stat_text);
    path = dc_proc_path(pid, "exe");
    if (!dc_readlink(path, &process->exe, error)) { free(path); goto failed; }
    process->exe_sha256 = dc_sha256_path(path, error);
    free(path);
    if (!process->exe_sha256) goto failed;
    path = dc_proc_path(pid, "cwd");
    if (!dc_readlink(path, &process->cwd, error)) { free(path); goto failed; }
    free(path);
    path = dc_proc_path(pid, "cmdline");
    char *cmdline = dc_read_file(path, &length, error);
    free(path);
    if (!cmdline) goto failed;
    process->argv = json_object_new_array();
    size_t offset = 0;
    while (offset < length) {
        size_t end = offset;
        while (end < length && cmdline[end]) end++;
        json_object_array_add(process->argv, json_object_new_string_len(cmdline + offset, (int)(end - offset)));
        offset = end + 1;
    }
    free(cmdline);
    path = dc_proc_path(pid, "comm");
    char *comm = dc_read_file(path, &length, error);
    free(path);
    if (!comm) goto failed;
    while (length && isspace((unsigned char)comm[length - 1])) comm[--length] = '\0';
    process->comm = comm;
    return 1;
failed:
    dc_process_free(process);
    return 0;
}

static json_object *dc_process_json(const dc_process *process) {
    json_object *value = json_object_new_object();
    json_object_object_add(value, "pid", json_object_new_int(process->pid));
    json_object_object_add(value, "start_ticks", json_object_new_string(process->start_ticks));
    json_object_object_add(value, "exe", json_object_new_string(process->exe));
    json_object_object_add(value, "exe_sha256", json_object_new_string(process->exe_sha256));
    json_object_object_add(value, "cwd", json_object_new_string(process->cwd));
    json_object_object_add(value, "argv", json_object_get(process->argv));
    json_object_object_add(value, "comm", json_object_new_string(process->comm));
    return value;
}

static int dc_engine_identity(pid_t pid, json_object **identity, char **error) {
    char *maps_path = dc_proc_path(pid, "maps");
    char *maps = dc_read_file(maps_path, NULL, error);
    free(maps_path);
    if (!maps) return 0;
    char **row = NULL;
    int ok = dc_engine_mapping(pid, maps, &row, error);
    free(maps);
    if (!ok) return 0;
    if (strstr(row[5], " (deleted)")) {
        for (int i = 0; i < 6; i++) free(row[i]);
        free(row);
        return dc_fail(error, "mapped engine file was replaced/deleted");
    }
    struct stat st;
    if (stat(row[5], &st) != 0) {
        int saved = errno; for (int i = 0; i < 6; i++) free(row[i]); free(row);
        return dc_fail(error, "%s", strerror(saved));
    }
    char *end = NULL;
    unsigned long long mapped_inode = strtoull(row[4], &end, 10);
    if (!end || *end || st.st_ino != mapped_inode) {
        for (int i = 0; i < 6; i++) free(row[i]);
        free(row);
        return dc_fail(error, "mapped engine inode differs from file");
    }
    char *hash = dc_sha256_path(row[5], error);
    if (!hash) { for (int i = 0; i < 6; i++) free(row[i]); free(row); return 0; }
    json_object *result = json_object_new_object();
    json_object_object_add(result, "path", json_object_new_string(row[5]));
    json_object_object_add(result, "inode", json_object_new_int64(st.st_ino));
    json_object_object_add(result, "size", json_object_new_int64(st.st_size));
    long long mtime_ns; dc_time_ns(&st, &mtime_ns);
    json_object_object_add(result, "mtime_ns", json_object_new_int64(mtime_ns));
    json_object_object_add(result, "sha256", json_object_new_string(hash));
    free(hash);
    for (int i = 0; i < 6; i++) free(row[i]);
    free(row);
    *identity = result;
    return 1;
}

static int dc_model_identity(const char *path, dc_model *model, char **error) {
    memset(model, 0, sizeof(*model));
    model->path = strdup(path);
    model->realpath = realpath(path, NULL);
    if (!model->realpath) return dc_fail(error, "%s", strerror(errno));
    struct stat st;
    if (stat(model->realpath, &st) != 0) return dc_fail(error, "%s", strerror(errno));
    if (!S_ISREG(st.st_mode)) return dc_fail(error, "GGUF is not a regular file");
    model->device = st.st_dev;
    model->inode = st.st_ino;
    model->size = st.st_size;
    dc_time_ns(&st, &model->mtime_ns);
    return 1;
}

static json_object *dc_model_json(const dc_model *model) {
    json_object *value = json_object_new_object();
    json_object_object_add(value, "identity_kind", json_object_new_string("stat"));
    json_object_object_add(value, "path", json_object_new_string(model->path));
    json_object_object_add(value, "realpath", json_object_new_string(model->realpath));
    json_object_object_add(value, "device", json_object_new_int64(model->device));
    json_object_object_add(value, "inode", json_object_new_int64(model->inode));
    json_object_object_add(value, "size", json_object_new_int64(model->size));
    json_object_object_add(value, "mtime_ns", json_object_new_int64(model->mtime_ns));
    return value;
}

static void dc_model_free(dc_model *model) {
    free(model->path); free(model->realpath); memset(model, 0, sizeof(*model));
}

static void dc_process_identity_cleanup(dc_process *process) {
    dc_process_free(process);
}

static int dc_write_json(const char *path, json_object *value, int exclusive, char **error) {
    char *text = db_json_serialize(value, 1);
    if (!text) return dc_fail(error, "out of memory");
    dc_buffer buffer = {0};
    if (!dc_append(&buffer, text, strlen(text)) || !dc_append(&buffer, "\n", 1)) {
        free(text); free(buffer.text);
        return dc_fail(error, "out of memory");
    }
    int result = dc_write_file(path, buffer.text, buffer.length, exclusive, 0600, error);
    free(text); free(buffer.text);
    return result;
}

static int dc_wait_fd(int fd, short events, double deadline, char **error) {
    for (;;) {
        double remaining = deadline - dc_monotonic();
        if (remaining <= 0) return dc_fail(error, "protocol I/O timeout");
        int milliseconds = (int)(remaining * 1000.0);
        if (milliseconds < 1) milliseconds = 1;
        struct pollfd wait_for = {.fd = fd, .events = events};
        int ready = poll(&wait_for, 1, milliseconds);
        if (ready > 0) {
            if (wait_for.revents & (POLLERR | POLLNVAL)) return dc_fail(error, "protocol socket error");
            if (wait_for.revents & (events | POLLHUP)) return 1;
        } else if (ready == 0) {
            return dc_fail(error, "protocol I/O timeout");
        } else if (errno != EINTR) {
            return dc_fail(error, "%s", strerror(errno));
        }
    }
}

static int dc_line(int fd, int eof_ok, char **result, char **error) {
    dc_buffer line = {0};
    double deadline = dc_monotonic() + DC_IO_TIMEOUT;
    for (;;) {
        if (!dc_wait_fd(fd, POLLIN, deadline, error)) { free(line.text); return 0; }
        char byte;
        ssize_t got = read(fd, &byte, 1);
        if (got == 0) {
            if (eof_ok && line.length == 0) { free(line.text); *result = NULL; return 1; }
            if (line.length) { free(line.text); return dc_fail(error, "unterminated/oversized protocol line"); }
            free(line.text);
            return dc_fail(error, "unexpected protocol EOF");
        }
        if (got < 0) {
            if (errno == EINTR) continue;
            int saved = errno; free(line.text);
            return dc_fail(error, "%s", strerror(saved));
        }
        if (byte == '\0') { free(line.text); return dc_fail(error, "NUL in protocol line"); }
        if (line.length >= DC_MAX_LINE) { free(line.text); return dc_fail(error, "unterminated/oversized protocol line"); }
        if (!dc_append(&line, &byte, 1)) { free(line.text); return dc_fail(error, "out of memory"); }
        if (byte == '\n') {
            *result = line.text;
            return 1;
        }
    }
}

static int dc_write(int fd, const char *text, char **error) {
    size_t length = strlen(text), offset = 0;
    double deadline = dc_monotonic() + DC_IO_TIMEOUT;
    while (offset < length) {
        ssize_t written = send(fd, text + offset, length - offset, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (!dc_wait_fd(fd, POLLOUT, deadline, error)) return 0;
                continue;
            }
            return dc_fail(error, "%s", strerror(errno));
        }
        offset += (size_t)written;
    }
    return 1;
}

static int dc_connect_core(const char *path, pid_t pid, int *result, char **error) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return dc_fail(error, "%s", strerror(errno));
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(path) >= sizeof(address.sun_path)) {
        close(fd);
        return dc_fail(error, "UNIX socket path too long");
    }
    strcpy(address.sun_path, path);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        int saved = errno; close(fd); return dc_fail(error, "%s", strerror(saved));
    }
    struct ucred credential;
    socklen_t length = sizeof(credential);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credential, &length) != 0) {
        int saved = errno; close(fd); return dc_fail(error, "%s", strerror(saved));
    }
    if (credential.pid != pid) {
        close(fd);
        return dc_fail(error, "core peer PID %d differs from requested PID %d", credential.pid, pid);
    }
    *result = fd;
    return 1;
}

static int dc_read_config(int fd, char **config, char **error) {
    if (!dc_write(fd, "config\n", error)) return 0;
    if (!dc_line(fd, 0, config, error)) return 0;
    json_object *parsed = NULL;
    if (!db_config_validate(*config, &parsed, error)) { free(*config); *config = NULL; return 0; }
    json_object_put(parsed);
    return 1;
}

static void dc_relay_set_failure(dc_relay *relay, const char *message) {
    pthread_mutex_lock(&relay->mutex);
    if (!relay->failure) relay->failure = strdup(message ? message : "relay failure");
    pthread_cond_broadcast(&relay->condition);
    pthread_mutex_unlock(&relay->mutex);
}

static int dc_relay_log(dc_relay *relay, const char *direction, const char *raw, char **error) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    struct tm utc;
    gmtime_r(&now.tv_sec, &utc);
    char timestamp[64];
    snprintf(timestamp, sizeof(timestamp), "%04d-%02d-%02dT%02d:%02d:%02d.%06ldZ",
             utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
             utc.tm_sec, now.tv_nsec / 1000);
    json_object *entry = json_object_new_object();
    json_object_object_add(entry, "at", json_object_new_string(timestamp));
    json_object_object_add(entry, "direction", json_object_new_string(direction));
    json_object_object_add(entry, "raw", json_object_new_string(raw));
    char *json = db_json_serialize(entry, 0);
    if (!json) { json_object_put(entry); return dc_fail(error, "out of memory"); }
    size_t length = strlen(json);
    int ok = write(relay->log_fd, json, length) == (ssize_t)length &&
             write(relay->log_fd, "\n", 1) == 1;
    free(json);
    json_object_put(entry);
    if (!ok) return dc_fail(error, "%s", strerror(errno));
    return 1;
}

static int dc_relay_send_core(dc_relay *relay, const char *raw, char **error) {
    if (!dc_relay_log(relay, "frontend_or_collector_to_core", raw, error)) return 0;
    return dc_write(relay->core_fd, raw, error);
}

static int dc_relay_from_core(dc_relay *relay, char **raw, char **error) {
    if (!dc_line(relay->core_fd, 0, raw, error)) return 0;
    return dc_relay_log(relay, "core_to_relay", *raw, error);
}

static int dc_relay_profile(dc_relay *relay, int die, json_object **profile, char **error) {
    char command[64];
    snprintf(command, sizeof(command), "prof %d all\n", die);
    if (!dc_relay_send_core(relay, command, error)) return 0;
    char *header = NULL;
    if (!dc_relay_from_core(relay, &header, error)) return 0;
    if (strncmp(header, "prof: ", 6) != 0) { free(header); return dc_fail(error, "invalid profile header"); }
    dc_buffer raw = {0};
    if (!dc_append(&raw, header, strlen(header))) { free(header); return dc_fail(error, "out of memory"); }
    free(header);
    for (;;) {
        char *part = NULL;
        if (!dc_relay_from_core(relay, &part, error)) { free(raw.text); return 0; }
        size_t length = strlen(part);
        if (raw.length + length > DC_MAX_LINE || !dc_append(&raw, part, length)) {
            free(part); free(raw.text); return dc_fail(error, "oversized profile");
        }
        int footer = strncmp(part, "  longest:", 10) == 0;
        int row = 0;
        if (!footer) {
            if (part[0] != ' ' || part[1] != ' ' || !part[2] || isspace((unsigned char)part[2])) {
                free(part); free(raw.text); return dc_fail(error, "unexpected profile record");
            }
            char *p = part;
            while (isspace((unsigned char)*p)) p++;
            char word[64]; int phases;
            row = sscanf(p, "%63s %d phases", word, &phases) == 2 && phases > 0;
            if (!row) { free(part); free(raw.text); return dc_fail(error, "unexpected profile record"); }
        }
        free(part);
        if (footer) break;
    }
    json_object *entry = json_object_new_object();
    json_object_object_add(entry, "die", json_object_new_int(die));
    json_object_object_add(entry, "raw", json_object_new_string_len(raw.text, (int)raw.length));
    free(raw.text);
    db_profile parsed;
    if (!db_profile_validate(entry, &parsed, error)) { json_object_put(entry); return 0; }
    *profile = entry;
    return 1;
}

static int dc_words(char *line, char **words, size_t capacity, size_t *count) {
    size_t n = 0;
    char *save = NULL;
    for (char *word = strtok_r(line, " \t\r\n", &save); word; word = strtok_r(NULL, " \t\r\n", &save)) {
        if (n < capacity) words[n] = word;
        n++;
    }
    *count = n;
    return n <= capacity;
}

static int dc_digits(const char *text, int positive) {
    if (!text || !*text || (positive && *text == '0')) return 0;
    for (const char *p = text; *p; p++) if (!isdigit((unsigned char)*p)) return 0;
    return 1;
}

static int dc_float_token(const char *text, double *value) {
    if (!text || !*text) return 0;
    const char *cursor = text;
    if (*cursor == '+' || *cursor == '-') cursor++;
    int digits = 0;
    while (isdigit((unsigned char)*cursor)) { cursor++; digits++; }
    if (*cursor == '.') {
        cursor++;
        while (isdigit((unsigned char)*cursor)) { cursor++; digits++; }
    }
    if (!digits) return 0;
    if (*cursor == 'e' || *cursor == 'E') {
        cursor++;
        if (*cursor == '+' || *cursor == '-') cursor++;
        int exponent = 0;
        while (isdigit((unsigned char)*cursor)) { cursor++; exponent++; }
        if (!exponent) return 0;
    }
    if (*cursor) return 0;
    char *end = NULL;
    errno = 0;
    double n = strtod(text, &end);
    if (errno == ERANGE || end == text || *end || !isfinite(n)) return 0;
    *value = n;
    return 1;
}

static int dc_response_matches(const char *command, char **words, size_t count) {
    if (!count) return 0;
    if (!strcmp(command, "info")) return count == 3 && !strcmp(words[0], "info") && dc_digits(words[1], 0) && dc_digits(words[2], 0);
    if (!strcmp(command, "config")) return count >= 2 && !strcmp(words[0], "config") && !strncmp(words[1], "rev=", 4);
    if (!strcmp(command, "stop")) return count == 4 && !strcmp(words[0], "ok:") && !strcmp(words[1], "stop") && dc_digits(words[2], 0) && dc_digits(words[3], 0);
    if (!strcmp(command, "sample")) return count >= 3 && !strcmp(words[0], "ok:") && !strcmp(words[1], "sample");
    if (!strcmp(command, "snap")) return count == 3 && !strcmp(words[0], "ok:") && !strcmp(words[1], "snap") && dc_digits(words[2], 0);
    return 0;
}

static int dc_relay_push(dc_relay *relay, json_object *evidence, char **error) {
    pthread_mutex_lock(&relay->mutex);
    if (relay->has_result) {
        pthread_mutex_unlock(&relay->mutex);
        json_object_put(evidence);
        return dc_fail(error, "relay evidence queue was not drained");
    }
    relay->result = evidence;
    relay->has_result = 1;
    pthread_cond_broadcast(&relay->condition);
    pthread_mutex_unlock(&relay->mutex);
    return 1;
}

static int dc_relay_session(dc_relay *relay, char **error) {
    for (;;) {
        char *raw = NULL;
        if (!dc_line(relay->client_fd, 1, &raw, error)) return 0;
        if (!raw) return 1;
        char *copy = strdup(raw);
        if (!copy) { free(raw); return dc_fail(error, "out of memory"); }
        char *words[64] = {0};
        size_t count = 0;
        if (!dc_words(copy, words, 64, &count) || count == 0) {
            free(copy); free(raw); return dc_fail(error, "refusing frontend command");
        }
        char *command = words[0];
        int chat = 0;
        if (!strcmp(command, "info")) {
            if (count != 1) goto refused;
        } else if (!strcmp(command, "config")) {
            if (count != 1) goto refused;
        } else if (!strcmp(command, "stop")) {
            if (count != 3 || !dc_digits(words[1], 0) || !dc_digits(words[2], 0)) goto refused;
        } else if (!strcmp(command, "sample")) {
            double zero;
            double ignored;
            if (count != 5 || !dc_digits(words[1], 0) || !dc_float_token(words[2], &zero) || zero != 0.0 ||
                !dc_float_token(words[3], &ignored) || !dc_float_token(words[4], &ignored)) goto refused;
        } else if (!strcmp(command, "snap")) {
            if (count != 2 || !dc_digits(words[1], 0)) goto refused;
        } else if (!strcmp(command, "chat")) {
            if (count < 3 || count > 64) goto refused;
            long max_tokens = strtol(words[1], NULL, 10);
            if (!dc_digits(words[1], 0) || max_tokens < 1 || max_tokens > 39) goto refused;
            for (size_t i = 2; i < count; i++) if (!dc_digits(words[i], 0)) goto refused;
            chat = 1;
        } else goto refused;
        if (!dc_relay_send_core(relay, raw, error)) { free(copy); free(raw); return 0; }
        if (!chat) {
            char *reply = NULL;
            if (!dc_relay_from_core(relay, &reply, error)) { free(copy); free(raw); return 0; }
            char *reply_copy = strdup(reply);
            char *reply_words[64] = {0}; size_t reply_count = 0;
            if (!reply_copy || !dc_words(reply_copy, reply_words, 64, &reply_count) ||
                !dc_response_matches(command, reply_words, reply_count)) {
                free(reply_copy); free(reply); free(copy); free(raw);
                return dc_fail(error, "unexpected protocol response");
            }
            if (!dc_write(relay->client_fd, reply, error)) { free(reply_copy); free(reply); free(copy); free(raw); return 0; }
            free(reply_copy); free(reply); free(copy); free(raw);
            continue;
        }
        json_object *evidence = json_object_new_object();
        json_object *input = json_object_new_array();
        json_object *output = json_object_new_array();
        for (size_t i = 2; i < count; i++) json_object_array_add(input, json_object_new_int64(strtoll(words[i], NULL, 10)));
        json_object_object_add(evidence, "input_tokens", input);
        json_object_object_add(evidence, "output_tokens", output);
        free(copy);
        free(raw);
        for (;;) {
            char *reply = NULL;
            if (!dc_relay_from_core(relay, &reply, error)) { json_object_put(evidence); return 0; }
            if (!strncmp(reply, "tok ", 4)) {
                const char *digits = reply + 4;
                const char *end = strchr(digits, '\n');
                int valid_id = end && end[1] == '\0' && end != digits;
                unsigned long long id = 0;
                for (const char *p = digits; valid_id && p < end; p++) {
                    if (!isdigit((unsigned char)*p)) valid_id = 0;
                    else {
                        id = id * 10 + (unsigned)(*p - '0');
                        if (id > 4294967295ULL) valid_id = 0;
                    }
                }
                if (!valid_id) {
                    free(reply); json_object_put(evidence); return dc_fail(error, "unexpected chat response");
                }
                json_object_array_add(output, json_object_new_int64(id));
                if (json_object_array_length(output) > 40) { free(reply); json_object_put(evidence); return dc_fail(error, "too many output tokens"); }
                if (!dc_write(relay->client_fd, reply, error)) { free(reply); json_object_put(evidence); return 0; }
                free(reply);
                continue;
            }
            if (!strncmp(reply, "done ", 5)) {
                char *done_copy = strdup(reply);
                char *done_words[10] = {0}; size_t done_count = 0;
                if (!done_copy || !dc_words(done_copy, done_words, 10, &done_count) || done_count != 8 ||
                    !dc_digits(done_words[1], 0) || !dc_digits(done_words[2], 0) ||
                    !dc_digits(done_words[3], 0) || !dc_float_token(done_words[4], &(double){0}) ||
                    !dc_float_token(done_words[5], &(double){0}) || !dc_digits(done_words[6], 0) || !dc_digits(done_words[7], 0) ||
                    strtoull(done_words[1], NULL, 10) != json_object_array_length(output) ||
                    strtoull(done_words[3], NULL, 10) != json_object_array_length(input)) {
                    free(done_copy); free(reply); json_object_put(evidence);
                    return dc_fail(error, "malformed or incomplete done record");
                }
                free(done_copy);
                json_object_object_add(evidence, "done", json_object_new_string(reply));
                char *config = NULL;
                if (!dc_relay_send_core(relay, "config\n", error) || !dc_relay_from_core(relay, &config, error)) {
                    free(reply); json_object_put(evidence); return 0;
                }
                json_object *parsed_config = NULL;
                if (!db_config_validate(config, &parsed_config, error)) {
                    free(config); free(reply); json_object_put(evidence); return 0;
                }
                json_object_put(parsed_config);
                json_object_object_add(evidence, "config", json_object_new_string(config));
                free(config);
                json_object *profiles = json_object_new_array();
                for (int die = 0; die < 6; die++) {
                    json_object *profile = NULL;
                    if (!dc_relay_profile(relay, die, &profile, error)) {
                        json_object_put(profiles); free(reply); json_object_put(evidence); return 0;
                    }
                    json_object_array_add(profiles, profile);
                }
                json_object_object_add(evidence, "profiles", profiles);
                if (!dc_relay_push(relay, evidence, error)) { free(reply); return 0; }
                if (!dc_write(relay->client_fd, reply, error)) { free(reply); return 0; }
                free(reply);
                break;
            }
            free(reply);
            json_object_put(evidence);
            return dc_fail(error, "unexpected chat response");
        }
        continue;
refused:
        free(copy); free(raw);
        return dc_fail(error, "refusing frontend command %s", command);
    }
}

static void *dc_relay_thread(void *argument) {
    dc_relay *relay = argument;
    while (1) {
        pthread_mutex_lock(&relay->mutex);
        int stopping = relay->stopping;
        pthread_mutex_unlock(&relay->mutex);
        if (stopping) break;
        int client = accept4(relay->server_fd, NULL, NULL, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct timespec pause = {.tv_sec = 0, .tv_nsec = 25000000};
                nanosleep(&pause, NULL);
                continue;
            }
            pthread_mutex_lock(&relay->mutex); stopping = relay->stopping; pthread_mutex_unlock(&relay->mutex);
            if (!stopping) dc_relay_set_failure(relay, strerror(errno));
            break;
        }
        relay->client_fd = client;
        char *error = NULL;
        int session_failed = 0;
        if (!dc_connect_core(relay->upstream, relay->core_pid, &relay->core_fd, &error) ||
            !dc_relay_session(relay, &error)) {
            pthread_mutex_lock(&relay->mutex); stopping = relay->stopping; pthread_mutex_unlock(&relay->mutex);
            if (!stopping) {
                dc_relay_set_failure(relay, error ? error : "relay failure");
                session_failed = 1;
            }
        }
        if (relay->client_fd >= 0) close(relay->client_fd);
        if (relay->core_fd >= 0) close(relay->core_fd);
        relay->client_fd = relay->core_fd = -1;
        if (error) free(error);
        if (session_failed) break;
    }
    return NULL;
}

dc_relay *dc_relay_open(const char *path, const char *upstream, pid_t core_pid,
                        const char *log_path, char **error) {
    dc_relay *relay = calloc(1, sizeof(*relay));
    if (!relay) { dc_fail(error, "out of memory"); return NULL; }
    relay->server_fd = relay->client_fd = relay->core_fd = relay->log_fd = -1;
    relay->path = strdup(path); relay->upstream = strdup(upstream);
    relay->log_path = strdup(log_path); relay->core_pid = core_pid;
    pthread_mutex_init(&relay->mutex, NULL);
    pthread_cond_init(&relay->condition, NULL);
    relay->log_fd = open(log_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (relay->log_fd < 0) { dc_fail(error, "%s", strerror(errno)); goto failed; }
    relay->server_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (relay->server_fd < 0) { dc_fail(error, "%s", strerror(errno)); goto failed; }
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (strlen(path) >= sizeof(address.sun_path)) { dc_fail(error, "UNIX socket path too long"); goto failed; }
    strcpy(address.sun_path, path);
    if (bind(relay->server_fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        dc_fail(error, "%s", strerror(errno)); goto failed;
    }
    chmod(path, 0600);
    if (listen(relay->server_fd, 8) != 0) { dc_fail(error, "%s", strerror(errno)); goto failed; }
    int server_flags = fcntl(relay->server_fd, F_GETFL);
    if (server_flags < 0 || fcntl(relay->server_fd, F_SETFL, server_flags | O_NONBLOCK) != 0) {
        dc_fail(error, "%s", strerror(errno)); goto failed;
    }
    if (pthread_create(&relay->thread, NULL, dc_relay_thread, relay) != 0) {
        dc_fail(error, "cannot start relay thread"); goto failed;
    }
    return relay;
failed:
    if (relay->server_fd >= 0) close(relay->server_fd);
    if (relay->log_fd >= 0) close(relay->log_fd);
    unlink(path);
    pthread_cond_destroy(&relay->condition); pthread_mutex_destroy(&relay->mutex);
    free(relay->path); free(relay->upstream); free(relay->log_path); free(relay);
    return NULL;
}

int dc_relay_check(dc_relay *relay, char **error) {
    pthread_mutex_lock(&relay->mutex);
    char *failure = relay->failure ? strdup(relay->failure) : NULL;
    pthread_mutex_unlock(&relay->mutex);
    if (failure) { int result = dc_fail(error, "%s", failure); free(failure); return result; }
    return 1;
}

int dc_relay_pop(dc_relay *relay, json_object **evidence, char **error) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += DC_IO_TIMEOUT;
    pthread_mutex_lock(&relay->mutex);
    while (!relay->has_result && !relay->failure) {
        int result = pthread_cond_timedwait(&relay->condition, &relay->mutex, &deadline);
        if (result == ETIMEDOUT) {
            pthread_mutex_unlock(&relay->mutex);
            return dc_fail(error, "protocol I/O timeout");
        }
    }
    if (relay->failure) {
        char *failure = strdup(relay->failure);
        pthread_mutex_unlock(&relay->mutex);
        int result = dc_fail(error, "%s", failure ? failure : "relay failure");
        free(failure);
        return result;
    }
    *evidence = relay->result;
    relay->result = NULL;
    relay->has_result = 0;
    pthread_mutex_unlock(&relay->mutex);
    return 1;
}

int dc_relay_close(dc_relay *relay, char **error) {
    if (!relay) return 1;
    pthread_mutex_lock(&relay->mutex);
    relay->stopping = 1;
    pthread_mutex_unlock(&relay->mutex);
    if (relay->server_fd >= 0) shutdown(relay->server_fd, SHUT_RDWR);
    if (relay->client_fd >= 0) shutdown(relay->client_fd, SHUT_RDWR);
    if (relay->core_fd >= 0) shutdown(relay->core_fd, SHUT_RDWR);
    int joined = pthread_timedjoin_np(relay->thread, NULL, &(struct timespec){.tv_sec = time(NULL) + 3, .tv_nsec = 0});
    if (joined == ETIMEDOUT) dc_fail(error, "relay did not stop after sockets closed");
    if (relay->server_fd >= 0) close(relay->server_fd);
    if (relay->client_fd >= 0) close(relay->client_fd);
    if (relay->core_fd >= 0) close(relay->core_fd);
    if (relay->log_fd >= 0) close(relay->log_fd);
    struct stat st;
    if (lstat(relay->path, &st) == 0 && S_ISSOCK(st.st_mode)) unlink(relay->path);
    if (relay->result) json_object_put(relay->result);
    free(relay->failure);
    pthread_cond_destroy(&relay->condition); pthread_mutex_destroy(&relay->mutex);
    free(relay->path); free(relay->upstream); free(relay->log_path); free(relay);
    return joined == 0 ? 1 : 0;
}

static char *dc_join_path(const char *directory, const char *name) {
    char *path = NULL;
    asprintf(&path, "%s/%s", directory, name);
    return path;
}

static json_object *dc_argv_json(char *const argv[]) {
    json_object *array = json_object_new_array();
    for (size_t i = 0; argv[i]; i++) json_object_array_add(array, json_object_new_string(argv[i]));
    return array;
}

static int dc_command(char *const argv[], const char *out, const char *name, int seconds,
                      const char *chdir_path, char **result, char **error) {
    char *stdout_name = NULL, *stderr_name = NULL, *command_name = NULL, *exit_name = NULL;
    char *filename = NULL;
    asprintf(&filename, "%s.stdout", name); stdout_name = dc_join_path(out, filename); free(filename);
    asprintf(&filename, "%s.stderr", name); stderr_name = dc_join_path(out, filename); free(filename);
    asprintf(&filename, "%s.command.json", name); command_name = dc_join_path(out, filename); free(filename);
    asprintf(&filename, "%s.exit", name); exit_name = dc_join_path(out, filename); free(filename);
    json_object *command_json = dc_argv_json(argv);
    char *command_text = db_json_serialize(command_json, 0);
    if (!command_text) {
        json_object_put(command_json);
        free(stdout_name); free(stderr_name); free(command_name); free(exit_name);
        return dc_fail(error, "out of memory");
    }
    dc_buffer command_buffer = {0};
    dc_append(&command_buffer, command_text, strlen(command_text));
    dc_append(&command_buffer, "\n", 1);
    free(command_text);
    json_object_put(command_json);
    if (!dc_write_file(command_name, command_buffer.text, command_buffer.length, 0, 0600, error)) goto failed;
    free(command_buffer.text);
    command_buffer.text = NULL;
    pid_t child = fork();
    if (child < 0) { dc_fail(error, "%s", strerror(errno)); goto failed; }
    if (child == 0) {
        int input = open("/dev/null", O_RDONLY);
        int output = open(stdout_name, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int errors = open(stderr_name, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (input >= 0) dup2(input, STDIN_FILENO);
        if (output >= 0) dup2(output, STDOUT_FILENO);
        if (errors >= 0) dup2(errors, STDERR_FILENO);
        if (chdir_path && chdir(chdir_path) != 0) _exit(126);
        execvp(argv[0], argv);
        _exit(127);
    }
    double deadline = dc_monotonic() + seconds;
    int status = 0;
    for (;;) {
        pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) {
            dc_fail(error, "%s", strerror(errno));
            dc_stop_owned_child(child);
            goto failed;
        }
        if (dc_monotonic() >= deadline) {
            dc_stop_owned_child(child);
            dc_fail(error, "command timed out: %s", argv[0]);
            goto failed;
        }
        struct timespec pause = {.tv_sec = 0, .tv_nsec = 50000000};
        nanosleep(&pause, NULL);
    }
    int exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    char status_text[32];
    snprintf(status_text, sizeof(status_text), "%d\n", exit_status);
    if (!dc_write_file(exit_name, status_text, strlen(status_text), 0, 0600, error)) goto failed;
    if (exit_status != 0) {
        json_object *array = dc_argv_json(argv);
        char *shown = db_json_serialize(array, 0);
        dc_fail(error, "command failed: %s", shown);
        free(shown);
        json_object_put(array);
        goto failed;
    }
    *result = dc_read_file(stdout_name, NULL, error);
    if (!*result) goto failed;
    free(stdout_name); free(stderr_name); free(command_name); free(exit_name);
    return 1;
failed:
    free(command_buffer.text);
    free(stdout_name); free(stderr_name); free(command_name); free(exit_name);
    return 0;
}

static int dc_hardware_snapshot(const char *out, const char *prefix,
                                json_object **result, char **error) {
    char name[128];
    char *identity = NULL, *telemetry = NULL;
    char *gpu_argv[] = {"nvidia-smi", "-L", NULL};
    snprintf(name, sizeof(name), "%s-gpu-identity", prefix);
    if (!dc_command(gpu_argv, out, name, 30, NULL, &identity, error)) return 0;
    char *query_argv[] = {"nvidia-smi", "-q", NULL};
    snprintf(name, sizeof(name), "%s-gpu-telemetry", prefix);
    if (!dc_command(query_argv, out, name, 30, NULL, &telemetry, error)) { free(identity); return 0; }
    free(telemetry);
    char links[3][128];
    for (size_t i = 0; i < 3; i++) {
        char port_arg[16];
        snprintf(port_arg, sizeof(port_arg), "%s", dc_root_ports[i]);
        char *lspci_argv[] = {"lspci", "-vv", "-s", port_arg, NULL};
        snprintf(name, sizeof(name), "%s-pcie-%s", prefix, i == 0 ? "00--01-0" : i == 1 ? "00--02-0" : "00--03-0");
        char *raw = NULL;
        if (!dc_command(lspci_argv, out, name, 30, NULL, &raw, error)) { free(identity); return 0; }
        if (!strstr(raw, "LnkSta:")) {
            char *sudo_argv[] = {"sudo", "-n", "lspci", "-vv", "-s", port_arg, NULL};
            char *privileged = NULL;
            char privileged_name[160];
            snprintf(privileged_name, sizeof(privileged_name), "%s-privileged", name);
            free(raw);
            if (!dc_command(sudo_argv, out, privileged_name, 30, NULL, &privileged, error)) { free(identity); return 0; }
            raw = privileged;
        }
        char *status = strstr(raw, "LnkSta:");
        char speed[32] = {0}, width[16] = {0};
        char *speed_field = status ? strstr(status, "Speed") : NULL;
        char *width_field = status ? strstr(status, "Width") : NULL;
        int speed_ok = 0, width_ok = 0;
        if (speed_field) {
            speed_field += 5;
            while (isspace((unsigned char)*speed_field)) speed_field++;
            size_t n = 0;
            while (speed_field[n] && !isspace((unsigned char)speed_field[n]) && speed_field[n] != ',' && n + 1 < sizeof(speed)) n++;
            memcpy(speed, speed_field, n);
            speed[n] = '\0';
            char *gt = strstr(speed, "GT/s");
            speed_ok = gt && gt + 4 == speed + n && n > 4;
            for (char *p = speed; speed_ok && p < gt; p++) if (!(isdigit((unsigned char)*p) || *p == '.')) speed_ok = 0;
        }
        if (width_field) {
            width_field += 5;
            while (isspace((unsigned char)*width_field)) width_field++;
            size_t n = 0;
            while (width_field[n] && !isspace((unsigned char)width_field[n]) && width_field[n] != ',' && n + 1 < sizeof(width)) n++;
            memcpy(width, width_field, n);
            width[n] = '\0';
            width_ok = n >= 2 && width[0] == 'x';
            for (size_t j = 1; width_ok && j < n; j++) if (!isdigit((unsigned char)width[j])) width_ok = 0;
        }
        if (!status || !speed_ok || !width_ok) {
            free(raw); free(identity);
            return dc_fail(error, "cannot read root-port speed/width for %s", port_arg);
        }
        snprintf(links[i], sizeof(links[i]), "%s Speed %s Width %s", port_arg, speed, width);
        free(raw);
    }
    json_object *snapshot = json_object_new_object();
    json_object_object_add(snapshot, "identity", json_object_new_string(identity));
    free(identity);
    dc_buffer joined = {0};
    for (size_t i = 0; i < 3; i++) {
        dc_append(&joined, links[i], strlen(links[i]));
        dc_append(&joined, "\n", 1);
    }
    json_object_object_add(snapshot, "pci_link_snapshot", json_object_new_string(joined.text ? joined.text : ""));
    free(joined.text);
    *result = snapshot;
    return 1;
}

static int dc_idle_preflight(const char *out, pid_t core_pid, const char *socket,
                             char **error) {
    char *ps_argv[] = {"ps", "-eo", "pid,ppid,lstart,stat,args", NULL};
    char *ignored = NULL;
    if (!dc_command(ps_argv, out, "preflight-processes", 30, NULL, &ignored, error)) return 0;
    free(ignored);
    char *gpu_argv[] = {"nvidia-smi", "--query-gpu=index,uuid,utilization.gpu",
                        "--format=csv,noheader,nounits", NULL};
    char *metrics = NULL;
    if (!dc_command(gpu_argv, out, "preflight-gpu-load", 30, NULL, &metrics, error)) return 0;
    json_object *rows = NULL;
    if (!dc_nvidia_rows(metrics, 3, &rows, error)) { free(metrics); return 0; }
    free(metrics);
    int selected[6] = {0};
    const char *selected_uuids[6] = {0};
    for (size_t i = 0; i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        int index = atoi(json_object_get_string(json_object_array_get_idx(row, 0)));
        if (index < 0 || index >= 6) continue;
        if (selected[index]) {
            json_object_put(rows);
            return dc_fail(error, "dies 0-5 are absent or busy; wait for an idle host");
        }
        selected[index] = 1;
        selected_uuids[index] = json_object_get_string(json_object_array_get_idx(row, 1));
        if (strcmp(json_object_get_string(json_object_array_get_idx(row, 2)), "0")) {
            json_object_put(rows);
            return dc_fail(error, "dies 0-5 are absent or busy; wait for an idle host");
        }
    }
    json_object_put(rows);
    for (int i = 0; i < 6; i++) if (!selected[i])
        return dc_fail(error, "dies 0-5 are absent or busy; wait for an idle host");
    char *apps_argv[] = {"nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                         "--format=csv,noheader,nounits", NULL};
    char *apps = NULL;
    if (!dc_command(apps_argv, out, "preflight-compute-apps", 30, NULL, &apps, error)) return 0;
    if (!dc_nvidia_rows(apps, 2, &rows, error)) { free(apps); return 0; }
    free(apps);
    for (size_t i = 0; i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        pid_t pid = (pid_t)strtol(json_object_get_string(json_object_array_get_idx(row, 0)), NULL, 10);
        const char *uuid = json_object_get_string(json_object_array_get_idx(row, 1));
        int participating = 0;
        for (int die = 0; die < 6; die++) if (selected_uuids[die] && !strcmp(uuid, selected_uuids[die])) participating = 1;
        if (!participating) continue;
        char *path = dc_proc_path(pid, "comm");
        char *comm = dc_read_file(path, NULL, error);
        free(path);
        if (!comm) { json_object_put(rows); return 0; }
        size_t length = strlen(comm);
        while (length && isspace((unsigned char)comm[length - 1])) comm[--length] = '\0';
        if (pid != core_pid && strcmp(comm, "hold") && strcmp(comm, "fence")) {
            int failed = dc_fail(error, "unrelated compute process %d (%s) occupies a participating die", (int)pid, comm);
            free(comm); json_object_put(rows);
            return failed;
        }
        free(comm);
    }
    json_object_put(rows);
    char *ss_argv[] = {"ss", "-xnp", NULL};
    char *sockets = NULL;
    if (!dc_command(ss_argv, out, "preflight-sockets", 30, NULL, &sockets, error)) return 0;
    char *socket_copy = strdup(sockets);
    char *save = NULL;
    for (char *line = strtok_r(socket_copy, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        char *established = strstr(line, "ESTAB");
        if (!strstr(line, socket) || !established) continue;
        int left_boundary = established == line || isspace((unsigned char)established[-1]);
        int right_boundary = established[5] == '\0' || isspace((unsigned char)established[5]);
        if (left_boundary && right_boundary) {
            free(socket_copy); free(sockets);
            return dc_fail(error, "resident engine has an established client; wait until it disconnects");
        }
    }
    free(socket_copy);
    free(sockets);
    return 1;
}

static int dc_owned_listener(pid_t pid, int port) {
    char *fd_path = dc_proc_path(pid, "fd");
    DIR *directory = opendir(fd_path);
    free(fd_path);
    if (!directory) return 0;
    char **inodes = NULL;
    size_t inode_count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
        if (entry->d_name[0] == '.') continue;
        char *path = NULL;
        asprintf(&path, "/proc/%d/fd/%s", (int)pid, entry->d_name);
        char target[256];
        ssize_t length = readlink(path, target, sizeof(target) - 1);
        free(path);
        if (length <= 0) continue;
        target[length] = '\0';
        unsigned long long inode;
        if (sscanf(target, "socket:[%llu]", &inode) == 1) {
            char **next = realloc(inodes, (inode_count + 1) * sizeof(*next));
            if (!next) break;
            inodes = next;
            asprintf(&inodes[inode_count++], "%llu", inode);
        }
    }
    closedir(directory);
    FILE *tcp = fopen("/proc/net/tcp", "r");
    if (!tcp) { for (size_t i = 0; i < inode_count; i++) free(inodes[i]); free(inodes); return 0; }
    char expected[32];
    snprintf(expected, sizeof(expected), "0100007F:%04X", port);
    char *line = NULL;
    size_t capacity = 0;
    int found = 0;
    getline(&line, &capacity, tcp);
    while (getline(&line, &capacity, tcp) > 0) {
        char local[64], remote[64], state[16], uid[32], inode[32];
        int parsed = sscanf(line, "%*s %63s %63s %15s %*s %*s %*s %31s %*s %31s", local, remote, state, uid, inode);
        if (parsed < 5 || strcmp(local, expected) || strcmp(state, "0A")) continue;
        for (size_t i = 0; i < inode_count; i++) if (!strcmp(inodes[i], inode)) found = 1;
        if (found) break;
    }
    free(line); fclose(tcp);
    for (size_t i = 0; i < inode_count; i++) free(inodes[i]);
    free(inodes);
    return found;
}

static char *dc_expand_path(const char *base, const char *path) {
    char *combined = NULL;
    if (path[0] == '/') combined = strdup(path);
    else asprintf(&combined, "%s/%s", base, path);
    if (!combined) return NULL;
    char *copy = combined;
    char **parts = NULL;
    size_t count = 0;
    char *save = NULL;
    for (char *part = strtok_r(copy, "/", &save); part; part = strtok_r(NULL, "/", &save)) {
        if (!strcmp(part, ".") || !*part) continue;
        if (!strcmp(part, "..")) {
            if (count) { free(parts[--count]); }
            continue;
        }
        char **next = realloc(parts, (count + 1) * sizeof(*next));
        if (!next) { for (size_t i = 0; i < count; i++) free(parts[i]); free(parts); free(combined); return NULL; }
        parts = next;
        parts[count++] = strdup(part);
    }
    dc_buffer result = {0};
    dc_append(&result, "/", 1);
    for (size_t i = 0; i < count; i++) {
        if (i) dc_append(&result, "/", 1);
        dc_append(&result, parts[i], strlen(parts[i]));
        free(parts[i]);
    }
    free(parts);
    free(combined);
    return result.text;
}

static json_object *dc_request_json(void) {
    json_object *request = json_object_new_object();
    json_object *messages = json_object_new_array();
    json_object *message = json_object_new_object();
    json_object_object_add(message, "role", json_object_new_string("user"));
    json_object_object_add(message, "content", json_object_new_string("What is the capital of France? Answer in one word."));
    json_object_array_add(messages, message);
    json_object_object_add(request, "messages", messages);
    json_object_object_add(request, "max_tokens", json_object_new_int(40));
    json_object_object_add(request, "temperature", json_object_new_int(0));
    return request;
}

static int dc_read_frontend_env(pid_t pid, json_object **behavior, char ***envp, char **error) {
    char *path = dc_proc_path(pid, "environ");
    size_t length = 0;
    char *raw = dc_read_file(path, &length, error);
    free(path);
    if (!raw) return 0;
    json_object *values = json_object_new_object();
    char **environment = calloc(10, sizeof(*environment));
    if (!environment) { free(raw); json_object_put(values); return dc_fail(error, "out of memory"); }
    size_t used = 0;
    for (size_t offset = 0; offset < length;) {
        size_t end = offset;
        while (end < length && raw[end]) end++;
        if (end == offset) { offset = end + 1; continue; }
        char *item = strndup(raw + offset, end - offset);
        char *equals = strchr(item, '=');
        if (equals) {
            *equals = '\0';
            int wanted = 0;
            for (size_t i = 0; i < sizeof(dc_frontend_env) / sizeof(dc_frontend_env[0]); i++)
                if (!strcmp(item, dc_frontend_env[i])) wanted = 1;
            if (wanted) {
                json_object_object_add(values, item, json_object_new_string(equals + 1));
                asprintf(&environment[used++], "%s=%s", item, equals + 1);
            }
        }
        free(item);
        offset = end + 1;
    }
    free(raw);
    const char *path_value = getenv("PATH");
    if (!path_value) path_value = "/usr/bin:/bin";
    asprintf(&environment[used++], "PATH=%s", path_value);
    environment[used] = NULL;
    *behavior = values;
    *envp = environment;
    return 1;
}

static int dc_same_process(const dc_process *left, const dc_process *right) {
    return left->pid == right->pid &&
           !strcmp(left->start_ticks, right->start_ticks) &&
           !strcmp(left->exe, right->exe) &&
           !strcmp(left->exe_sha256, right->exe_sha256) &&
           !strcmp(left->cwd, right->cwd) &&
           json_object_equal(left->argv, right->argv) &&
           !strcmp(left->comm, right->comm);
}

static int dc_same_model(const dc_model *left, const dc_model *right) {
    return !strcmp(left->path, right->path) && !strcmp(left->realpath, right->realpath) &&
           left->device == right->device && left->inode == right->inode &&
           left->size == right->size && left->mtime_ns == right->mtime_ns;
}

static char *dc_source_root(void) {
    char executable[4096];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length <= 0) return NULL;
    executable[length] = '\0';
    char *slash = strrchr(executable, '/');
    if (!slash) return NULL;
    *slash = '\0';
    slash = strrchr(executable, '/');
    if (!slash) return NULL;
    *slash = '\0';
    return strdup(executable);
}

static size_t dc_curl_write(char *data, size_t size, size_t count, void *context) {
    dc_buffer *buffer = context;
    size_t length = size * count;
    return dc_append(buffer, data, length) ? length : 0;
}

static int dc_http_request(int port, const char *out, int index,
                           json_object *request, json_object **body, char **error) {
    char *url = NULL;
    asprintf(&url, "http://127.0.0.1:%d/v1/chat/completions", port);
    char *request_text = db_json_serialize(request, 0);
    if (!request_text) { free(url); return dc_fail(error, "out of memory"); }
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    CURL *curl = curl_easy_init();
    if (!curl) { free(url); free(request_text); curl_slist_free_all(headers); return dc_fail(error, "cannot initialize HTTP client"); }
    dc_buffer response = {0};
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_PROXY, "");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_text);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(request_text));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)DC_IO_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dc_curl_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    CURLcode result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    free(url);
    free(request_text);
    if (result != CURLE_OK) {
        free(response.text);
        return dc_fail(error, "HTTP request failed: %s", curl_easy_strerror(result));
    }
    char filename[64];
    snprintf(filename, sizeof(filename), "http-%02d.json", index);
    char *path = dc_join_path(out, filename);
    if (!dc_write_file(path, response.text ? response.text : "", response.length, 0, 0600, error)) {
        free(path); free(response.text); return 0;
    }
    free(path);
    if (status < 200 || status >= 300) {
        free(response.text);
        return dc_fail(error, "HTTP request failed: %ld", status);
    }
    json_tokener *tokener = json_tokener_new();
    json_object *parsed = json_tokener_parse_ex(tokener, response.text, (int)response.length);
    enum json_tokener_error parse_error = json_tokener_get_error(tokener);
    json_tokener_free(tokener);
    free(response.text);
    if (parse_error != json_tokener_success || !parsed || json_object_get_type(parsed) != json_type_object ||
        db_get(parsed, "error") || !db_get(parsed, "choices") ||
        json_object_get_type(db_get(parsed, "choices")) != json_type_array) {
        if (parsed) json_object_put(parsed);
        return dc_fail(error, "invalid HTTP JSON");
    }
    *body = parsed;
    return 1;
}

typedef struct {
    const char *out;
    const char *label;
    const char *socket;
    const char *lock;
    int core_pid;
    int flashd_pid;
    int requests;
    int has_core_pid;
    int has_flashd_pid;
} dc_options;

static const char *dc_usage =
    "Usage: ruby kern/die_balance_capture.rb --out FRESH_DIR --label LABEL --core-pid PID --flashd-pid PID\n"
    "        --out PATH\n"
    "        --label LABEL\n"
    "        --core-pid PID\n"
    "        --flashd-pid PID\n"
    "        --socket PATH\n"
    "        --requests N\n"
    "        --lock PATH\n"
    "    -h, --help\n";

static int dc_parse_integer(const char *text, int *value) {
    if (!text || !*text) return 0;
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 0);
    if (errno || end == text || *end || parsed < INT32_MIN || parsed > INT32_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static int dc_parse_options(int argc, char **argv, dc_options *options) {
    memset(options, 0, sizeof(*options));
    options->socket = "/tmp/flash.sock";
    options->requests = 3;
    options->lock = "/tmp/flash-sass-engine-validation.lock";
    for (int i = 1; i < argc; i++) {
        const char *argument = argv[i];
        char name_storage[128];
        const char *name = argument;
        const char *value = NULL;
        const char *equals = strchr(argument, '=');
        if (equals) {
            size_t length = (size_t)(equals - argument);
            if (length >= sizeof(name_storage)) {
                fprintf(stderr, "OptionParser::InvalidOption: invalid option: %s\n", argument);
                return 0;
            }
            memcpy(name_storage, argument, length);
            name_storage[length] = '\0';
            name = name_storage;
            value = equals + 1;
        }
        if (!strcmp(name, "-h") || !strcmp(name, "--help")) return 2;
        if (!strcmp(name, "--")) {
            if (i + 1 < argc) {
                fprintf(stderr, "CaptureError: unexpected positional arguments\n");
                return 0;
            }
            break;
        }
        if (!strcmp(name, "--out") || !strcmp(name, "--label") ||
            !strcmp(name, "--core-pid") || !strcmp(name, "--flashd-pid") ||
            !strcmp(name, "--socket") || !strcmp(name, "--requests") || !strcmp(name, "--lock")) {
            int next_is_integer = 0;
            if (!value && i + 1 < argc &&
                (!strcmp(name, "--core-pid") || !strcmp(name, "--flashd-pid") || !strcmp(name, "--requests"))) {
                int ignored;
                next_is_integer = dc_parse_integer(argv[i + 1], &ignored);
            }
            if (!value && (i + 1 >= argc || (argv[i + 1][0] == '-' && !next_is_integer))) {
                fprintf(stderr, "OptionParser::MissingArgument: missing argument: %s\n", name);
                return 0;
            }
            if (!value) value = argv[++i];
            if (!strcmp(name, "--out")) options->out = value;
            else if (!strcmp(name, "--label")) options->label = value;
            else if (!strcmp(name, "--socket")) options->socket = value;
            else if (!strcmp(name, "--lock")) options->lock = value;
            else if (!strcmp(name, "--core-pid")) {
                if (!dc_parse_integer(value, &options->core_pid)) {
                    fprintf(stderr, "OptionParser::InvalidArgument: invalid argument: --core-pid %s\n", value);
                    return 0;
                }
                options->has_core_pid = 1;
            } else if (!strcmp(name, "--flashd-pid")) {
                if (!dc_parse_integer(value, &options->flashd_pid)) {
                    fprintf(stderr, "OptionParser::InvalidArgument: invalid argument: --flashd-pid %s\n", value);
                    return 0;
                }
                options->has_flashd_pid = 1;
            } else if (!dc_parse_integer(value, &options->requests)) {
                fprintf(stderr, "OptionParser::InvalidArgument: invalid argument: --requests %s\n", value);
                return 0;
            }
        } else if (name[0] == '-') {
            fprintf(stderr, "OptionParser::InvalidOption: invalid option: %s\n", argument);
            return 0;
        } else {
            fprintf(stderr, "CaptureError: unexpected positional arguments\n");
            return 0;
        }
    }
    if (!options->out) { fprintf(stderr, "CaptureError: missing --out\n"); return 0; }
    if (!options->label) { fprintf(stderr, "CaptureError: missing --label\n"); return 0; }
    if (!options->has_core_pid) { fprintf(stderr, "CaptureError: missing --core-pid\n"); return 0; }
    if (!options->has_flashd_pid) { fprintf(stderr, "CaptureError: missing --flashd-pid\n"); return 0; }
    if (!db_text_valid(options->label)) { fprintf(stderr, "CaptureError: label must not be empty\n"); return 0; }
    if (options->requests < 3) { fprintf(stderr, "CaptureError: at least three requests are required\n"); return 0; }
    if (options->core_pid <= 1 || options->flashd_pid <= 1 || options->core_pid == options->flashd_pid) {
        fprintf(stderr, "CaptureError: invalid resident PIDs\n"); return 0;
    }
    return 1;
}

static int dc_spawn_frontend(pid_t resident_pid, const char *model_path, const char *relay_path,
                             int port, char *const envp[], const char *cwd, const char *out,
                             pid_t *child, char **error) {
    char listen[64];
    snprintf(listen, sizeof(listen), "127.0.0.1:%d", port);
    char *stdout_path = dc_join_path(out, "frontend.stdout");
    char *stderr_path = dc_join_path(out, "frontend.stderr");
    pid_t pid = fork();
    if (pid < 0) {
        free(stdout_path); free(stderr_path);
        return dc_fail(error, "%s", strerror(errno));
    }
    if (pid == 0) {
        int input = open("/dev/null", O_RDONLY);
        int stdout_fd = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int stderr_fd = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (input >= 0) dup2(input, STDIN_FILENO);
        if (stdout_fd >= 0) dup2(stdout_fd, STDOUT_FILENO);
        if (stderr_fd >= 0) dup2(stderr_fd, STDERR_FILENO);
        if (chdir(cwd) != 0) _exit(126);
        char executable[64];
        snprintf(executable, sizeof(executable), "/proc/%d/exe", (int)resident_pid);
        char *args[] = {executable, (char *)model_path, (char *)relay_path, listen, NULL};
        execve(executable, args, envp);
        _exit(127);
    }
    free(stdout_path); free(stderr_path);
    *child = pid;
    return 1;
}

static int dc_validation_capture(const char *partial, char **error) {
    char executable[4096];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length <= 0) return dc_fail(error, "cannot locate analyzer executable");
    executable[length] = '\0';
    char *slash = strrchr(executable, '/');
    if (!slash) return dc_fail(error, "cannot locate analyzer executable");
    strcpy(slash + 1, "die_balance");
    pid_t child = fork();
    if (child < 0) return dc_fail(error, "%s", strerror(errno));
    if (child == 0) {
        int nullfd = open("/dev/null", O_WRONLY);
        if (nullfd >= 0) { dup2(nullfd, STDOUT_FILENO); dup2(nullfd, STDERR_FILENO); }
        char *args[] = {executable, "--baseline", (char *)partial, NULL};
        execv(executable, args);
        _exit(127);
    }
    int status;
    while (waitpid(child, &status, 0) < 0) if (errno != EINTR) return dc_fail(error, "%s", strerror(errno));
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return dc_fail(error, "capture validation failed");
    return 1;
}

static int dc_remove_tree(const char *path) {
    struct stat st;
    if (lstat(path, &st) != 0) return errno == ENOENT;
    if (!S_ISDIR(st.st_mode)) return unlink(path) == 0;
    DIR *directory = opendir(path);
    if (!directory) return 0;
    struct dirent *entry;
    int ok = 1;
    while ((entry = readdir(directory))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *child = dc_join_path(path, entry->d_name);
        if (!dc_remove_tree(child)) ok = 0;
        free(child);
    }
    closedir(directory);
    return rmdir(path) == 0 && ok;
}

static void dc_free_env(char **envp) {
    if (!envp) return;
    for (size_t i = 0; envp[i]; i++) free(envp[i]);
    free(envp);
}

static int dc_capture_execute(const dc_options *options) {
    int lock_fd = -1;
    char *out = NULL, *target_socket = NULL, *model_path = NULL;
    char *temp = NULL, *config_before = NULL, *config_after = NULL;
    char *revision = NULL, *source_status = NULL, *error = NULL;
    dc_process core = {0}, frontend = {0}, core_after = {0}, frontend_after = {0};
    dc_model model = {0}, model_after = {0};
    json_object *engine = NULL, *engine_after = NULL, *hardware = NULL, *hardware_after = NULL;
    json_object *behavior_env = NULL, *runs = NULL, *capture = NULL, *request = NULL;
    dc_relay *relay = NULL;
    pid_t clone_pid = -1;
    char **envp = NULL;
    int created_out = 0, result = 1;

    const char *job_id = getenv("GPUQ_JOB_ID");
    const char *lock_descriptor = getenv("GPUQ_LOCK_FD");
    if (!dc_validation_lock(options->lock, job_id, lock_descriptor, &lock_fd, &error)) goto failed;
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) { dc_fail(&error, "%s", strerror(errno)); goto failed; }
    out = dc_expand_path(cwd, options->out);
    if (!out) { dc_fail(&error, "out of memory"); goto failed; }
    if (mkdir(out, 0700) != 0) {
        if (errno == EEXIST) asprintf(&error, "Errno::EEXIST: File exists @ dir_s_mkdir - %s", out);
        else dc_fail(&error, "%s", strerror(errno));
        goto failed;
    }
    created_out = 1;
    request = dc_request_json();
    char *request_path = dc_join_path(out, "request.json");
    if (!dc_write_json(request_path, request, 0, &error)) { free(request_path); goto failed; }
    free(request_path);
    if (!dc_process_identity((pid_t)options->core_pid, &core, &error) ||
        !dc_process_identity((pid_t)options->flashd_pid, &frontend, &error)) goto failed;
    if (strcmp(core.comm, "flashcore") || strcmp(frontend.comm, "flashd")) {
        dc_fail(&error, "requested PIDs must identify flashcore and flashd"); goto failed;
    }
    if (json_object_array_length(frontend.argv) != 4) {
        dc_fail(&error, "expected flashd argv: executable GGUF socket listen"); goto failed;
    }
    const char *resident_socket = json_object_get_string(json_object_array_get_idx(frontend.argv, 2));
    const char *resident_model = json_object_get_string(json_object_array_get_idx(frontend.argv, 1));
    target_socket = dc_expand_path(cwd, options->socket);
    char *socket_from_cwd = dc_expand_path(frontend.cwd, resident_socket);
    model_path = dc_expand_path(frontend.cwd, resident_model);
    if (!target_socket || !socket_from_cwd || !model_path) {
        free(socket_from_cwd); dc_fail(&error, "cannot expand resident paths"); goto failed;
    }
    if (strcmp(socket_from_cwd, target_socket)) {
        free(socket_from_cwd); dc_fail(&error, "resident flashd socket differs from requested socket"); goto failed;
    }
    free(socket_from_cwd);
    if (!dc_model_identity(model_path, &model, &error) ||
        !dc_engine_identity((pid_t)options->core_pid, &engine, &error) ||
        !dc_read_frontend_env((pid_t)options->flashd_pid, &behavior_env, &envp, &error)) goto failed;
    json_object *resident = json_object_new_object();
    json_object_object_add(resident, "core", dc_process_json(&core));
    json_object_object_add(resident, "frontend", dc_process_json(&frontend));
    json_object_object_add(resident, "engine", json_object_get(engine));
    char *resident_path = dc_join_path(out, "resident-processes.json");
    if (!dc_write_json(resident_path, resident, 0, &error)) {
        json_object_put(resident); free(resident_path); goto failed;
    }
    json_object_put(resident); free(resident_path);
    if (!dc_idle_preflight(out, (pid_t)options->core_pid, target_socket, &error)) goto failed;
    if (!dc_hardware_snapshot(out, "before", &hardware, &error)) goto failed;
    char *source_root = dc_source_root();
    if (!source_root) { dc_fail(&error, "cannot locate source root"); goto failed; }
    char *revision_argv[] = {"git", "rev-parse", "HEAD", NULL};
    if (!dc_command(revision_argv, out, "source-revision", 30, source_root, &revision, &error)) { free(source_root); goto failed; }
    while (*revision && isspace((unsigned char)revision[strlen(revision) - 1])) revision[strlen(revision) - 1] = '\0';
    char *status_argv[] = {"git", "status", "--porcelain=v1", NULL};
    if (!dc_command(status_argv, out, "source-status", 30, source_root, &source_status, &error)) { free(source_root); goto failed; }
    free(source_root);
    int core_fd = -1;
    if (!dc_connect_core(target_socket, (pid_t)options->core_pid, &core_fd, &error)) goto failed;
    if (!dc_read_config(core_fd, &config_before, &error)) { close(core_fd); goto failed; }
    close(core_fd);
    char *config_path = dc_join_path(out, "config-before.txt");
    if (!dc_write_file(config_path, config_before, strlen(config_before), 0, 0600, &error)) { free(config_path); goto failed; }
    free(config_path);
    char temp_template[] = "/tmp/flash-die-balance-XXXXXX";
    temp = strdup(temp_template);
    if (!temp || !mkdtemp(temp)) { dc_fail(&error, "%s", strerror(errno)); goto failed; }
    char *relay_path = dc_join_path(temp, "relay.sock");
    char *protocol_path = dc_join_path(out, "protocol.jsonl");
    relay = dc_relay_open(relay_path, target_socket, (pid_t)options->core_pid, protocol_path, &error);
    free(protocol_path);
    if (!relay) { free(relay_path); goto failed; }
    int reservation = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (reservation < 0) { free(relay_path); dc_fail(&error, "%s", strerror(errno)); goto failed; }
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK), .sin_port = 0};
    if (bind(reservation, (struct sockaddr *)&address, sizeof(address)) != 0) {
        int saved = errno; close(reservation); free(relay_path); dc_fail(&error, "%s", strerror(saved)); goto failed;
    }
    socklen_t address_length = sizeof(address);
    getsockname(reservation, (struct sockaddr *)&address, &address_length);
    int port = ntohs(address.sin_port);
    close(reservation);
    if (!dc_spawn_frontend((pid_t)options->flashd_pid, model_path, relay_path, port,
                           envp, frontend.cwd, out, &clone_pid, &error)) { free(relay_path); goto failed; }
    free(relay_path);
    double startup_deadline = dc_monotonic() + 30.0;
    while (!dc_owned_listener(clone_pid, port)) {
        if (!dc_relay_check(relay, &error)) goto failed;
        int child_status = 0;
        pid_t waited = waitpid(clone_pid, &child_status, WNOHANG);
        if (waited == clone_pid) {
            clone_pid = -1;
            dc_fail(&error, "owned frontend exited during startup"); goto failed;
        }
        if (dc_monotonic() >= startup_deadline) { dc_fail(&error, "owned frontend startup timeout"); goto failed; }
        struct timespec pause = {.tv_sec = 0, .tv_nsec = 50000000};
        nanosleep(&pause, NULL);
    }
    runs = json_object_new_array();
    for (int index = 0; index <= options->requests; index++) {
        if (!dc_relay_check(relay, &error)) goto failed;
        json_object *response = NULL;
        if (!dc_http_request(port, out, index, request, &response, &error)) goto failed;
        json_object *evidence = NULL;
        if (!dc_relay_pop(relay, &evidence, &error)) { json_object_put(response); goto failed; }
        json_object_object_add(evidence, "index", json_object_new_int(index));
        json_object_object_add(evidence, "warmup", json_object_new_boolean(index == 0));
        json_object_object_add(evidence, "response", response);
        json_object_array_add(runs, evidence);
    }
    if (!dc_relay_check(relay, &error)) goto failed;
    dc_stop_owned_child(clone_pid);
    clone_pid = -1;
    if (!dc_relay_close(relay, &error)) { relay = NULL; goto failed; }
    relay = NULL;
    if (!dc_connect_core(target_socket, (pid_t)options->core_pid, &core_fd, &error)) goto failed;
    if (!dc_read_config(core_fd, &config_after, &error)) { close(core_fd); goto failed; }
    close(core_fd);
    config_path = dc_join_path(out, "config-after.txt");
    if (!dc_write_file(config_path, config_after, strlen(config_after), 0, 0600, &error)) { free(config_path); goto failed; }
    free(config_path);
    if (!dc_process_identity((pid_t)options->core_pid, &core_after, &error) ||
        !dc_process_identity((pid_t)options->flashd_pid, &frontend_after, &error) ||
        !dc_engine_identity((pid_t)options->core_pid, &engine_after, &error) ||
        !dc_model_identity(model_path, &model_after, &error) ||
        !dc_hardware_snapshot(out, "after", &hardware_after, &error)) goto failed;
    if (!dc_same_process(&core, &core_after)) { dc_fail(&error, "core identity changed"); goto failed; }
    if (!dc_same_process(&frontend, &frontend_after)) { dc_fail(&error, "frontend identity changed"); goto failed; }
    if (!json_object_equal(engine, engine_after)) { dc_fail(&error, "engine library changed"); goto failed; }
    if (!dc_same_model(&model, &model_after)) { dc_fail(&error, "model stat identity changed"); goto failed; }
    if (!json_object_equal(hardware, hardware_after)) { dc_fail(&error, "GPU identities or PCIe links changed"); goto failed; }
    capture = json_object_new_object();
    json_object_object_add(capture, "schema", json_object_new_string(DB_SCHEMA));
    json_object_object_add(capture, "label", json_object_new_string(options->label));
    struct timespec created; clock_gettime(CLOCK_REALTIME, &created);
    struct tm created_utc; gmtime_r(&created.tv_sec, &created_utc);
    char created_at[64];
    snprintf(created_at, sizeof(created_at), "%04d-%02d-%02dT%02d:%02d:%02d.%06ldZ",
             created_utc.tm_year + 1900, created_utc.tm_mon + 1, created_utc.tm_mday,
             created_utc.tm_hour, created_utc.tm_min, created_utc.tm_sec, created.tv_nsec / 1000);
    json_object_object_add(capture, "created_at", json_object_new_string(created_at));
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname) - 1) != 0) { dc_fail(&error, "%s", strerror(errno)); goto failed; }
    json_object_object_add(capture, "hostname", json_object_new_string(hostname));
    json_object_object_add(capture, "request", json_object_get(request));
    json_object_object_add(capture, "warmup_requests", json_object_new_int(1));
    json_object *manifest = json_object_new_object();
    json_object_object_add(manifest, "config_before", json_object_new_string(config_before));
    json_object_object_add(manifest, "config_after", json_object_new_string(config_after));
    json_object_object_add(manifest, "frontend_sha256", json_object_new_string(frontend.exe_sha256));
    json_object_object_add(manifest, "frontend_env", json_object_get(behavior_env));
    json_object_object_add(manifest, "core_exe_sha256", json_object_new_string(core.exe_sha256));
    json_object_object_add(manifest, "engine_sha256", json_object_get(db_get(engine, "sha256")));
    json_object_object_add(manifest, "model", dc_model_json(&model));
    json_object_object_add(manifest, "hardware", json_object_get(hardware));
    json_object_object_add(manifest, "source_revision", json_object_new_string(revision));
    json_object_object_add(manifest, "source_status", json_object_new_string(source_status));
    json_object_object_add(capture, "manifest", manifest);
    json_object_object_add(capture, "runs", json_object_get(runs));
    char *partial = dc_join_path(out, "capture.json.partial");
    char *final = dc_join_path(out, "capture.json");
    if (!dc_write_json(partial, capture, 1, &error)) { free(partial); free(final); goto failed; }
    if (!dc_validation_capture(partial, &error)) { free(partial); free(final); goto failed; }
    if (rename(partial, final) != 0) { dc_fail(&error, "%s", strerror(errno)); free(partial); free(final); goto failed; }
    puts(final);
    free(partial); free(final);
    result = 0;
    goto cleanup;

failed:
    if (!error) error = strdup("capture failed");
    if (created_out) {
        char *error_path = dc_join_path(out, "error.txt");
        char *text = NULL;
        if (error && !strncmp(error, "Errno::", 7)) asprintf(&text, "%s\n", error);
        else asprintf(&text, "CaptureError: %s\n", error);
        dc_write_file(error_path, text, strlen(text), 0, 0600, NULL);
        free(error_path); free(text);
    }
    if (error && !strncmp(error, "Errno::", 7)) fprintf(stderr, "%s\n", error);
    else fprintf(stderr, "CaptureError: %s\n", error ? error : "capture failed");
    result = 1;
cleanup:
    if (clone_pid > 0) dc_stop_owned_child(clone_pid);
    if (relay) {
        char *cleanup_error = NULL;
        if (!dc_relay_close(relay, &cleanup_error) && cleanup_error)
            fprintf(stderr, "relay cleanup: %s\n", cleanup_error);
        free(cleanup_error);
    }
    if (temp) dc_remove_tree(temp);
    if (lock_fd >= 0) close(lock_fd);
    if (request) json_object_put(request);
    if (runs) json_object_put(runs);
    if (capture) json_object_put(capture);
    if (engine) json_object_put(engine);
    if (engine_after) json_object_put(engine_after);
    if (hardware) json_object_put(hardware);
    if (hardware_after) json_object_put(hardware_after);
    if (behavior_env) json_object_put(behavior_env);
    dc_process_identity_cleanup(&core); dc_process_identity_cleanup(&frontend);
    dc_process_identity_cleanup(&core_after); dc_process_identity_cleanup(&frontend_after);
    dc_model_free(&model); dc_model_free(&model_after);
    dc_free_env(envp);
    free(out); free(target_socket); free(model_path); free(temp);
    free(config_before); free(config_after); free(revision); free(source_status); free(error);
    return result;
}

int dc_capture_run(int argc, char **argv) {
    dc_options options;
    int parsed = dc_parse_options(argc, argv, &options);
    if (parsed == 2) { fputs(dc_usage, stdout); return 0; }
    if (parsed == 0) return 1;
    return dc_capture_execute(&options);
}

#ifndef DIE_BALANCE_CAPTURE_NO_MAIN
int main(int argc, char **argv) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        fprintf(stderr, "CaptureError: cannot initialize HTTP client\n");
        return 1;
    }
    int result = dc_capture_run(argc, argv);
    curl_global_cleanup();
    return result;
}
#endif
