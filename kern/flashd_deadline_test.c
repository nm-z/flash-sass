#define _GNU_SOURCE
#include "../validation/issue80/test_fixture.h"
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} buffer;

typedef struct {
    int status;
    char *body;
    double elapsed;
    char *error;
} http_result;

typedef struct {
    pid_t pid;
    int process_group;
} owned_child;

static char temp_dir[PATH_MAX];
static char socket_path[PATH_MAX];
static char mode_path[PATH_MAX];
static char binary_path[PATH_MAX];
static int tcp_port;
static pid_t owner_pid;
static owned_child owned[32];
static size_t owned_count;
static int test_number;
static int test_failed;

static int join_path(char *out, size_t capacity, const char *dir, const char *leaf) {
    size_t dir_length = strlen(dir), leaf_length = strlen(leaf);
    int separator = dir_length && dir[dir_length - 1] != '/';
    if (dir_length + (size_t)separator + leaf_length + 1 > capacity) {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(out, dir, dir_length);
    size_t offset = dir_length;
    if (separator) out[offset++] = '/';
    memcpy(out + offset, leaf, leaf_length + 1);
    return 0;
}

static double monotonic_time(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void pause_for(double seconds) {
    struct timespec request = {
        .tv_sec = (time_t)seconds,
        .tv_nsec = (long)((seconds - (time_t)seconds) * 1e9),
    };
    while (nanosleep(&request, &request) < 0 && errno == EINTR) { }
}

static void track_child(pid_t pid, int process_group) {
    if (pid > 0 && owned_count < sizeof owned / sizeof owned[0])
        owned[owned_count++] = (owned_child){pid, process_group};
}

static void forget_child(pid_t pid) {
    for (size_t i = 0; i < owned_count; ++i) {
        if (owned[i].pid != pid) continue;
        owned[i] = owned[--owned_count];
        return;
    }
}

static int wait_child(pid_t pid, int options) {
    int status = 0;
    pid_t result;
    do result = waitpid(pid, &status, options); while (result < 0 && errno == EINTR);
    if (result == 0) return -2;
    if (result < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

static void cleanup(void) {
    if (getpid() != owner_pid) return;
    for (size_t i = 0; i < owned_count; ++i)
        kill(owned[i].process_group ? -owned[i].pid : owned[i].pid, SIGTERM);
    double until = monotonic_time() + 0.5;
    while (owned_count && monotonic_time() < until) {
        for (size_t i = 0; i < owned_count;) {
            int result = wait_child(owned[i].pid, WNOHANG);
            if (result != -2) {
                owned[i] = owned[--owned_count];
            } else {
                ++i;
            }
        }
        if (owned_count) pause_for(0.01);
    }
    for (size_t i = 0; i < owned_count; ++i) {
        pid_t pid = owned[i].pid;
        kill(owned[i].process_group ? -pid : pid, SIGKILL);
        (void)wait_child(pid, 0);
    }
    owned_count = 0;
    if (socket_path[0]) unlink(socket_path);
    if (mode_path[0]) unlink(mode_path);
    if (temp_dir[0]) {
        DIR *directory = opendir(temp_dir);
        if (directory) {
            struct dirent *entry;
            while ((entry = readdir(directory)) != NULL) {
                if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
                char path[PATH_MAX];
                if (join_path(path, sizeof path, temp_dir, entry->d_name) == 0) unlink(path);
            }
            closedir(directory);
        }
        rmdir(temp_dir);
    }
}

static void bail_out(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("Bail out! ", stdout);
    vprintf(format, args);
    putchar('\n');
    va_end(args);
    fflush(stdout);
    exit(255);
}

static void ok(int passed, const char *name) {
    ++test_number;
    if (passed) printf("ok %d - %s\n", test_number, name);
    else {
        printf("not ok %d - %s\n", test_number, name);
        test_failed = 1;
    }
}

static void diag(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("# ", stderr);
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
}

static void buffer_reserve(buffer *out, size_t more) {
    if (out->length + more + 1 <= out->capacity) return;
    size_t capacity = out->capacity ? out->capacity : 1024;
    while (capacity < out->length + more + 1) capacity *= 2;
    char *data = realloc(out->data, capacity);
    if (!data) bail_out("out of memory");
    out->data = data;
    out->capacity = capacity;
}

static void buffer_append(buffer *out, const void *data, size_t length) {
    buffer_reserve(out, length);
    memcpy(out->data + out->length, data, length);
    out->length += length;
    out->data[out->length] = 0;
}

static void buffer_printf(buffer *out, const char *format, ...) {
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    int length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (length < 0) { va_end(args); bail_out("formatting failed"); }
    buffer_reserve(out, (size_t)length);
    vsnprintf(out->data + out->length, out->capacity - out->length, format, args);
    out->length += (size_t)length;
    va_end(args);
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return strdup("");
    if (fseek(file, 0, SEEK_END) < 0) { fclose(file); return strdup(""); }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) < 0) { fclose(file); return strdup(""); }
    char *data = malloc((size_t)size + 1);
    if (!data) { fclose(file); bail_out("out of memory"); }
    size_t got = fread(data, 1, (size_t)size, file);
    fclose(file);
    data[got] = 0;
    return data;
}

static int write_text(const char *path, const char *text) {
    char temp[PATH_MAX];
    char suffix[32];
    int suffix_length = snprintf(suffix, sizeof suffix, ".tmp.%ld", (long)getpid());
    size_t path_length = strlen(path);
    if (suffix_length < 0 || (size_t)suffix_length >= sizeof suffix
        || path_length + (size_t)suffix_length + 1 > sizeof temp) return -1;
    memcpy(temp, path, path_length);
    memcpy(temp + path_length, suffix, (size_t)suffix_length + 1);
    FILE *file = fopen(temp, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int result = fwrite(text, 1, length, file) == length ? 0 : -1;
    if (fclose(file) != 0) result = -1;
    if (result == 0 && rename(temp, path) < 0) result = -1;
    if (result < 0) unlink(temp);
    return result;
}

static void save_core_log(const char *path, const char *mode, pid_t pid,
                          int include_received, size_t received, int eof) {
    char json[256];
    if (include_received)
        snprintf(json, sizeof json,
                 "{\"mode\":\"%s\",\"accepted\":1,\"pid\":%ld,\"received_bytes\":%zu,\"eof\":%d}",
                 mode, (long)pid, received, eof);
    else
        snprintf(json, sizeof json, "{\"mode\":\"%s\",\"accepted\":1,\"pid\":%ld}",
                 mode, (long)pid);
    (void)write_text(path, json);
}

static int write_all(int fd, const void *data, size_t length) {
    const char *bytes = data;
    size_t offset = 0;
    while (offset < length) {
        ssize_t wrote = send(fd, bytes + offset, length - offset, MSG_NOSIGNAL);
        if (wrote < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (wrote == 0) { errno = EIO; return -1; }
        offset += (size_t)wrote;
    }
    return 0;
}

static int sink_handler(int fd, const char *log_path, const char *mode) {
    pause_for(2.0);
    size_t received = 0;
    int eof = 0;
    double until = monotonic_time() + 3.0;
    while (monotonic_time() < until) {
        struct pollfd item = {.fd = fd, .events = POLLIN | POLLHUP};
        int ready = poll(&item, 1, 50);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return 1;
        }
        if (!ready) continue;
        char bytes[65536];
        ssize_t got = read(fd, bytes, sizeof bytes);
        if (got < 0) {
            if (errno == EINTR) continue;
            perror("sink read");
            return 1;
        }
        if (got == 0) { eof = 1; break; }
        received += (size_t)got;
    }
    save_core_log(log_path, mode, getpid(), 1, received, eof);
    close(fd);
    return 0;
}

static int core_handler(int fd, const char *mode, const char *dir) {
    alarm(7);
    char log_path[PATH_MAX];
    char leaf[128];
    snprintf(leaf, sizeof leaf, "core-%.16s-%ld.json", mode, (long)getpid());
    if (join_path(log_path, sizeof log_path, dir, leaf) < 0) return 1;
    save_core_log(log_path, mode, getpid(), 0, 0, 0);
    if (!strcmp(mode, "sink")) return sink_handler(fd, log_path, mode);

    FILE *client = fdopen(fd, "r+");
    if (!client) { close(fd); return 1; }
    setvbuf(client, NULL, _IOLBF, 0);
    char *line = NULL;
    size_t capacity = 0;
    int result = 0;
    while (getline(&line, &capacity, client) >= 0) {
        size_t length = strlen(line);
        while (length && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = 0;
        if (!strcmp(line, "info")) {
            if (write_all(fd, "info 262144 0\n", strlen("info 262144 0\n")) < 0) {
                result = 1; break;
            }
        } else if (!strncmp(line, "timings", 7) || !strncmp(line, "cancel_on_eof", 13)) {
            char response[8192];
            int n = snprintf(response, sizeof response, "ok: %s\n", line);
            if (n < 0 || n >= (int)sizeof response
                || write_all(fd, response, (size_t)n) < 0) { result = 1; break; }
        } else if (!strncmp(line, "stop", 4) || !strncmp(line, "sample", 6)
                   || !strncmp(line, "snap", 4)) {
            if (write_all(fd, "ok\n", 3) < 0) { result = 1; break; }
        } else if (!strncmp(line, "chat ", 5)) {
            int fields = 0;
            char *copy = strdup(line);
            if (!copy) { result = 1; break; }
            for (char *word = strtok(copy, " \t"); word; word = strtok(NULL, " \t"))
                ++fields;
            free(copy);
            int np = fields - 2;
            if (!strcmp(mode, "stall") || !strcmp(mode, "laggard")) {
                if (write_all(fd, "tok 2\n", 6) < 0) { result = 1; break; }
                while (getline(&line, &capacity, client) >= 0) { }
                if (!strcmp(mode, "laggard")) pause_for(2.5);
                char response[256];
                int n = snprintf(response, sizeof response, "done 1 0 %d 1000 0.1 0 0\n", np);
                if (n > 0 && n < (int)sizeof response) (void)write_all(fd, response, (size_t)n);
                save_core_log(log_path, mode, getpid(), 1, 0, 1);
            } else {
                char response[256];
                int n = snprintf(response, sizeof response,
                                 "tok 2\ntok 0\ndone 2 1 %d 1.0 0.1 0 0\n", np);
                if (n < 0 || n >= (int)sizeof response
                    || write_all(fd, response, (size_t)n) < 0) result = 1;
            }
            break;
        } else {
            fprintf(stderr, "unexpected fake core command: %s\n", line);
            result = 1;
            break;
        }
    }
    free(line);
    fclose(client);
    return result;
}

static void core_server(int listener) {
    signal(SIGCHLD, SIG_IGN);
    for (;;) {
        int fd = accept(listener, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            _exit(1);
        }
        char *mode_text = read_file(mode_path);
        char mode[32] = "quick";
        if (mode_text) {
            size_t n = strcspn(mode_text, "\r\n");
            if (n < sizeof mode) {
                memcpy(mode, mode_text, n);
                mode[n] = 0;
            }
            free(mode_text);
        }
        if (strcmp(mode, "quick") && strcmp(mode, "sink") && strcmp(mode, "stall")
            && strcmp(mode, "laggard")) strcpy(mode, "quick");
        (void)write_text(mode_path, "quick\n");
        pid_t child = fork();
        if (child < 0) { close(fd); continue; }
        if (child == 0) {
            close(listener);
            int result = core_handler(fd, mode, temp_dir);
            _exit(result ? 1 : 0);
        }
        close(fd);
    }
}

static int connect_with_timeout(int fd, const struct sockaddr *address, socklen_t length,
                                double timeout) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -1;
    int result = connect(fd, address, length);
    if (result < 0 && errno != EINPROGRESS) return -1;
    if (result < 0) {
        struct pollfd item = {.fd = fd, .events = POLLOUT};
        int ready = poll(&item, 1, (int)(timeout * 1000));
        if (ready <= 0) { errno = ready == 0 ? ETIMEDOUT : errno; return -1; }
        int error = 0;
        socklen_t size = sizeof error;
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) < 0) return -1;
        if (error) { errno = error; return -1; }
    }
    return fcntl(fd, F_SETFL, flags);
}

static int http_request(const char *text, int stream, http_result *result) {
    memset(result, 0, sizeof *result);
    result->body = strdup("");
    result->error = strdup("");
    double started = monotonic_time();
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { result->error = strdup("HTTP socket failed"); return -1; }
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)tcp_port);
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (connect_with_timeout(fd, (struct sockaddr *)&address, sizeof address, 1.0) < 0) {
        char error[256];
        snprintf(error, sizeof error, "HTTP connect: %s", strerror(errno));
        free(result->error); result->error = strdup(error);
        close(fd);
        return -1;
    }

    buffer body = {0}, wire = {0}, response = {0};
    buffer_printf(&body,
                  "{\"model\":\"deadline-test\",\"max_tokens\":2,\"stream\":%s,\"messages\":[{\"role\":\"user\",\"content\":\"",
                  stream ? "true" : "false");
    buffer_append(&body, text, strlen(text));
    buffer_append(&body, "\"}]}", 4);
    buffer_printf(&wire,
                  "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                  "Content-Type: application/json\r\nConnection: close\r\n"
                  "Content-Length: %zu\r\n\r\n", body.length);
    buffer_append(&wire, body.data, body.length);
    if (write_all(fd, wire.data, wire.length) < 0) {
        char error[256];
        snprintf(error, sizeof error, "HTTP write: %s", strerror(errno));
        free(result->error); result->error = strdup(error);
        goto done;
    }

    for (;;) {
        double remaining = 6.0 - (monotonic_time() - started);
        if (remaining <= 0) {
            free(result->error); result->error = strdup("HTTP response timed out");
            break;
        }
        struct pollfd item = {.fd = fd, .events = POLLIN | POLLHUP};
        int ready = poll(&item, 1, (int)(remaining < 0.05 ? remaining * 1000 : 50));
        if (ready < 0) {
            if (errno == EINTR) continue;
            char error[256];
            snprintf(error, sizeof error, "HTTP read: %s", strerror(errno));
            free(result->error); result->error = strdup(error);
            break;
        }
        if (!ready) continue;
        char bytes[65536];
        ssize_t got = read(fd, bytes, sizeof bytes);
        if (got < 0) {
            if (errno == EINTR) continue;
            char error[256];
            snprintf(error, sizeof error, "HTTP read: %s", strerror(errno));
            free(result->error); result->error = strdup(error);
            break;
        }
        if (got == 0) break;
        buffer_append(&response, bytes, (size_t)got);
    }
    if (response.data) {
        int status = 0;
        if (sscanf(response.data, "HTTP/%*d.%*d %d", &status) == 1) result->status = status;
        char *body_start = strstr(response.data, "\r\n\r\n");
        if (!body_start) body_start = strstr(response.data, "\n\n");
        if (body_start) {
            body_start = strstr(body_start, "\r\n\r\n") == body_start ? body_start + 4 : body_start + 2;
            free(result->body);
            result->body = strdup(body_start);
        }
    }
done:
    result->elapsed = monotonic_time() - started;
    close(fd);
    free(body.data);
    free(wire.data);
    free(response.data);
    return result->error && result->error[0] ? -1 : 0;
}

static void save_http_result(const char *path, const http_result *result) {
    char temp[PATH_MAX];
    char suffix[32];
    int suffix_length = snprintf(suffix, sizeof suffix, ".tmp.%ld", (long)getpid());
    size_t path_length = strlen(path);
    if (suffix_length < 0 || (size_t)suffix_length >= sizeof suffix
        || path_length + (size_t)suffix_length + 1 > sizeof temp) return;
    memcpy(temp, path, path_length);
    memcpy(temp + path_length, suffix, (size_t)suffix_length + 1);
    FILE *file = fopen(temp, "wb");
    if (!file) return;
    const char *error = result->error ? result->error : "";
    const char *body = result->body ? result->body : "";
    size_t error_length = strlen(error), body_length = strlen(body);
    fprintf(file, "%d\n%.9f\n%zu\n%zu\n", result->status, result->elapsed,
            error_length, body_length);
    fwrite(error, 1, error_length, file);
    fwrite(body, 1, body_length, file);
    fclose(file);
    rename(temp, path);
}

static int load_http_result(const char *path, http_result *result) {
    char *text = read_file(path);
    if (!text || !*text) { free(text); return 0; }
    char *cursor = text;
    char *line = strchr(cursor, '\n');
    if (!line) { free(text); return 0; }
    result->status = atoi(cursor);
    cursor = line + 1;
    line = strchr(cursor, '\n');
    if (!line) { free(text); return 0; }
    result->elapsed = atof(cursor);
    cursor = line + 1;
    line = strchr(cursor, '\n');
    if (!line) { free(text); return 0; }
    size_t error_length = (size_t)strtoull(cursor, NULL, 10);
    cursor = line + 1;
    line = strchr(cursor, '\n');
    if (!line) { free(text); return 0; }
    size_t body_length = (size_t)strtoull(cursor, NULL, 10);
    cursor = line + 1;
    if (strlen(cursor) < error_length + body_length) { free(text); return 0; }
    result->error = strndup(cursor, error_length);
    result->body = strndup(cursor + error_length, body_length);
    free(text);
    return result->error && result->body;
}

static int create_frontend(const char *binary, const char *gguf, const char *sock,
                           int port, const char *dir) {
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        setenv("FLASHD_REQUEST_MAX_S", "1", 1);
        setenv("FLASHD_THINKING", "0", 1);
        setenv("FLASHD_GREEDY", "1", 1);
        char out[PATH_MAX], err[PATH_MAX], address[128];
        if (join_path(out, sizeof out, dir, "frontend.stdout") < 0
            || join_path(err, sizeof err, dir, "frontend.stderr") < 0) _exit(126);
        int outfd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        int errfd = open(err, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (outfd < 0 || errfd < 0 || dup2(outfd, STDOUT_FILENO) < 0
            || dup2(errfd, STDERR_FILENO) < 0) _exit(126);
        close(outfd); close(errfd);
        snprintf(address, sizeof address, "127.0.0.1:%d", port);
        execl(binary, binary, gguf, sock, address, (char *)NULL);
        perror(binary);
        _exit(127);
    }
    track_child(child, 0);
    return child;
}

static int socket_ready(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    int result = connect_with_timeout(fd, (struct sockaddr *)&address, sizeof address, 0.1) == 0;
    close(fd);
    return result;
}

static int reserve_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&address, sizeof address) < 0 || listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }
    socklen_t length = sizeof address;
    if (getsockname(fd, (struct sockaddr *)&address, &length) < 0) {
        close(fd);
        return -1;
    }
    int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

static char *core_log_path(const char *mode) {
    char *path = malloc(PATH_MAX);
    if (!path) bail_out("out of memory");
    char leaf[64];
    snprintf(leaf, sizeof leaf, "core-%.16s-*.json", mode);
    if (join_path(path, PATH_MAX, temp_dir, leaf) < 0) {
        free(path);
        return NULL;
    }
    return path;
}

static char *find_core_log(const char *mode) {
    char *pattern = core_log_path(mode);
    if (!pattern) return NULL;
    glob_t matches;
    memset(&matches, 0, sizeof matches);
    char *found = NULL;
    if (glob(pattern, 0, NULL, &matches) == 0 && matches.gl_pathc)
        found = strdup(matches.gl_pathv[0]);
    globfree(&matches);
    free(pattern);
    return found;
}

static int mode_file_write(const char *mode) {
    char value[64];
    snprintf(value, sizeof value, "%s\n", mode);
    return write_text(mode_path, value);
}

static pid_t spawn_http(const char *save_path, const char *text, int stream) {
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        http_result result;
        (void)http_request(text, stream, &result);
        save_http_result(save_path, &result);
        _exit(0);
    }
    track_child(child, 0);
    return child;
}

static void start_core_server(void) {
    int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) bail_out("fake core socket: %s", strerror(errno));
    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof address.sun_path) bail_out("fake core socket path too long");
    strcpy(address.sun_path, socket_path);
    unlink(socket_path);
    if (bind(listener, (struct sockaddr *)&address, sizeof address) < 0
        || listen(listener, 16) < 0) bail_out("listen %s: %s", socket_path, strerror(errno));
    pid_t child = fork();
    if (child < 0) bail_out("fork: %s", strerror(errno));
    if (child == 0) {
        setpgid(0, 0);
        core_server(listener);
        _exit(0);
    }
    track_child(child, 1);
    close(listener);
}

static pid_t start_saved_http(const char *path, const char *text, int stream) {
    return spawn_http(path, text, stream);
}

static int run_test(void) {
    if (access(binary_path, X_OK) < 0) bail_out("flashd is not executable: %s", binary_path);
    if (!mkdtemp(temp_dir)) bail_out("mkdtemp: %s", strerror(errno));
    if (join_path(socket_path, sizeof socket_path, temp_dir, "engine.sock") < 0
        || join_path(mode_path, sizeof mode_path, temp_dir, "next-mode") < 0)
        bail_out("test directory path is too long");
    owner_pid = getpid();
    atexit(cleanup);
    signal(SIGPIPE, SIG_IGN);

    char fixture_path[PATH_MAX];
    if (join_path(fixture_path, sizeof fixture_path, temp_dir, "tokenizer.gguf") < 0)
        bail_out("test fixture path is too long");
    FILE *fixture = fopen(fixture_path, "wb");
    if (!fixture || issue80_write_fixture(fixture) < 0 || fclose(fixture) != 0)
        bail_out("cannot write test tokenizer");

    tcp_port = reserve_port();
    if (tcp_port < 0) bail_out("reserve port: %s", strerror(errno));
    if (mode_file_write("quick") < 0) bail_out("write mode file: %s", strerror(errno));
    start_core_server();

    (void)create_frontend(binary_path, fixture_path, socket_path, tcp_port, temp_dir);
    double ready_until = monotonic_time() + 3.0;
    while (monotonic_time() < ready_until && !socket_ready(tcp_port)) pause_for(0.01);
    if (!socket_ready(tcp_port)) {
        char path[PATH_MAX];
        if (join_path(path, sizeof path, temp_dir, "frontend.stderr") < 0)
            bail_out("frontend error path is too long");
        char *error = read_file(path);
        bail_out("frontend did not listen:\n%s", error ? error : "");
    }

    http_result quick;
    (void)http_request("abc", 0, &quick);
    ok(quick.status == 200, "quick request succeeds");
    ok(quick.error && !quick.error[0], "quick request has no client I/O failure");
    ok(quick.body && strstr(quick.body, "\"content\":\"a\""),
       "quick response contains text");
    free(quick.body); free(quick.error);

    mode_file_write("sink");
    char slow_path[PATH_MAX];
    if (join_path(slow_path, sizeof slow_path, temp_dir, "slow-http.json") < 0)
        bail_out("slow response path is too long");
    char *large = malloc(250001);
    if (!large) bail_out("out of memory");
    memset(large, 'a', 250000);
    large[250000] = 0;
    pid_t slow_pid = start_saved_http(slow_path, large, 0);
    free(large);
    char *sink_path = NULL;
    double sink_until = monotonic_time() + 2.0;
    while (monotonic_time() < sink_until && !(sink_path = find_core_log("sink"))) pause_for(0.01);
    if (!sink_path) {
        char *slow = read_file(slow_path);
        char frontend_path[PATH_MAX];
        if (join_path(frontend_path, sizeof frontend_path, temp_dir, "frontend.stderr") < 0)
            bail_out("frontend error path is too long");
        char *error = read_file(frontend_path);
        bail_out("large request never reached the non-reading core:\n%s\n%s", slow, error);
    }
    pause_for(0.1);
    http_result queued;
    (void)http_request("abc", 0, &queued);
    if (slow_pid > 0) {
        (void)wait_child(slow_pid, 0);
        forget_child(slow_pid);
    }
    http_result slow;
    if (!load_http_result(slow_path, &slow)) bail_out("large-request client never completed");
    ok(slow.status == 504, "blocked engine write produces HTTP 504");
    ok(slow.body && strstr(slow.body, "request_timeout"),
       "blocked write reports request_timeout");
    ok(slow.elapsed < 2.0, "blocked write expires within two seconds");
    ok(!slow.error[0], "blocked-write response completes normally");
    ok(queued.status == 200, "next queued request progresses");
    double queue_ms = 0;
    char *queue_key = queued.body ? strstr(queued.body, "\"queue_ms\"") : NULL;
    if (queue_key) {
        queue_key = strchr(queue_key, ':');
        if (queue_key) queue_ms = strtod(queue_key + 1, NULL);
    }
    ok(queue_ms > 0, "second request actually waited in the queue");
    double sink_deadline = monotonic_time() + 4.0;
    char *sink_json = NULL;
    while (monotonic_time() < sink_deadline) {
        sink_json = read_file(sink_path);
        if (sink_json && strstr(sink_json, "received_bytes")) break;
        free(sink_json); sink_json = NULL;
        pause_for(0.01);
    }
    int sink_eof = sink_json && strstr(sink_json, "\"eof\":1");
    size_t received = 0;
    if (sink_json) {
        char *key = strstr(sink_json, "\"received_bytes\":");
        if (key) received = (size_t)strtoull(strchr(key, ':') + 1, NULL, 10);
    }
    ok(sink_eof, "non-reading core eventually observes input EOF");
    ok(received > 0, "frontend sent a nonzero portion of the protocol");
    ok(received < 500007, "frontend abandoned the write before the full large chat line");
    diag("blocked write %.3f s; next request queue %.3f ms; partial engine bytes %zu",
         slow.elapsed, queue_ms, received);
    free(sink_json); free(sink_path); free(slow.body); free(slow.error);
    free(queued.body); free(queued.error);

    mode_file_write("stall");
    http_result sse;
    (void)http_request("abc", 1, &sse);
    ok(sse.status == 200, "stream begins with HTTP 200");
    ok(sse.body && strstr(sse.body, "\"content\":\"a\""), "stream emits text before stalling");
    ok(sse.body && strstr(sse.body, "request_timeout"), "running stream reports request_timeout");
    int done_count = 0;
    if (sse.body) {
        const char *where = sse.body;
        while ((where = strstr(where, "[DONE]")) != NULL) { ++done_count; where += 6; }
    }
    ok(done_count == 1, "timed-out stream emits exactly one DONE");
    ok(sse.body && strstr(sse.body, "\"aborted\":true"), "stream timing reports http.aborted");
    ok(sse.elapsed < 2.0, "cooperative stream cancellation drains promptly");
    ok(!sse.error[0], "stream response closes without client timeout");
    free(sse.body); free(sse.error);

    mode_file_write("laggard");
    char draining_path[PATH_MAX];
    if (join_path(draining_path, sizeof draining_path, temp_dir, "draining-http.json") < 0)
        bail_out("draining response path is too long");
    pid_t draining_pid = start_saved_http(draining_path, "abc", 0);
    char *laggard_log = NULL;
    double laggard_until = monotonic_time() + 2.0;
    while (monotonic_time() < laggard_until && !(laggard_log = find_core_log("laggard"))) pause_for(0.01);
    if (!laggard_log) bail_out("draining request never reached its core");
    glob_t before = {0};
    char pattern[PATH_MAX];
    if (join_path(pattern, sizeof pattern, temp_dir, "core-*.json") < 0)
        bail_out("core log pattern is too long");
    glob(pattern, 0, NULL, &before);
    size_t before_count = before.gl_pathc;
    http_result expired;
    (void)http_request("abc", 0, &expired);
    ok(expired.status == 504, "queued request expires on its own deadline");
    ok(expired.body && strstr(expired.body, "request_timeout"), "queue expiry reports request_timeout");
    ok(expired.elapsed < 2.0, "queue expiry does not wait for core draining");
    glob_t after = {0};
    glob(pattern, 0, NULL, &after);
    ok(after.gl_pathc == before_count, "expired queued request never connects to the core");
    globfree(&before); globfree(&after);
    if (draining_pid > 0) {
        double drain_until = monotonic_time() + 4.0;
        while (monotonic_time() < drain_until && access(draining_path, F_OK) < 0) pause_for(0.01);
    }
    http_result drained;
    if (!load_http_result(draining_path, &drained)) bail_out("draining request did not complete");
    ok(drained.status == 504, "noncooperative cancellation eventually returns 504");
    ok(drained.elapsed < 4.2, "cancellation drain has a fixed bound");
    if (draining_pid > 0) {
        (void)wait_child(draining_pid, 0);
        forget_child(draining_pid);
    }
    http_result recovered;
    (void)http_request("abc", 0, &recovered);
    ok(recovered.status == 200, "engine queue recovers after cancellation drain expires");
    free(expired.body); free(expired.error); free(drained.error);
    free(recovered.body); free(recovered.error); free(laggard_log);
    printf("1..%d\n", test_number);
    return test_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *binary = argc > 1 ? argv[1] : "./flashd";
    if (snprintf(binary_path, sizeof binary_path, "%s", binary) >= (int)sizeof binary_path) {
        fprintf(stderr, "flashd path too long\n");
        return 1;
    }
    strcpy(temp_dir, "/tmp/flashd-deadline-XXXXXX");
    owner_pid = getpid();
    return run_test();
}
