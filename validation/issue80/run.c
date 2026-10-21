#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    pid_t core;
    pid_t flashd;
    int port;
    char mode[16];
    char socket_path[PATH_MAX];
} service;

static char flashd_path[PATH_MAX];
static char gguf_arg[PATH_MAX];
static char fake_core_path[PATH_MAX];
static char out_dir[PATH_MAX];
static int failures;

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

static void sleep_seconds(double seconds) {
    struct timespec request = {
        .tv_sec = (time_t)seconds,
        .tv_nsec = (long)((seconds - (time_t)seconds) * 1e9),
    };
    while (nanosleep(&request, &request) < 0 && errno == EINTR) { }
}

static int wait_child(pid_t pid) {
    if (pid <= 0) return 0;
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return strdup("");
    if (fseek(file, 0, SEEK_END) < 0) { fclose(file); return strdup(""); }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) < 0) { fclose(file); return strdup(""); }
    char *data = malloc((size_t)length + 1);
    if (!data) { fclose(file); return NULL; }
    size_t got = fread(data, 1, (size_t)length, file);
    fclose(file);
    data[got] = 0;
    return data;
}

static bool mkdir_parents(const char *path) {
    char buffer[PATH_MAX];
    if (strlen(path) >= sizeof buffer) return false;
    strcpy(buffer, path);
    for (char *p = buffer + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(buffer, 0777) < 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return mkdir(buffer, 0777) == 0 || errno == EEXIST;
}

static int run_curl(const char *url, const char *body_path, const char *status_path,
                    const char *request) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        int output = open(status_path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (output < 0 || dup2(output, STDOUT_FILENO) < 0) _exit(126);
        close(output);
        char *const args[] = {
            "curl", "-sS", "-m", "60", "-o", (char *)body_path,
            "-w", "%{http_code} %{time_total}\n", (char *)url,
            "-H", "Content-Type: application/json", "-d", (char *)request, NULL,
        };
        execvp("curl", args);
        perror("curl");
        _exit(127);
    }
    return wait_child(pid);
}

static int health_check(int port) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        int nullfd = open("/dev/null", O_WRONLY);
        if (nullfd >= 0) {
            dup2(nullfd, STDOUT_FILENO);
            dup2(nullfd, STDERR_FILENO);
            close(nullfd);
        }
        char url[128];
        snprintf(url, sizeof url, "http://127.0.0.1:%d/health", port);
        char *const args[] = {"curl", "-s", "-m", "1", url, NULL};
        execvp("curl", args);
        _exit(127);
    }
    return wait_child(pid);
}

static int path_is_socket(const char *path) {
    struct stat st;
    return lstat(path, &st) == 0 && S_ISSOCK(st.st_mode);
}

static void stop_service(service *s) {
    if (s->flashd > 0) kill(s->flashd, SIGTERM);
    if (s->core > 0) kill(s->core, SIGTERM);
    if (s->flashd > 0) wait_child(s->flashd);
    if (s->core > 0) wait_child(s->core);
    s->flashd = 0;
    s->core = 0;
}

static int start_service(service *s, const char *mode, const char *limit) {
    memset(s, 0, sizeof *s);
    snprintf(s->mode, sizeof s->mode, "%s", mode);
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    unsigned int seed = (unsigned int)now.tv_nsec ^ (unsigned int)getpid();
    s->port = 18600 + (int)(rand_r(&seed) % 300);
    char leaf[64];
    snprintf(leaf, sizeof leaf, "core-%s.sock", mode);
    if (join_path(s->socket_path, sizeof s->socket_path, out_dir, leaf) < 0) return -1;

    char log_path[PATH_MAX];
    snprintf(leaf, sizeof leaf, "core-%s.log", mode);
    if (join_path(log_path, sizeof log_path, out_dir, leaf) < 0) return -1;
    FILE *log = fopen(log_path, "w");
    if (!log) {
        perror(log_path);
        return -1;
    }
    fclose(log);

    s->core = fork();
    if (s->core < 0) return -1;
    if (s->core == 0) {
        execl(fake_core_path, "fake_core", s->socket_path, mode, log_path, (char *)NULL);
        perror(fake_core_path);
        _exit(127);
    }
    int ready = 0;
    for (int i = 0; i < 600; ++i) {
        if (path_is_socket(s->socket_path)) { ready = 1; break; }
        if (waitpid(s->core, NULL, WNOHANG) == s->core) {
            s->core = 0;
            break;
        }
        sleep_seconds(0.1);
    }
    if (!ready) {
        puts("FAIL fake core did not start");
        failures = 1;
        stop_service(s);
        return -1;
    }

    char err_path[PATH_MAX];
    snprintf(leaf, sizeof leaf, "flashd-%s.err", mode);
    if (join_path(err_path, sizeof err_path, out_dir, leaf) < 0) {
        stop_service(s);
        return -1;
    }
    s->flashd = fork();
    if (s->flashd < 0) {
        stop_service(s);
        return -1;
    }
    if (s->flashd == 0) {
        int errfd = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (errfd < 0 || dup2(errfd, STDERR_FILENO) < 0) _exit(126);
        close(errfd);
        char address[128];
        snprintf(address, sizeof address, "127.0.0.1:%d", s->port);
        setenv("FLASHD_REQUEST_MAX_S", limit, 1);
        execl(flashd_path, flashd_path, gguf_arg, s->socket_path, address, (char *)NULL);
        perror(flashd_path);
        _exit(127);
    }

    for (int i = 0; i < 600; ++i) {
        if (health_check(s->port) == 0) return 0;
        if (kill(s->flashd, 0) < 0) {
            char *errors = read_file(err_path);
            printf("FAIL flashd exited: %s\n", errors ? errors : "");
            free(errors);
            failures = 1;
            stop_service(s);
            return -1;
        }
        sleep_seconds(0.5);
    }
    stop_service(s);
    return -1;
}

static void check(const char *name, int passed) {
    printf("%s %s\n", passed ? "PASS" : "FAIL", name);
    if (!passed) failures = 1;
}

static char *status_path(const char *name) {
    char *path = malloc(PATH_MAX);
    if (!path) return NULL;
    snprintf(path, PATH_MAX, "%s.status", name);
    return path;
}

static char *body_path(const char *name) {
    char *path = malloc(PATH_MAX);
    if (!path) return NULL;
    snprintf(path, PATH_MAX, "%s.body", name);
    return path;
}

static int request(service *s, const char *name, const char *stream) {
    char url[128];
    snprintf(url, sizeof url, "http://127.0.0.1:%d/v1/chat/completions", s->port);
    char *status = status_path(name);
    char *body = body_path(name);
    if (!status || !body) { free(status); free(body); return -1; }
    char payload[256];
    snprintf(payload, sizeof payload,
             "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"stream\":%s}",
             stream);
    int result = run_curl(url, body, status, payload);
    free(status);
    free(body);
    return result;
}

static pid_t start_request(service *s, const char *name, const char *stream) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) _exit(request(s, name, stream) == 0 ? 0 : 1);
    return pid;
}

static int parse_status(const char *name, int *code, double *seconds) {
    char *path = status_path(name);
    if (!path) return 0;
    char *text = read_file(path);
    free(path);
    if (!text) return 0;
    int count = sscanf(text, "%d %lf", code, seconds);
    free(text);
    return count == 2;
}

static char *read_body_named(const char *name) {
    char *path = body_path(name);
    if (!path) return NULL;
    char *text = read_file(path);
    free(path);
    return text;
}

static int has_line_match(const char *text, const char *pattern) {
    regex_t regex;
    if (regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB) != 0) return 0;
    int found = 0;
    const char *line = text;
    while (*line) {
        const char *end = strchr(line, '\n');
        size_t length = end ? (size_t)(end - line) : strlen(line);
        char *copy = strndup(line, length);
        if (copy && regexec(&regex, copy, 0, NULL, 0) == 0) found = 1;
        free(copy);
        if (!end) break;
        line = end + 1;
    }
    regfree(&regex);
    return found;
}

static int count_substring(const char *text, const char *needle) {
    int count = 0;
    size_t length = strlen(needle);
    if (!length) return 0;
    while ((text = strstr(text, needle)) != NULL) {
        ++count;
        text += length;
    }
    return count;
}

static void run_first_acceptance(service *s) {
    char *stream_status = status_path("stream");
    char *plain_status = status_path("plain");
    char *queued_status = status_path("queued");
    char *stream_body = read_body_named("stream");
    char *plain_body = read_body_named("plain");
    char *queued_body = read_body_named("queued");
    int stream_code = 0, plain_code = 0, queued_code = 0;
    double stream_time = 0, plain_time = 0, queued_time = 0;
    char *stream_status_text = stream_status ? read_file(stream_status) : strdup("");
    char *plain_status_text = plain_status ? read_file(plain_status) : strdup("");
    char *queued_status_text = queued_status ? read_file(queued_status) : strdup("");
    int have_stream = stream_status_text && sscanf(stream_status_text, "%d %lf", &stream_code, &stream_time) == 2;
    int have_plain = plain_status_text && sscanf(plain_status_text, "%d %lf", &plain_code, &plain_time) == 2;
    int have_queued = queued_status_text && sscanf(queued_status_text, "%d %lf", &queued_code, &queued_time) == 2;

    check("stream ends with request_timeout",
          stream_body && strstr(stream_body, "\"type\":\"request_timeout\"")
          && strstr(stream_body, "data: [DONE]"));
    check("stream cancelled after about 5 s",
          have_stream && stream_time >= 4.5 && stream_time < 9.0);
    check("non-streaming request gets 504", have_plain && plain_code == 504);
    check("queued request times out on its own deadline",
          have_queued && queued_code == 504 && queued_time >= 4.5 && queued_time < 6.0);

    char log_path[PATH_MAX];
    if (join_path(log_path, sizeof log_path, out_dir, "core-silent.log") < 0)
        log_path[0] = 0;
    char *core_log = read_file(log_path);
    check("core saw the running request hang up",
          core_log && has_line_match(core_log, "closed chat=1 after [4-8]\\."));
    check("a request whose deadline passed in the queue never reaches the core",
          core_log && count_substring(core_log, "closed chat=1") == 1);

    if (join_path(log_path, sizeof log_path, out_dir, "flashd-silent.err") < 0)
        log_path[0] = 0;
    char *flashd_log = read_file(log_path);
    check("flashd logged the cancellations",
          flashd_log && count_substring(flashd_log, "FLASHD_REQUEST_MAX_S") >= 3);

    free(stream_status); free(plain_status); free(queued_status);
    free(stream_body); free(plain_body); free(queued_body);
    free(stream_status_text); free(plain_status_text); free(queued_status_text);
    free(core_log); free(flashd_log);
    stop_service(s);
}

static void run_quick_acceptance(service *s) {
    int result = request(s, "quick", "false");
    int code = 0;
    double seconds = 0;
    int have_status = parse_status("quick", &code, &seconds);
    char *body = read_body_named("quick");
    check("a quick request still completes with 200",
          result == 0 && have_status && code == 200 && body
          && strstr(body, "\"finish_reason\":\"stop\""));
    free(body);
    stop_service(s);
}

static void print_trimmed_file(const char *path, const char *label) {
    char *text = read_file(path);
    if (!text) return;
    size_t length = strlen(text);
    while (length && text[length - 1] == '\n') text[--length] = 0;
    printf("%s: %s\n", label, text);
    free(text);
}

static void print_nonempty_lines(const char *path) {
    char *text = read_file(path);
    if (!text) return;
    const char *line = text;
    while (*line) {
        const char *end = strchr(line, '\n');
        size_t length = end ? (size_t)(end - line) : strlen(line);
        if (length) {
            fwrite(line, 1, length, stdout);
            putchar('\n');
        }
        if (!end) break;
        line = end + 1;
    }
    free(text);
}

static void print_results(void) {
    glob_t matches;
    memset(&matches, 0, sizeof matches);
    if (glob("*.status", 0, NULL, &matches) == 0) {
        for (size_t i = 0; i < matches.gl_pathc; ++i)
            print_trimmed_file(matches.gl_pathv[i], matches.gl_pathv[i]);
    }
    globfree(&matches);
    const char *patterns[] = {"core-*.log", "flashd-*.err"};
    for (size_t i = 0; i < sizeof patterns / sizeof patterns[0]; ++i) {
        memset(&matches, 0, sizeof matches);
        if (glob(patterns[i], 0, NULL, &matches) == 0)
            for (size_t j = 0; j < matches.gl_pathc; ++j)
                print_nonempty_lines(matches.gl_pathv[j]);
        globfree(&matches);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    char *self = realpath(argv[0], NULL);
    char *resolved_flashd = realpath(argv[1], NULL);
    if (!self || !resolved_flashd) {
        free(self); free(resolved_flashd);
        return 1;
    }
    char *slash = strrchr(self, '/');
    if (!slash) { free(self); free(resolved_flashd); return 1; }
    *slash = 0;
    if (snprintf(fake_core_path, sizeof fake_core_path, "%s/fake_core", self) >= (int)sizeof fake_core_path
        || snprintf(flashd_path, sizeof flashd_path, "%s", resolved_flashd) >= (int)sizeof flashd_path
        || snprintf(gguf_arg, sizeof gguf_arg, "%s", argv[2]) >= (int)sizeof gguf_arg) {
        free(self); free(resolved_flashd);
        return 1;
    }
    free(self);
    free(resolved_flashd);

    if (!mkdir_parents(argv[3])) return 1;
    char *resolved_out = realpath(argv[3], NULL);
    if (!resolved_out || snprintf(out_dir, sizeof out_dir, "%s", resolved_out) >= (int)sizeof out_dir) {
        free(resolved_out);
        return 1;
    }
    free(resolved_out);
    if (chdir(out_dir) < 0) return 1;

    service current = {0};
    if (start_service(&current, "silent", "5") < 0) return 1;
    pid_t stream_pid = start_request(&current, "stream", "true");
    sleep_seconds(1.0);
    pid_t plain_pid = start_request(&current, "plain", "false");
    pid_t queued_pid = start_request(&current, "queued", "false");
    if (stream_pid < 0 || plain_pid < 0 || queued_pid < 0) failures = 1;
    if (stream_pid > 0 && wait_child(stream_pid) != 0) failures = 1;
    if (plain_pid > 0 && wait_child(plain_pid) != 0) failures = 1;
    if (queued_pid > 0 && wait_child(queued_pid) != 0) failures = 1;
    run_first_acceptance(&current);

    if (start_service(&current, "answer", "300") < 0) return 1;
    if (request(&current, "quick", "false") < 0) failures = 1;
    run_quick_acceptance(&current);
    print_results();
    return failures;
}
