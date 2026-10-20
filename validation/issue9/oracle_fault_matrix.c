#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define ORACLE_FAULT_OUTPUT_LIMIT (16u * 1024u * 1024u)
#define ORACLE_FAULT_READ_CHUNK 8192u
#define ORACLE_FAULT_TOKEN_COUNT 9u
#define ORACLE_FAULT_CASE_COUNT 7u
#define ORACLE_FAULT_MAX_REVISION 63u
#define ORACLE_FAULT_PATH_LIMIT 4096u

typedef struct {
    const char *name;
    const char *inject;
    const char *reached;
    const char *flags;
    unsigned host_kv;
    unsigned expected_false_position;
    int single_launch;
} fault_case_t;

typedef struct {
    char *bytes;
    size_t length;
    pid_t process_group;
    int exit_code;
    int term_signal;
    int overflow;
    int local_timeout;
    int interrupted;
    int io_error;
    int pipe_eof;
    int group_live_after_cleanup;
} child_output_t;

typedef struct {
    unsigned marker_lines;
    unsigned marker_exact;
    unsigned scalar_count;
    unsigned true_count;
    unsigned false_count;
    uint64_t false_position;
    unsigned make_error_count;
    int make_error_code;
    unsigned capture_found;
    unsigned forbidden_found;
    unsigned malformed_scalar;
    unsigned malformed_make_error;
    unsigned positions_seen;
    unsigned char seen[ORACLE_FAULT_TOKEN_COUNT];
} validation_t;

static volatile sig_atomic_t caught_signal;

static const fault_case_t fault_cases[ORACLE_FAULT_CASE_COUNT] = {
    {"layer", "layer", "MODEL_INJECT=layer reached at b0.S.d0", "eh", 1u, 0u, 0},
    {"logits", "logits", "MODEL_INJECT=logits reached at logits.d0", "eh", 1u, 0u, 0},
    {"selection", "selection", "MODEL_INJECT=selection reached at b3.selected_ids", "eh", 1u, 7u, 0},
    {"dn", "dn", "MODEL_INJECT=dn reached at b0.dn_state.d0", "eh", 1u, 0u, 0},
    {"conv", "conv", "MODEL_INJECT=conv reached at b0.conv_state.d0", "eh", 1u, 0u, 0},
    {"ple", "ple", "MODEL_INJECT=ple reached at b1.ple_state.d0", "eh", 1u, 0u, 0},
    {"singlelaunch", "logits", "MODEL_INJECT=logits reached at logits.d0", "ehl", 0u, 8u, 1}
};

static const char *const timeout_argv[] = {
    "timeout", "--signal=TERM", "--kill-after=2s", "360s"
};

static size_t find_bytes(const char *haystack, size_t haystack_len,
                         const char *needle, size_t needle_len, size_t start) {
    if (!needle_len || needle_len > haystack_len ||
        start > haystack_len - needle_len) return SIZE_MAX;
    for (size_t i = start; i <= haystack_len - needle_len; i++)
        if (!memcmp(haystack + i, needle, needle_len)) return i;
    return SIZE_MAX;
}

static void on_signal(int signo) {
    if (!caught_signal) caught_signal = signo;
}

static int install_signal_handlers(void) {
    struct sigaction action;
    struct sigaction ignore;
    memset(&action, 0, sizeof action);
    sigemptyset(&action.sa_mask);
    action.sa_handler = on_signal;
    if (sigaction(SIGINT, &action, NULL) < 0 ||
        sigaction(SIGTERM, &action, NULL) < 0 ||
        sigaction(SIGHUP, &action, NULL) < 0) return -1;
    memset(&ignore, 0, sizeof ignore);
    sigemptyset(&ignore.sa_mask);
    ignore.sa_handler = SIG_IGN;
    return sigaction(SIGPIPE, &ignore, NULL);
}

static uint64_t monotonic_milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) return 0;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static int set_close_on_exec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = data;
    size_t offset = 0;
    while (offset < length) {
        ssize_t count = write(fd, bytes + offset, length - offset);
        if (count < 0 && errno == EINTR) {
            if (caught_signal) return -1;
            continue;
        }
        if (count <= 0) return -1;
        offset += (size_t)count;
    }
    return 0;
}

static int observe_child_exit(pid_t pid, int *exited) {
    siginfo_t info;
    memset(&info, 0, sizeof info);
    if (*exited) return 0;
    if (waitid(P_PID, (id_t)pid, &info, WEXITED | WNOHANG | WNOWAIT) < 0) {
        if (errno == EINTR) return 0;
        if (errno == ECHILD) { *exited = 1; return 0; }
        return -1;
    }
    if (info.si_pid == pid) *exited = 1;
    return 0;
}

static int read_proc_stat(pid_t pid, pid_t *process_group, char *state) {
    char path[64], contents[4096];
    size_t used = 0;
    int fd, n;
    char *right_paren;
    long parent_value, group_value;
    n = snprintf(path, sizeof path, "/proc/%ld/stat", (long)pid);
    if (n < 0 || (size_t)n >= sizeof path) return -1;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return errno == ENOENT || errno == ESRCH ? 1 : -1;
    while (used < sizeof contents - 1u) {
        ssize_t count = read(fd, contents + used, sizeof contents - 1u - used);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) { close(fd); return -1; }
        if (!count) break;
        used += (size_t)count;
    }
    if (close(fd) < 0) return -1;
    contents[used] = '\0';
    right_paren = strrchr(contents, ')');
    if (!right_paren || sscanf(right_paren + 1, " %c %ld %ld", state,
                               &parent_value, &group_value) != 3 ||
        group_value < 0 || group_value > INT_MAX) return -1;
    *process_group = (pid_t)group_value;
    return 0;
}

static int process_group_has_live_member(pid_t group) {
    DIR *proc = opendir("/proc");
    struct dirent *entry;
    int unknown = 0;
    if (!proc) return -1;
    while ((entry = readdir(proc)) != NULL) {
        char *end = NULL;
        long value;
        pid_t process_group;
        char state;
        int rc;
        if (!isdigit((unsigned char)entry->d_name[0])) continue;
        errno = 0;
        value = strtol(entry->d_name, &end, 10);
        if (errno || !end || *end || value <= 0 || value > INT_MAX) continue;
        rc = read_proc_stat((pid_t)value, &process_group, &state);
        if (rc == 1) continue;
        if (rc < 0) {
            if (errno != EACCES && errno != EPERM && errno != ENOENT && errno != ESRCH)
                unknown = 1;
            continue;
        }
        if (process_group == group && state != 'Z' && state != 'X') {
            closedir(proc);
            return 1;
        }
    }
    closedir(proc);
    return unknown ? -1 : 0;
}

static int process_is_live(pid_t pid) {
    pid_t process_group;
    char state;
    int rc = read_proc_stat(pid, &process_group, &state);
    (void)process_group;
    if (rc == 1) return 0;
    if (rc < 0) return -1;
    return state != 'Z' && state != 'X';
}

static int capture_output_chunk(child_output_t *result, const void *data,
                                size_t length, int echo_output) {
    size_t available = ORACLE_FAULT_OUTPUT_LIMIT - result->length;
    size_t kept = length < available ? length : available;
    if (kept) {
        memcpy(result->bytes + result->length, data, kept);
        result->length += kept;
        if (echo_output && write_all(STDOUT_FILENO, data, kept) < 0) {
            result->io_error = 1;
            return -1;
        }
    }
    if (kept != length) {
        result->overflow = 1;
        return 1;
    }
    return 0;
}

static void drain_pipe_after_cleanup(int fd, int echo_output,
                                     child_output_t *result) {
    uint64_t deadline = monotonic_milliseconds() + 2000u;
    while (!result->pipe_eof && monotonic_milliseconds() < deadline) {
        unsigned char chunk[ORACLE_FAULT_READ_CHUNK];
        ssize_t count = read(fd, chunk, sizeof chunk);
        if (count > 0) {
            if (capture_output_chunk(result, chunk, (size_t)count, echo_output) < 0)
                result->io_error = 1;
            continue;
        }
        if (count == 0) {
            result->pipe_eof = 1;
            return;
        }
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            result->io_error = 1;
            return;
        }
        {
            struct pollfd pollfd = {fd, POLLIN | POLLHUP | POLLERR, 0};
            int rc = poll(&pollfd, 1, 100);
            if (rc < 0 && errno != EINTR) {
                result->io_error = 1;
                return;
            }
        }
    }
    if (!result->pipe_eof) result->io_error = 1;
}

/* The child remains waitable while its group is retired. In particular, a
 * leader exit does not short-circuit the TERM grace or the unconditional group
 * SIGKILL; a pipe-holding descendant cannot outlive its launch group. */
static void terminate_and_reap(pid_t pid, int *status, int *leader_exited,
                               int *leader_reaped, int pipe_fd, int echo_output,
                               child_output_t *result) {
    uint64_t deadline;
    int live;
    int term_group_result, term_group_errno = 0;
    struct timespec pause_time = {0, 50000000L};
    term_group_result = kill(-pid, SIGTERM);
    if (term_group_result < 0) term_group_errno = errno;
    if (!*leader_exited) (void)kill(pid, SIGTERM);
    if (term_group_result == 0 || term_group_errno != ESRCH || !*leader_exited) {
        deadline = monotonic_milliseconds() + 2000u;
        do {
            if (observe_child_exit(pid, leader_exited) < 0) result->io_error = 1;
            live = process_group_has_live_member(pid);
            if (live < 0) result->io_error = 1;
            if (live == 0 && *leader_exited) break;
            (void)nanosleep(&pause_time, NULL);
        } while (monotonic_milliseconds() < deadline);
    }

    /* Always signal the owned process group, even after its leader exited. */
    (void)kill(-pid, SIGKILL);
    if (!*leader_exited) (void)kill(pid, SIGKILL);
    deadline = monotonic_milliseconds() + 2000u;
    for (;;) {
        live = process_group_has_live_member(pid);
        if (live == 0) break;
        if (live < 0) { result->io_error = 1; break; }
        if (monotonic_milliseconds() >= deadline) break;
        (void)nanosleep(&pause_time, NULL);
    }
    if (live != 0) {
        result->group_live_after_cleanup = live > 0;
        result->io_error = 1;
    }
    if (!*leader_reaped) {
        for (;;) {
            pid_t waited = waitpid(pid, status, 0);
            if (waited == pid) { *leader_reaped = 1; break; }
            if (waited < 0 && errno == EINTR) continue;
            if (waited < 0 && errno == ECHILD) *leader_reaped = 1;
            else result->io_error = 1;
            break;
        }
    }
    if (!result->pipe_eof) drain_pipe_after_cleanup(pipe_fd, echo_output, result);
}

static void child_output_init(child_output_t *result) {
    memset(result, 0, sizeof *result);
    result->exit_code = -1;
    result->bytes = malloc(ORACLE_FAULT_OUTPUT_LIMIT + 1u);
}

static void child_output_finish(child_output_t *result, int status) {
    if (!result->bytes) return;
    result->bytes[result->length] = '\0';
    if (WIFEXITED(status)) result->exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result->term_signal = WTERMSIG(status);
}

/* Run one argv vector in a private process group, capture merged stdout/stderr,
 * echo at most the fixed cap, and reap the command on timeout or interruption. */
static int run_process(char *const command[], unsigned hard_timeout_seconds,
                       int echo_output, child_output_t *result) {
    int pipefd[2] = {-1, -1};
    int old_flags, status = 0, leader_exited = 0, leader_reaped = 0;
    pid_t child;
    sigset_t block_set, old_mask;
    uint64_t started;

    child_output_init(result);
    if (!result->bytes) {
        fprintf(stderr, "oracle fault runner: allocate bounded output buffer failed\n");
        return -1;
    }
    if (pipe(pipefd) < 0 || set_close_on_exec(pipefd[0]) < 0 ||
        set_close_on_exec(pipefd[1]) < 0) {
        fprintf(stderr, "oracle fault runner: create output pipe: %s\n", strerror(errno));
        if (pipefd[0] >= 0) close(pipefd[0]);
        if (pipefd[1] >= 0) close(pipefd[1]);
        result->io_error = 1;
        return -1;
    }
    old_flags = fcntl(pipefd[0], F_GETFL);
    if (old_flags < 0 || fcntl(pipefd[0], F_SETFL, old_flags | O_NONBLOCK) < 0) {
        fprintf(stderr, "oracle fault runner: configure output pipe: %s\n", strerror(errno));
        close(pipefd[0]); close(pipefd[1]);
        result->io_error = 1;
        return -1;
    }
    sigemptyset(&block_set);
    sigaddset(&block_set, SIGINT);
    sigaddset(&block_set, SIGTERM);
    sigaddset(&block_set, SIGHUP);
    if (sigprocmask(SIG_BLOCK, &block_set, &old_mask) < 0) {
        fprintf(stderr, "oracle fault runner: block signals for child setup: %s\n", strerror(errno));
        close(pipefd[0]); close(pipefd[1]);
        result->io_error = 1;
        return -1;
    }
    child = fork();
    if (child < 0) {
        int saved = errno;
        (void)sigprocmask(SIG_SETMASK, &old_mask, NULL);
        close(pipefd[0]); close(pipefd[1]);
        fprintf(stderr, "oracle fault runner: fork: %s\n", strerror(saved));
        result->io_error = 1;
        return -1;
    }
    if (child == 0) {
        struct sigaction default_action;
        (void)close(pipefd[0]);
        if (setpgid(0, 0) < 0) _exit(126);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0 ||
            dup2(pipefd[1], STDERR_FILENO) < 0) _exit(126);
        (void)close(pipefd[1]);
        memset(&default_action, 0, sizeof default_action);
        sigemptyset(&default_action.sa_mask);
        default_action.sa_handler = SIG_DFL;
        (void)sigaction(SIGINT, &default_action, NULL);
        (void)sigaction(SIGTERM, &default_action, NULL);
        (void)sigaction(SIGHUP, &default_action, NULL);
        (void)sigprocmask(SIG_SETMASK, &old_mask, NULL);
        execvp(command[0], command);
        _exit(127);
    }
    result->process_group = child;
    close(pipefd[1]);
    pipefd[1] = -1;
    if (setpgid(child, child) < 0 && errno != EACCES && errno != ESRCH) {
        fprintf(stderr, "oracle fault runner: set child process group: %s\n", strerror(errno));
        result->io_error = 1;
        terminate_and_reap(child, &status, &leader_exited, &leader_reaped,
                           pipefd[0], echo_output, result);
        close(pipefd[0]);
        (void)sigprocmask(SIG_SETMASK, &old_mask, NULL);
        if (leader_reaped) child_output_finish(result, status);
        return -1;
    }
    if (sigprocmask(SIG_SETMASK, &old_mask, NULL) < 0) {
        fprintf(stderr, "oracle fault runner: restore signal mask: %s\n", strerror(errno));
        result->io_error = 1;
        terminate_and_reap(child, &status, &leader_exited, &leader_reaped,
                           pipefd[0], echo_output, result);
        close(pipefd[0]);
        if (leader_reaped) child_output_finish(result, status);
        return -1;
    }
    started = monotonic_milliseconds();

    while (!result->pipe_eof || !leader_exited) {
        struct pollfd pollfd;
        int poll_result;
        if (caught_signal) {
            result->interrupted = (int)caught_signal;
            goto cleanup;
        }
        if (!result->local_timeout &&
            monotonic_milliseconds() - started >= (uint64_t)hard_timeout_seconds * 1000u) {
            result->local_timeout = 1;
            goto cleanup;
        }
        pollfd.fd = result->pipe_eof ? -1 : pipefd[0];
        pollfd.events = POLLIN | POLLHUP | POLLERR;
        pollfd.revents = 0;
        poll_result = poll(&pollfd, 1, 100);
        if (poll_result < 0 && errno != EINTR) {
            result->io_error = 1;
            goto cleanup;
        }
        if (poll_result > 0 && !result->pipe_eof && pollfd.revents) {
            for (;;) {
                unsigned char chunk[ORACLE_FAULT_READ_CHUNK];
                ssize_t count = read(pipefd[0], chunk, sizeof chunk);
                if (count < 0 && errno == EINTR) {
                    if (caught_signal) break;
                    continue;
                }
                if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                if (count < 0) {
                    result->io_error = 1;
                    goto cleanup;
                }
                if (count == 0) {
                    result->pipe_eof = 1;
                    break;
                }
                {
                    int capture = capture_output_chunk(result, chunk, (size_t)count,
                                                       echo_output);
                    if (capture < 0 || capture > 0) goto cleanup;
                }
            }
        }
        if (observe_child_exit(child, &leader_exited) < 0) {
            result->io_error = 1;
            goto cleanup;
        }
    }
cleanup:
    /* Sanitize any same-group descendants even after normal leader completion. */
    terminate_and_reap(child, &status, &leader_exited, &leader_reaped,
                       pipefd[0], echo_output, result);
    close(pipefd[0]);
    if (leader_reaped) child_output_finish(result, status);
    if (result->bytes) result->bytes[result->length] = '\0';
    if (!leader_reaped) result->io_error = 1;
    return 0;
}

static int line_has(const char *line, size_t length, const char *needle) {
    return find_bytes(line, length, needle, strlen(needle), 0) != SIZE_MAX;
}

static int line_equals(const char *line, size_t length, const char *expected) {
    size_t expected_length = strlen(expected);
    return length == expected_length && !memcmp(line, expected, length);
}

static int starts_with(const char *line, size_t length, const char *prefix) {
    size_t prefix_length = strlen(prefix);
    return length >= prefix_length && !memcmp(line, prefix, prefix_length);
}

static void skip_json_whitespace(const char *line, size_t length, size_t *at) {
    while (*at < length && (line[*at] == ' ' || line[*at] == '\t' ||
                            line[*at] == '\r' || line[*at] == '\n')) (*at)++;
}

static int json_literal(const char *line, size_t length, size_t *at,
                        const char *literal) {
    size_t literal_length = strlen(literal);
    if (length - *at < literal_length ||
        memcmp(line + *at, literal, literal_length) != 0) return -1;
    *at += literal_length;
    return 0;
}

static int parse_json_unsigned(const char *line, size_t length, size_t *at,
                               uint64_t *value_out) {
    uint64_t value = 0;
    size_t start = *at;
    if (*at >= length || !isdigit((unsigned char)line[*at])) return -1;
    if (line[*at] == '0') {
        (*at)++;
        if (*at < length && isdigit((unsigned char)line[*at])) return -1;
    } else {
        while (*at < length && isdigit((unsigned char)line[*at])) {
            unsigned digit = (unsigned)(line[*at] - '0');
            if (value > (UINT64_MAX - digit) / 10u) return -1;
            value = value * 10u + digit;
            (*at)++;
        }
    }
    if (*at == start) return -1;
    *value_out = value;
    return 0;
}

static int json_key(const char *line, size_t length, size_t *at,
                    const char *key) {
    size_t key_length = strlen(key);
    skip_json_whitespace(line, length, at);
    if (json_literal(line, length, at, "\"") < 0 ||
        length - *at < key_length || memcmp(line + *at, key, key_length)) return -1;
    *at += key_length;
    if (json_literal(line, length, at, "\"") < 0) return -1;
    skip_json_whitespace(line, length, at);
    if (*at >= length || line[(*at)++] != ':') return -1;
    skip_json_whitespace(line, length, at);
    return 0;
}

static int parse_scalar_record(const char *line, size_t length,
                               uint64_t *position, uint64_t *visible,
                               int *match) {
    size_t at = 0;
    skip_json_whitespace(line, length, &at);
    if (at >= length || line[at++] != '{') return -1;
    if (json_key(line, length, &at, "position") < 0 ||
        parse_json_unsigned(line, length, &at, position) < 0) return -1;
    skip_json_whitespace(line, length, &at);
    if (at >= length || line[at++] != ',') return -1;
    if (json_key(line, length, &at, "visible_length") < 0 ||
        parse_json_unsigned(line, length, &at, visible) < 0) return -1;
    skip_json_whitespace(line, length, &at);
    if (at >= length || line[at++] != ',') return -1;
    if (json_key(line, length, &at, "scalar_match") < 0) return -1;
    if (json_literal(line, length, &at, "true") == 0) *match = 1;
    else if (json_literal(line, length, &at, "false") == 0) *match = 0;
    else return -1;
    skip_json_whitespace(line, length, &at);
    if (at >= length || line[at++] != '}') return -1;
    skip_json_whitespace(line, length, &at);
    return at == length ? 0 : -1;
}

static int parse_make_error(const char *line, size_t length, int *error_code) {
    size_t at = find_bytes(line, length, "Error ", 6u, 0);
    uint64_t value = 0;
    if (at == SIZE_MAX || !line_has(line, length, "model-oracle")) return -1;
    at += 6u;
    if (at == length || !isdigit((unsigned char)line[at])) return -1;
    while (at < length && isdigit((unsigned char)line[at])) {
        unsigned digit = (unsigned)(line[at] - '0');
        if (value > (UINT_MAX - digit) / 10u) return -1;
        value = value * 10u + digit;
        at++;
    }
    while (at < length && isspace((unsigned char)line[at])) at++;
    if (at != length) return -1;
    *error_code = (int)value;
    return 0;
}

static void validate_output(const child_output_t *output,
                            const fault_case_t *expectation,
                            validation_t *validation) {
    size_t offset = 0;
    static const char *const capture_markers[] = {
        "flash-sass.qsa-selected-blocks.v1",
        "actual_model_gpu_readback",
        "\"selected_block_ids\"",
        "\"selected_ids_match_scalar\"",
        "MODEL_CAPTURE_QSA:"
    };
    static const char *const forbidden_markers[] = {
        "cuInit failed", "CUDA_ERROR", "CUDA error", "CUDA Error",
        "CUDA initialization failed", "cuCtxSetCurrent failed",
        "out of memory", "No such file or directory", "command not found",
        "timed out", "timeout: sending signal", "setup failed", "SETUP_ERROR"
    };
    memset(validation, 0, sizeof *validation);
    validation->false_position = UINT64_MAX;
    for (size_t i = 0; i < sizeof capture_markers / sizeof capture_markers[0]; i++)
        if (find_bytes(output->bytes, output->length, capture_markers[i],
                       strlen(capture_markers[i]), 0) != SIZE_MAX)
            validation->capture_found = 1u;
    for (size_t i = 0; i < sizeof forbidden_markers / sizeof forbidden_markers[0]; i++)
        if (find_bytes(output->bytes, output->length, forbidden_markers[i],
                       strlen(forbidden_markers[i]), 0) != SIZE_MAX)
            validation->forbidden_found = 1u;

    while (offset < output->length) {
        size_t end = offset, line_length;
        const char *line = output->bytes + offset;
        while (end < output->length && output->bytes[end] != '\n') end++;
        line_length = end - offset;
        if (line_length && line[line_length - 1u] == '\r') line_length--;

        if (line_has(line, line_length, " reached at ")) {
            validation->marker_lines++;
            if (line_equals(line, line_length, expectation->reached))
                validation->marker_exact++;
        }
        if (line_has(line, line_length, "CUDA") &&
            !line_has(line, line_length, "CUDA_VISIBLE_DEVICES="))
            validation->forbidden_found = 1u;
        if (line_has(line, line_length, "\"scalar_match\"")) {
            uint64_t position, visible;
            int match;
            validation->scalar_count++;
            if (parse_scalar_record(line, line_length, &position, &visible, &match) < 0) {
                validation->malformed_scalar++;
            } else if (position >= ORACLE_FAULT_TOKEN_COUNT || visible != position + 1u ||
                       validation->seen[position]) {
                validation->malformed_scalar++;
            } else {
                validation->seen[position] = 1u;
                validation->positions_seen++;
                if (match) validation->true_count++;
                else {
                    validation->false_count++;
                    validation->false_position = position;
                }
            }
        }
        if (starts_with(line, line_length, "make") &&
            line_has(line, line_length, "***")) {
            int code;
            validation->make_error_count++;
            if (parse_make_error(line, line_length, &code) < 0)
                validation->malformed_make_error++;
            else validation->make_error_code = code;
        }
        if (end == output->length) break;
        offset = end + 1u;
    }
}

static int validation_passed(const child_output_t *output,
                             const fault_case_t *expectation,
                             const validation_t *validation) {
    unsigned wanted_count = expectation->single_launch ? 1u : ORACLE_FAULT_TOKEN_COUNT;
    unsigned wanted_true = expectation->single_launch ? 0u : ORACLE_FAULT_TOKEN_COUNT - 1u;
    if (output->exit_code != 2 || output->term_signal || output->overflow ||
        output->local_timeout || output->interrupted || output->io_error ||
        !output->pipe_eof || output->group_live_after_cleanup) return 0;
    if (validation->marker_lines != 1u || validation->marker_exact != 1u ||
        validation->scalar_count != wanted_count ||
        validation->true_count != wanted_true || validation->false_count != 1u ||
        validation->false_position != expectation->expected_false_position ||
        validation->malformed_scalar || validation->positions_seen != wanted_count ||
        validation->make_error_count != 1u || validation->make_error_code != 3 ||
        validation->malformed_make_error || validation->capture_found ||
        validation->forbidden_found) return 0;
    if (expectation->single_launch) {
        if (!validation->seen[8]) return 0;
        for (size_t i = 0; i < 8u; i++) if (validation->seen[i]) return 0;
    } else {
        for (size_t i = 0; i < ORACLE_FAULT_TOKEN_COUNT; i++)
            if (!validation->seen[i]) return 0;
    }
    return 1;
}

static void print_case_boundary(const fault_case_t *test_case, const char *kind) {
    printf("ORACLE_FAULT_CASE_%s {\"name\":\"%s\",\"inject\":\"%s\","
           "\"tokens\":9,\"blocks\":1,\"host_kv\":%u,\"flags\":\"%s\",\"capture\":0}\n",
           kind, test_case->name, test_case->inject, test_case->host_kv,
           test_case->flags);
    fflush(stdout);
}

static void print_case_result(const fault_case_t *test_case,
                              const child_output_t *output,
                              const validation_t *validation,
                              int accepted) {
    printf("ORACLE_FAULT_CASE_RESULT {\"name\":\"%s\",\"accepted\":%s,"
           "\"make_exit\":%d,\"signal\":%d,\"make_error\":%d,"
           "\"marker_lines\":%u,\"marker_exact\":%u,\"comparisons\":%u,"
           "\"matches\":%u,\"mismatches\":%u,\"mismatch_position\":",
           test_case->name, accepted ? "true" : "false", output->exit_code,
           output->term_signal, validation->make_error_code,
           validation->marker_lines, validation->marker_exact,
           validation->scalar_count, validation->true_count, validation->false_count);
    if (validation->false_position == UINT64_MAX) fputs("null", stdout);
    else printf("%" PRIu64, validation->false_position);
    printf(",\"capture_seen\":%s,\"output_bytes\":%zu,\"output_limit_exceeded\":%s,"
           "\"timed_out\":%s,\"interrupted\":%d,\"pipe_eof\":%s,"
           "\"group_live_after_cleanup\":%s}\n",
           validation->capture_found ? "true" : "false", output->length,
           output->overflow ? "true" : "false",
           (output->local_timeout || output->exit_code == 124 || output->exit_code == 137)
               ? "true" : "false",
           output->interrupted, output->pipe_eof ? "true" : "false",
           output->group_live_after_cleanup ? "true" : "false");
    fflush(stdout);
}

static int valid_make_path(const char *text) {
    if (!text || !*text) return 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (!isalnum(*p) && *p != '/' && *p != '.' && *p != '_' &&
            *p != '-' && *p != '+') return 0;
    }
    return 1;
}

static int valid_revision(const char *revision) {
    size_t length;
    if (!revision || !(length = strlen(revision)) || length > ORACLE_FAULT_MAX_REVISION) return 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)revision[i];
        if (!isalnum(c) && c != '.' && c != '_' && c != '-') return 0;
    }
    return 1;
}

static int make_command(const char *makefile, const char *oracle_data,
                        const char *revision, const fault_case_t *test_case,
                        char *data_arg, size_t data_arg_capacity,
                        char *flags_arg, size_t flags_arg_capacity,
                        char *revision_arg, size_t revision_arg_capacity,
                        char *inject_arg, size_t inject_arg_capacity,
                        char *command[], size_t command_capacity) {
    size_t count = 0;
    int n;
    for (size_t i = 0; i < sizeof timeout_argv / sizeof timeout_argv[0]; i++) {
        if (count + 1u >= command_capacity) return -1;
        command[count++] = (char *)timeout_argv[i];
    }
    command[count++] = "make";
    command[count++] = "-f";
    if (count + 14u >= command_capacity) return -1;
    command[count++] = (char *)makefile;
    command[count++] = "TOKENS=9";
    command[count++] = "BLOCKS=1";
    command[count++] = (char *)(test_case->host_kv ? "HOST_KV=1" : "HOST_KV=0");
    n = snprintf(flags_arg, flags_arg_capacity, "FLAGS=%s", test_case->flags);
    if (n < 0 || (size_t)n >= flags_arg_capacity) return -1;
    command[count++] = flags_arg;
    command[count++] = "CAPTURE=0";
    n = snprintf(data_arg, data_arg_capacity, "ORACLE_DATA=%s", oracle_data);
    if (n < 0 || (size_t)n >= data_arg_capacity) return -1;
    n = snprintf(revision_arg, revision_arg_capacity, "ORACLE_REVISION=%s", revision);
    if (n < 0 || (size_t)n >= revision_arg_capacity) return -1;
    n = snprintf(inject_arg, inject_arg_capacity, "INJECT=%s", test_case->inject);
    if (n < 0 || (size_t)n >= inject_arg_capacity) return -1;
    command[count++] = data_arg;
    command[count++] = revision_arg;
    command[count++] = inject_arg;
    command[count++] = "model-oracle";
    if (count >= command_capacity) return -1;
    command[count] = NULL;
    return 0;
}

static int command_matches(char *const actual[], const char *const expected[]) {
    size_t i;
    for (i = 0; i < 24u && expected[i]; i++)
        if (!actual[i] || strcmp(actual[i], expected[i])) return 0;
    return i < 24u && actual[i] == NULL;
}

static int self_test_make_argv(void) {
    static const char *const per_token_expected[] = {
        "timeout", "--signal=TERM", "--kill-after=2s", "360s", "make", "-f",
        "validation/issue9/Makefile", "TOKENS=9", "BLOCKS=1", "HOST_KV=1",
        "FLAGS=eh", "CAPTURE=0", "ORACLE_DATA=/fixture/oracle",
        "ORACLE_REVISION=0123456789abcdef0123456789abcdef01234567",
        "INJECT=layer", "model-oracle", NULL
    };
    static const char *const single_launch_expected[] = {
        "timeout", "--signal=TERM", "--kill-after=2s", "360s", "make", "-f",
        "validation/issue9/Makefile", "TOKENS=9", "BLOCKS=1", "HOST_KV=0",
        "FLAGS=ehl", "CAPTURE=0", "ORACLE_DATA=/fixture/oracle",
        "ORACLE_REVISION=0123456789abcdef0123456789abcdef01234567",
        "INJECT=logits", "model-oracle", NULL
    };
    char data_arg[128], flags_arg[16], revision_arg[96], inject_arg[32];
    char *command[24];
    if (make_command("validation/issue9/Makefile", "/fixture/oracle",
                     "0123456789abcdef0123456789abcdef01234567", &fault_cases[0],
                     data_arg, sizeof data_arg, flags_arg, sizeof flags_arg,
                     revision_arg, sizeof revision_arg, inject_arg, sizeof inject_arg,
                     command, sizeof command / sizeof command[0]) < 0 ||
        !command_matches(command, per_token_expected)) return -1;
    if (make_command("validation/issue9/Makefile", "/fixture/oracle",
                     "0123456789abcdef0123456789abcdef01234567", &fault_cases[6],
                     data_arg, sizeof data_arg, flags_arg, sizeof flags_arg,
                     revision_arg, sizeof revision_arg, inject_arg, sizeof inject_arg,
                     command, sizeof command / sizeof command[0]) < 0 ||
        !command_matches(command, single_launch_expected)) return -1;
    return 0;
}

static int fixture_emit(const char *mode) {
    unsigned false_at = 0;
    int omit_marker = !strcmp(mode, "missing-marker");
    int all_true = !strcmp(mode, "missing-mismatch");
    int add_capture = !strcmp(mode, "positive-capture");
    int wrong_exit = !strcmp(mode, "wrong-exit");
    int trailing_token = !strcmp(mode, "trailing-token");
    int trailing_garbage = !strcmp(mode, "trailing-garbage");
    int duplicate_key = !strcmp(mode, "duplicate-key");
    int extra_key = !strcmp(mode, "extra-key");
    if (!strcmp(mode, "descendant-holds-pipe")) {
        pid_t grandchild = fork();
        if (grandchild < 0) return 1;
        if (grandchild == 0) {
            char marker[96];
            int n;
            (void)signal(SIGTERM, SIG_IGN);
            n = snprintf(marker, sizeof marker, "FIXTURE_GRANDCHILD_PID=%ld PGID=%ld\n",
                         (long)getpid(), (long)getpgrp());
            if (n <= 0 || (size_t)n >= sizeof marker ||
                write_all(STDOUT_FILENO, marker, (size_t)n) < 0) _exit(1);
            for (;;) pause();
        }
        return 0;
    }
    if (strcmp(mode, "valid") && !omit_marker && !all_true && !add_capture &&
        !wrong_exit && !trailing_token && !trailing_garbage &&
        !duplicate_key && !extra_key)
        return 2;
    if (!omit_marker) puts("MODEL_INJECT=layer reached at b0.S.d0");
    for (unsigned position = 0; position < ORACLE_FAULT_TOKEN_COUNT; position++) {
        int match = all_true || position != false_at;
        if (position == 0u && trailing_token) {
            puts("{\"position\":0,\"visible_length\":1,\"scalar_match\":false true}");
            continue;
        }
        if (position == 0u && trailing_garbage) {
            puts("{\"position\":0,\"visible_length\":1,\"scalar_match\":false} junk");
            continue;
        }
        if (position == 0u && duplicate_key) {
            puts("{\"position\":0,\"visible_length\":1,\"scalar_match\":false,\"scalar_match\":false}");
            continue;
        }
        if (position == 0u && extra_key) {
            puts("{\"position\":0,\"visible_length\":1,\"scalar_match\":false,\"extra\":0}");
            continue;
        }
        printf("{\"position\":%u,\"visible_length\":%u,\"scalar_match\":%s}\n",
               position, position + 1u, match ? "true" : "false");
    }
    if (add_capture)
        puts("{\"schema\":\"flash-sass.qsa-selected-blocks.v1\",\"source\":\"actual_model_gpu_readback\",\"selected_block_ids\":[0]}");
    puts("make: *** [validation/issue9/Makefile:201: model-oracle] Error 3");
    fflush(stdout);
    return wrong_exit ? 1 : 2;
}

static int executable_path(const char *argv0, char *buffer, size_t capacity) {
    ssize_t length = readlink("/proc/self/exe", buffer, capacity - 1u);
    if (length > 0 && (size_t)length < capacity) {
        buffer[length] = '\0';
        return 0;
    }
    if (!argv0 || strlen(argv0) >= capacity) return -1;
    strcpy(buffer, argv0);
    return 0;
}

static int fixture_grandchild_info(const child_output_t *output, pid_t *pid_out,
                                   pid_t *group_out) {
    static const char marker[] = "FIXTURE_GRANDCHILD_PID=";
    static const char group_marker[] = " PGID=";
    size_t at = find_bytes(output->bytes, output->length, marker, sizeof marker - 1u, 0);
    long value = 0, group = 0;
    if (at == SIZE_MAX) return -1;
    at += sizeof marker - 1u;
    if (at == output->length || !isdigit((unsigned char)output->bytes[at])) return -1;
    while (at < output->length && isdigit((unsigned char)output->bytes[at])) {
        unsigned digit = (unsigned)(output->bytes[at] - '0');
        if (value > ((long)INT_MAX - (long)digit) / 10L) return -1;
        value = value * 10L + digit;
        at++;
    }
    if (value <= 1) return -1;
    if (find_bytes(output->bytes + at, output->length - at, group_marker,
                   sizeof group_marker - 1u, 0) != 0) return -1;
    at += sizeof group_marker - 1u;
    if (at == output->length || !isdigit((unsigned char)output->bytes[at])) return -1;
    while (at < output->length && isdigit((unsigned char)output->bytes[at])) {
        unsigned digit = (unsigned)(output->bytes[at] - '0');
        if (group > ((long)INT_MAX - (long)digit) / 10L) return -1;
        group = group * 10L + digit;
        at++;
    }
    if (group <= 1) return -1;
    *pid_out = (pid_t)value;
    *group_out = (pid_t)group;
    return 0;
}

static int self_test_descendant_cleanup(const char *self, int nested_timeout) {
    char *direct_command[] = {(char *)self, "--fixture", "descendant-holds-pipe", NULL};
    char *nested_command[] = {
        (char *)timeout_argv[0], (char *)timeout_argv[1],
        (char *)timeout_argv[2], (char *)timeout_argv[3],
        (char *)self, "--fixture", "descendant-holds-pipe", NULL
    };
    char **command = nested_timeout ? nested_command : direct_command;
    child_output_t output;
    pid_t grandchild = 0, grandchild_group = 0;
    int run_ok = run_process(command, 1u, 0, &output) == 0;
    int live = -1;
    if (run_ok && fixture_grandchild_info(&output, &grandchild, &grandchild_group) == 0)
        live = process_is_live(grandchild);
    if (run_ok && output.pipe_eof && !output.group_live_after_cleanup &&
        !output.io_error && output.local_timeout && grandchild > 1 && live == 0 &&
        grandchild_group == output.process_group) {
        printf("ORACLE_FAULT_SELF_TEST_CASE {\"mode\":\"%s\",\"accepted\":true,"
               "\"pipe_eof\":true,\"descendant_live\":false,\"observed_pgid\":%ld,"
               "\"runner_pgid\":%ld,\"gpu_calls\":0}\n",
               nested_timeout ? "nested-timeout-leader-exit-stubborn-grandchild"
                              : "leader-exit-term-ignoring-grandchild",
               (long)grandchild_group, (long)output.process_group);
        free(output.bytes);
        return 0;
    }
    fprintf(stderr, "ORACLE_FAULT_SELF_TEST_CASE_FAIL mode=%s "
            "run_ok=%d pipe_eof=%d group_live=%d timeout=%d descendant_pid=%ld "
            "descendant_pgid=%ld runner_pgid=%ld descendant_live=%d io_error=%d\n",
            nested_timeout ? "nested-timeout-leader-exit-stubborn-grandchild"
                           : "leader-exit-term-ignoring-grandchild",
            run_ok, output.pipe_eof, output.group_live_after_cleanup,
            output.local_timeout, (long)grandchild, (long)grandchild_group,
            (long)output.process_group, live, output.io_error);
    free(output.bytes);
    return -1;
}

static int self_test(const char *argv0) {
    static const struct {
        const char *mode;
        int should_pass;
    } fixtures[] = {
        {"valid", 1},
        {"wrong-exit", 0},
        {"missing-marker", 0},
        {"missing-mismatch", 0},
        {"positive-capture", 0},
        {"trailing-token", 0},
        {"trailing-garbage", 0},
        {"duplicate-key", 0},
        {"extra-key", 0}
    };
    char self[ORACLE_FAULT_PATH_LIMIT];
    unsigned passed = 0;
    int failed = 0;
    if (executable_path(argv0, self, sizeof self) < 0) {
        fprintf(stderr, "oracle fault self-test: resolve fixture executable failed\n");
        return 1;
    }
    puts("ORACLE_FAULT_SELF_TEST_BEGIN {\"gpu_calls\":0}");
    fflush(stdout);
    if (self_test_make_argv() < 0) {
        fprintf(stderr, "ORACLE_FAULT_SELF_TEST_CASE_FAIL mode=make-argv\n");
        failed++;
    } else {
        puts("ORACLE_FAULT_SELF_TEST_CASE {\"mode\":\"make-argv\",\"accepted\":true,\"gpu_calls\":0}");
        passed++;
    }
    for (size_t i = 0; i < sizeof fixtures / sizeof fixtures[0]; i++) {
        char *command[] = {self, "--fixture", (char *)fixtures[i].mode, NULL};
        child_output_t output;
        validation_t validation;
        int run_ok = run_process(command, 10u, 0, &output) == 0;
        int accepted = 0;
        if (run_ok) {
            validate_output(&output, &fault_cases[0], &validation);
            accepted = validation_passed(&output, &fault_cases[0], &validation);
        }
        if (!run_ok || accepted != fixtures[i].should_pass) {
            fprintf(stderr, "ORACLE_FAULT_SELF_TEST_CASE_FAIL mode=%s expected=%d accepted=%d\n",
                    fixtures[i].mode, fixtures[i].should_pass, accepted);
            failed++;
        } else {
            printf("ORACLE_FAULT_SELF_TEST_CASE {\"mode\":\"%s\",\"accepted\":%s,\"expected\":%s}\n",
                   fixtures[i].mode, accepted ? "true" : "false",
                   fixtures[i].should_pass ? "true" : "false");
            passed++;
        }
        free(output.bytes);
    }
    if (self_test_descendant_cleanup(self, 0) < 0) failed++;
    else passed++;
    if (self_test_descendant_cleanup(self, 1) < 0) failed++;
    else passed++;
    printf("ORACLE_FAULT_SELF_TEST_RESULT {\"passed\":%u,\"failed\":%d,\"gpu_calls\":0}\n",
           passed, failed);
    return failed ? 1 : 0;
}

static int matrix_main(const char *makefile, const char *oracle_data,
                       const char *revision) {
    unsigned passed = 0, failed = 0, completed = 0;
    char data_arg[ORACLE_FAULT_PATH_LIMIT + 16u];
    char flags_arg[16];
    char revision_arg[ORACLE_FAULT_MAX_REVISION + 24u];
    char inject_arg[32];
    if (!valid_make_path(makefile) || !valid_make_path(oracle_data) ||
        !valid_revision(revision)) {
        fprintf(stderr, "usage values must be nonempty paths and a revision token of at most 63 characters\n");
        return 2;
    }
    if (strlen(oracle_data) + sizeof "ORACLE_DATA=" > sizeof data_arg ||
        strlen(revision) + sizeof "ORACLE_REVISION=" > sizeof revision_arg) {
        fprintf(stderr, "oracle data path or revision is too long\n");
        return 2;
    }
    if (setenv("LC_ALL", "C", 1) < 0) {
        fprintf(stderr, "set deterministic Make locale: %s\n", strerror(errno));
        return 1;
    }
    for (size_t i = 0; i < ORACLE_FAULT_CASE_COUNT; i++) {
        char *command[24];
        child_output_t output;
        validation_t validation;
        int accepted, run_ok;
        if (caught_signal) break;
        print_case_boundary(&fault_cases[i], "BEGIN");
        if (fflush(stdout) == EOF) return 1;
        if (make_command(makefile, oracle_data, revision, &fault_cases[i],
                         data_arg, sizeof data_arg, flags_arg, sizeof flags_arg,
                         revision_arg,
                         sizeof revision_arg, inject_arg, sizeof inject_arg, command,
                         sizeof command / sizeof command[0]) < 0) {
            fprintf(stderr, "construct model-oracle Make command failed for %s\n",
                    fault_cases[i].name);
            failed++;
            memset(&output, 0, sizeof output);
            output.exit_code = -1;
            output.io_error = 1;
            output.bytes = calloc(1, 1);
            memset(&validation, 0, sizeof validation);
            validation.false_position = UINT64_MAX;
            accepted = 0;
        } else {
            run_ok = run_process(command, 370u, 1, &output) == 0;
            if (run_ok) {
                validate_output(&output, &fault_cases[i], &validation);
                accepted = validation_passed(&output, &fault_cases[i], &validation);
            } else {
                memset(&validation, 0, sizeof validation);
                validation.false_position = UINT64_MAX;
                accepted = 0;
            }
            if (accepted) passed++;
            else failed++;
        }
        completed++;
        if (output.length && output.bytes && output.bytes[output.length - 1u] != '\n')
            (void)write_all(STDOUT_FILENO, "\n", 1u);
        print_case_result(&fault_cases[i], &output, &validation, accepted);
        print_case_boundary(&fault_cases[i], "END");
        free(output.bytes);
        if (caught_signal || output.interrupted || output.io_error || output.overflow ||
            output.local_timeout || output.term_signal || output.exit_code != 2 ||
            validation.make_error_count != 1u || validation.make_error_code != 3 ||
            validation.malformed_make_error) break;
    }
    printf("ORACLE_FAULT_MATRIX_RESULT {\"cases\":%u,\"expected_cases\":%u,"
           "\"passed\":%u,\"failed\":%u,\"revision\":\"%s\"}\n",
           completed, ORACLE_FAULT_CASE_COUNT, passed, failed, revision);
    fflush(stdout);
    return (failed || passed != ORACLE_FAULT_CASE_COUNT ||
            completed != ORACLE_FAULT_CASE_COUNT || caught_signal) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (install_signal_handlers() < 0) {
        fprintf(stderr, "oracle fault runner: install signal handlers: %s\n", strerror(errno));
        return 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--self-test"))
        return self_test(argv[0]);
    if (argc == 3 && !strcmp(argv[1], "--fixture"))
        return fixture_emit(argv[2]);
    if (argc != 4) {
        fprintf(stderr, "usage: %s MAKEFILE ORACLE_DATA REVISION\n", argv[0]);
        return 2;
    }
    return matrix_main(argv[1], argv[2], argv[3]);
}
