/* Parent-side encoder supervision. This file uses no CUDA/backend APIs. */
#define _GNU_SOURCE
#include "vis_supervisor.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#define REQUEST_MAX ((size_t)64 << 20)
#define RESPONSE_MAX ((size_t)168000000)
#define STARTUP_MS 120000
#define LINE_CAP 512

struct worker { pid_t pid; int fd; };
static volatile sig_atomic_t stopping;
static int signal_pipe[2] = {-1, -1};

static void stop_signal(int sig) {
    int saved = errno;
    stopping = sig;
    unsigned char byte = (unsigned char)sig;
    if (signal_pipe[1] >= 0) (void)write(signal_pipe[1], &byte, 1);
    errno = saved;
}

static int64_t milliseconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return -1;
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

/* A zero deadline means no timeout, including long GPU requests. */
static int wait_io(int fd, short events, int64_t deadline) {
    for (;;) {
        if (stopping) { errno = EINTR; return -1; }
        int timeout = -1;
        if (deadline) {
            int64_t now = milliseconds();
            if (now < 0) return -1;
            int64_t left = deadline - now;
            if (left <= 0) { errno = ETIMEDOUT; return -1; }
            timeout = left > INT_MAX ? INT_MAX : (int)left;
        }
        struct pollfd p[2] = {{.fd = fd, .events = events},
                             {.fd = signal_pipe[0], .events = POLLIN}};
        int rc = poll(p, 2, timeout);
        if (rc < 0) { if (errno == EINTR) continue; return -1; }
        if (!rc) { errno = ETIMEDOUT; return -1; }
        if (stopping || p[1].revents) { errno = EINTR; return -1; }
        if (p[0].revents & POLLNVAL) { errno = EBADF; return -1; }
        /* Read any buffered data before interpreting HUP as EOF. */
        if (p[0].revents & (events | POLLHUP | POLLERR)) return 0;
    }
}

/* read_line: 0 complete, 1 orderly EOF, -1 error/partial/oversized line. */
static int read_line(int fd, char *line, size_t capacity, int64_t deadline) {
    size_t n = 0;
    for (;;) {
        if (wait_io(fd, POLLIN, deadline)) return -1;
        char c;
        ssize_t rc = read(fd, &c, 1);
        if (rc < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return -1;
        }
        if (!rc) { errno = ECONNRESET; return n ? -1 : 1; }
        if (c == '\n') { line[n] = 0; return 0; }
        if (!c || n + 1 >= capacity) { errno = EMSGSIZE; return -1; }
        line[n++] = c;
    }
}

static int read_full(int fd, void *buffer, size_t count) {
    unsigned char *p = buffer;
    while (count) {
        if (wait_io(fd, POLLIN, 0)) return -1;
        ssize_t n = read(fd, p, count);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return -1;
        }
        if (!n) { errno = ECONNRESET; return -1; }
        p += n; count -= (size_t)n;
    }
    return 0;
}

static int write_full(int fd, const void *buffer, size_t count) {
    const unsigned char *p = buffer;
    while (count) {
        if (wait_io(fd, POLLOUT, 0)) return -1;
        ssize_t n = send(fd, p, count, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return -1;
        }
        if (!n) { errno = EPIPE; return -1; }
        p += n; count -= (size_t)n;
    }
    return 0;
}

static int send_line(int fd, const char *line) {
    return write_full(fd, line, strlen(line)) || write_full(fd, "\n", 1) ? -1 : 0;
}

/* Strict decimal parsing avoids scanf integer overflow and signed lengths. */
static int decimal(const char **text, uint64_t limit, uint64_t *value) {
    const char *p = *text;
    uint64_t n = 0;
    if (*p < '0' || *p > '9') return -1;
    do {
        unsigned d = (unsigned)(*p++ - '0');
        if (d > limit || n > (limit - d) / 10) return -1;
        n = n * 10 + d;
    } while (*p >= '0' && *p <= '9');
    *text = p; *value = n;
    return 0;
}

static int request_size(const char *line, size_t *bytes) {
    if (strncmp(line, "enc ", 4)) return -1;
    const char *p = line + 4;
    uint64_t n;
    if (decimal(&p, REQUEST_MAX, &n) || *p || !n) return -1;
    *bytes = (size_t)n;
    return 0;
}

static int response_size(const char *line, size_t *bytes) {
    if (strncmp(line, "ok ", 3)) return -1;
    const char *p = line + 3;
    uint64_t v[5];
    for (unsigned i = 0; i < 5; i++) {
        uint64_t limit = i == 0 ? 16384 : UINT32_MAX;
        if (decimal(&p, limit, &v[i]) || !v[i]) return -1;
        if (i < 4) { if (*p++ != ' ') return -1; }
        else if (*p) return -1;
    }
    if (v[4] != 2560) return -1;
    uint64_t n = v[0] * (4 * v[4] + 12);
    if (n > RESPONSE_MAX || n > SIZE_MAX) return -1;
    *bytes = (size_t)n;
    return 0;
}

int vis_supervisor_worker_fd(int *argc, char **argv) {
    errno = 0;
    if (!argc || !argv || *argc < 1) { errno = EINVAL; return -1; }
    if (*argc < 3 || strcmp(argv[*argc - 2], "--worker-fd")) return -1;
    const char *p = argv[*argc - 1];
    uint64_t number;
    if (decimal(&p, INT_MAX, &number) || *p || number < 3) { errno = EINVAL; return -1; }
    int fd = (int)number, flags = fcntl(fd, F_GETFD);
    int type = 0;
    socklen_t type_size = sizeof type, peer_size;
    struct sockaddr_un peer;
    peer_size = sizeof peer;
    if (flags < 0 || getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_size) ||
        type != SOCK_STREAM || getpeername(fd, (struct sockaddr *)&peer, &peer_size) ||
        peer.sun_family != AF_UNIX || fcntl(fd, F_SETFD, flags | FD_CLOEXEC)) {
        errno = EINVAL; return -1;
    }
    *argc -= 2;
    argv[*argc] = NULL;
    return fd;
}

/* Ensure the private channel cannot replace stdin/stdout/stderr. */
static int channel(int pair[2]) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair)) return -1;
    for (unsigned i = 0; i < 2; i++) if (pair[i] < 3) {
        int copy = fcntl(pair[i], F_DUPFD_CLOEXEC, 3);
        if (copy < 0) { close(pair[0]); close(pair[1]); return -1; }
        close(pair[i]); pair[i] = copy;
    }
    return 0;
}

static int wait_brief(pid_t pid, int *status, int milliseconds_limit) {
    int waited = 0;
    for (;;) {
        pid_t rc = waitpid(pid, status, WNOHANG);
        if (rc == pid) return 1;
        if (rc < 0) { if (errno == EINTR) continue; return errno == ECHILD ? 1 : -1; }
        if (waited >= milliseconds_limit) return 0;
        (void)poll(NULL, 0, 10); waited += 10;
    }
}

static void stop_worker(struct worker *w) {
    if (w->fd >= 0) close(w->fd);
    w->fd = -1;
    if (w->pid <= 0) return;
    int status = 0;
    pid_t rc = waitpid(w->pid, &status, WNOHANG);
    if (rc == 0 || (rc < 0 && errno == EINTR)) {
        (void)kill(w->pid, SIGTERM);
        if (wait_brief(w->pid, &status, 2000) == 0) {
            (void)kill(w->pid, SIGKILL);
            while (waitpid(w->pid, &status, 0) < 0 && errno == EINTR) {}
        }
    }
    w->pid = 0;
}

static int exit_code(int status) {
    if (WIFEXITED(status) && WEXITSTATUS(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
}

static int start_worker(struct worker *w, int argc, char **argv) {
    int pair[2];
    if (channel(pair)) { perror("flashvis: worker socketpair"); return 1; }
    char **args = calloc((size_t)argc + 3, sizeof *args);
    if (!args) { close(pair[0]); close(pair[1]); return 1; }
    for (int i = 0; i < argc; i++) args[i] = argv[i];
    char fd_text[32];
    snprintf(fd_text, sizeof fd_text, "%d", pair[1]);
    args[argc] = "--worker-fd"; args[argc + 1] = fd_text;
    struct sigaction normal = {.sa_handler = SIG_DFL};
    sigemptyset(&normal.sa_mask);
    pid_t parent = getpid();
    pid_t pid = fork();
    if (pid == 0) {
        close(pair[0]);
        (void)sigaction(SIGTERM, &normal, NULL);
        (void)sigaction(SIGINT, &normal, NULL);
        (void)sigaction(SIGHUP, &normal, NULL);
#ifdef __linux__
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) || getppid() != parent) _exit(1);
#else
        (void)parent;
#endif
        if (fcntl(pair[1], F_SETFD, 0)) _exit(1);
        execv("/proc/self/exe", args);
        _exit(127);
    }
    close(pair[1]); free(args);
    if (pid < 0) { close(pair[0]); perror("flashvis: fork worker"); return 1; }
    w->pid = pid; w->fd = pair[0];
    int flags = fcntl(w->fd, F_GETFL);
    if (flags < 0 || fcntl(w->fd, F_SETFL, flags | O_NONBLOCK)) {
        stop_worker(w); return 1;
    }
    int64_t now = milliseconds();
    char line[LINE_CAP];
    if (now >= 0 && read_line(w->fd, line, sizeof line, now + STARTUP_MS) == 0 &&
        !strcmp(line, "ready")) return 0;
    int saved = errno, status = 0;
    if (saved == ETIMEDOUT) fprintf(stderr, "flashvis: GPU worker startup timed out\n");
    if (!stopping && wait_brief(w->pid, &status, 1000) == 1) {
        close(w->fd); w->fd = -1; w->pid = 0;
        return exit_code(status);
    }
    stop_worker(w);
    return stopping ? 128 + stopping : 1;
}

/* Idle worker death has no current image. Reinitialize before forwarding a
 * new request. A worker lost after forwarding is never retried here. */
static int worker_ready(struct worker *w, int argc, char **argv) {
    if (w->pid > 0) {
        int status;
        pid_t rc = waitpid(w->pid, &status, WNOHANG);
        if (rc == 0) {
            /* An exited worker can still await reaping; a worker that
             * closed the protocol fd can still be unwinding cleanup. */
            char byte;
            ssize_t n = recv(w->fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
            if (n < 0 && errno == EINTR) return -1;
            stop_worker(w);
            return start_worker(w, argc, argv);
        }
        if (rc < 0 && errno == EINTR) return -1;
        if (w->fd >= 0) close(w->fd);
        w->fd = -1; w->pid = 0;
    }
    return start_worker(w, argc, argv);
}

/* Returns -1 when this client must close; worker failure otherwise leaves
 * the client connected so its next image may use a fresh GPU process. */
static int serve_request(int client, struct worker *w, int argc, char **argv) {
    char line[LINE_CAP];
    int rc = read_line(client, line, sizeof line, 0);
    if (rc) {
        if (rc < 0 && errno == EMSGSIZE) (void)send_line(client, "err bad request");
        return -1;
    }
    size_t request_bytes;
    if (request_size(line, &request_bytes)) {
        (void)send_line(client, "err bad request"); return -1;
    }
    unsigned char *request = malloc(request_bytes);
    if (!request) {
        (void)send_line(client, "err cannot allocate image request"); return -1;
    }
    if (read_full(client, request, request_bytes)) { free(request); return -1; }
    if (worker_ready(w, argc, argv)) {
        free(request);
        return send_line(client, "err GPU encoder worker terminated");
    }
    char header[LINE_CAP];
    int length = snprintf(header, sizeof header, "enc %zu\n", request_bytes);
    int failed = write_full(w->fd, header, (size_t)length) ||
                 write_full(w->fd, request, request_bytes);
    free(request);
    if (failed || read_line(w->fd, header, sizeof header, 0)) goto terminated;
    if (!strncmp(header, "err ", 4) && header[4]) {
        /* Retire even an in-band failure: the worker may be unwinding a
         * poisoned CUDA context after writing its error. The next request
         * gets a fresh GPU process; this image is not resubmitted. */
        stop_worker(w);
        return send_line(client, header);
    }
    size_t response_bytes;
    if (response_size(header, &response_bytes)) goto terminated;
    unsigned char *response = malloc(response_bytes);
    if (!response) {
        stop_worker(w);
        return send_line(client, "err cannot allocate GPU encoder response");
    }
    if (read_full(w->fd, response, response_bytes)) { free(response); goto terminated; }
    /* No successful byte reaches the external client until the worker has
     * delivered the complete bounded response, including all positions. */
    rc = send_line(client, header) || write_full(client, response, response_bytes);
    free(response);
    return rc ? -1 : 0;
terminated:
    stop_worker(w);
    return send_line(client, "err GPU encoder worker terminated");
}

static int bind_listener(const char *path, struct stat *identity) {
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    memcpy(address.sun_path, path, strlen(path) + 1);
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;
    if (bind(fd, (struct sockaddr *)&address, sizeof address)) {
        int saved = errno;
        struct stat old;
        /* Recover a stale socket, preserving ordinary files and active
         * listeners. Do not unlink a different inode installed meanwhile. */
        if (saved == EADDRINUSE && !lstat(path, &old) && S_ISSOCK(old.st_mode)) {
            int probe = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
            int refused = 0;
            if (probe >= 0) {
                if (connect(probe, (struct sockaddr *)&address, sizeof address) < 0 &&
                    errno == ECONNREFUSED) refused = 1;
                close(probe);
            }
            struct stat current;
            if (refused && !lstat(path, &current) && old.st_dev == current.st_dev &&
                old.st_ino == current.st_ino && !unlink(path) &&
                !bind(fd, (struct sockaddr *)&address, sizeof address)) saved = 0;
        }
        if (saved) { close(fd); errno = saved; return -1; }
    }
    if (lstat(path, identity) || listen(fd, 4)) {
        int saved = errno; close(fd); unlink(path); errno = saved; return -1;
    }
    return fd;
}

int vis_supervisor_main(int argc, char **argv) {
    if (argc < 4 || !argv || !argv[3] || !*argv[3] ||
        strlen(argv[3]) >= sizeof ((struct sockaddr_un *)0)->sun_path) {
        fprintf(stderr, "flashvis: invalid arguments or Unix socket path\n"); return 2;
    }
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--worker-fd")) {
        fprintf(stderr, "flashvis: invalid private worker arguments\n"); return 2;
    }
    stopping = 0;
    if (pipe2(signal_pipe, O_CLOEXEC | O_NONBLOCK)) {
        perror("flashvis: signal pipe"); return 1;
    }
    const int signals[] = {SIGTERM, SIGINT, SIGHUP, SIGPIPE};
    struct sigaction saved[4], action = {.sa_handler = stop_signal};
    sigemptyset(&action.sa_mask);
    unsigned installed = 0;
    for (; installed < 4; installed++) {
        if (installed == 3) action.sa_handler = SIG_IGN;
        if (sigaction(signals[installed], &action, &saved[installed])) break;
    }
    struct worker w = {.pid = 0, .fd = -1};
    int result = 1, listener = -1, client = -1, bound = 0;
    struct stat identity;
    if (installed != 4) { perror("flashvis: sigaction"); goto done; }
    result = start_worker(&w, argc, argv);
    if (result || stopping) goto done;
    listener = bind_listener(argv[3], &identity);
    if (listener < 0) { perror(argv[3]); result = 1; goto done; }
    bound = 1;
    for (;;) {
        if (wait_io(listener, POLLIN, 0)) break;
        client = accept4(listener, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (client < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            perror("flashvis: accept"); break;
        }
        while (!stopping && serve_request(client, &w, argc, argv) == 0) {}
        close(client); client = -1;
        if (stopping) break;
    }
    result = stopping ? 128 + stopping : 1;
done:
    if (client >= 0) close(client);
    if (listener >= 0) close(listener);
    stop_worker(&w);
    if (bound) {
        struct stat current;
        if (!lstat(argv[3], &current) && current.st_dev == identity.st_dev &&
            current.st_ino == identity.st_ino) (void)unlink(argv[3]);
    }
    while (installed) { installed--; (void)sigaction(signals[installed], &saved[installed], NULL); }
    close(signal_pipe[0]); close(signal_pipe[1]);
    signal_pipe[0] = signal_pipe[1] = -1;
    return stopping ? 128 + stopping : result;
}
