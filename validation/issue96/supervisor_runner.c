/* CPU-only supervisor IPC regression. No shell, Ruby, GPU, or compiler
 * subprocess is used by this runner. Pass the separately built fixture.
 *
 * Suggested kern/Makefile targets (invoke Make through remote dsh build):
 * supervisor_fixture_test: ../validation/issue96/supervisor_test.c vis_supervisor.c vis_supervisor.h
 *     $(CC) -std=c11 -O2 -Wall -Wextra -Werror -I. $< vis_supervisor.c -o $@
 * supervisor_runner_test: ../validation/issue96/supervisor_runner.c
 *     $(CC) -std=c11 -O2 -Wall -Wextra -Werror $< -o $@
 * .PHONY: test-vis-supervisor
 * test-vis-supervisor: supervisor_fixture_test supervisor_runner_test
 *     ./supervisor_runner_test ./supervisor_fixture_test
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
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

_Static_assert(sizeof(float) == 4, "fixture protocol requires four-byte floats");
enum { TIMEOUT_MS = 5000, BODY_BYTES = 2560 * 4 + 12 };
static volatile sig_atomic_t interrupted;
struct process { pid_t pid; int reaped; };

static void interrupt_handler(int sig) { interrupted = sig; }

static int64_t now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static int fail(const char *message) {
    fprintf(stderr, "supervisor_runner: %s\n", message);
    return -1;
}

static int pause_ms(int ms) {
    struct timespec t = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L};
    while (nanosleep(&t, &t) && errno == EINTR) if (interrupted) return -1;
    return interrupted ? -1 : 0;
}

static int wait_fd(int fd, short events, int64_t deadline) {
    while (!interrupted) {
        int64_t left = deadline - now_ms();
        if (left <= 0) return fail("IPC timed out");
        /* Bounded poll slices also cover a signal delivered before poll. */
        struct pollfd p = {.fd = fd, .events = events};
        int rc = poll(&p, 1, left > 100 ? 100 : (int)left);
        if (rc < 0) { if (errno == EINTR) continue; perror("poll"); return -1; }
        if (p.revents & POLLNVAL) return fail("invalid IPC descriptor");
        if (rc > 0 && (p.revents & (events | POLLHUP | POLLERR))) return 0;
    }
    return fail("interrupted");
}

static int transfer(int fd, void *buffer, size_t size, int writing, int64_t deadline) {
    unsigned char *p = buffer;
    while (size) {
        if (wait_fd(fd, writing ? POLLOUT : POLLIN, deadline)) return -1;
        ssize_t n = writing ? send(fd, p, size, MSG_NOSIGNAL) : read(fd, p, size);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            perror(writing ? "send" : "read"); return -1;
        }
        if (!n) return fail("closed or partial public response");
        p += n; size -= (size_t)n;
    }
    return 0;
}

static int response_line(int fd, char *line, size_t capacity, int64_t deadline) {
    size_t n = 0;
    for (;;) {
        char c;
        if (transfer(fd, &c, 1, 0, deadline)) return -1;
        if (!c || n + 1 >= capacity) return fail("invalid public response line");
        line[n++] = c;
        if (c == '\n') { line[n] = 0; return 0; }
    }
}

static int spawn_fixture(struct process *p, const char *binary, const char *socket_path,
                         const char *log_path, const char *device) {
    int log = open(log_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (log < 0) { perror(log_path); return -1; }
    pid_t pid = fork();
    if (!pid) {
        if (setpgid(0, 0) || dup2(log, STDERR_FILENO) < 0 ||
            fcntl(STDERR_FILENO, F_SETFD, 0) ||
            setenv("FLASHVIS_DEVICE", device, 1)) _exit(126);
        if (log != STDERR_FILENO) close(log);
        char *const args[] = {(char *)binary, "projector-placeholder",
                              "text-placeholder", (char *)socket_path, NULL};
        execv(binary, args);
        _exit(127);
    }
    close(log);
    if (pid < 0) { perror("fork"); return -1; }
    p->pid = pid; p->reaped = 0;
    if (setpgid(pid, pid) && errno != EACCES && errno != ESRCH) {
        perror("setpgid"); return -1;
    }
    return 0;
}

/* Return 0 on exit, 1 while alive, -1 on error. Retain pgid after reaping
 * so cleanup can still terminate descendants if a broken supervisor exits. */
static int poll_exit(struct process *p, int *status) {
    pid_t rc;
    do { rc = waitpid(p->pid, status, WNOHANG); } while (rc < 0 && errno == EINTR);
    if (rc == p->pid) { p->reaped = 1; return 0; }
    if (rc == 0) return 1;
    perror("waitpid"); return -1;
}

static int wait_exit(struct process *p, int *status, int64_t deadline) {
    while (!interrupted) {
        int rc = poll_exit(p, status);
        if (rc <= 0) return rc;
        if (now_ms() >= deadline) return fail("supervisor did not exit before deadline");
        if (pause_ms(10)) break;
    }
    return fail("interrupted while waiting for supervisor");
}

static int stop_process(struct process *p, int require_graceful) {
    if (p->pid <= 0) return 0;
    int failed = 0, status = 0;
    (void)kill(-p->pid, SIGTERM);
    int64_t deadline = now_ms() + TIMEOUT_MS;
    while (!p->reaped && now_ms() < deadline) {
        if (poll_exit(p, &status) < 0) { failed = 1; break; }
        if (!p->reaped) (void)poll(NULL, 0, 10);
    }
    if (!p->reaped) {
        failed = 1;
        (void)kill(-p->pid, SIGKILL);
        (void)kill(p->pid, SIGKILL);
        while (waitpid(p->pid, &status, 0) < 0 && errno == EINTR) {}
        p->reaped = 1;
    }
    /* A normal group TERM must leave neither supervisor nor worker alive. */
    while (kill(-p->pid, 0) == 0 && now_ms() < deadline) (void)poll(NULL, 0, 10);
    if (kill(-p->pid, 0) == 0 || errno != ESRCH) {
        failed = 1;
        (void)kill(-p->pid, SIGKILL);
    }
    p->pid = 0;
    return require_graceful && failed ? fail("group TERM did not clean up the fixture") : 0;
}

static int wait_socket(struct process *p, const char *path) {
    int64_t deadline = now_ms() + TIMEOUT_MS;
    while (!interrupted) {
        int status;
        if (poll_exit(p, &status) != 1) return fail("supervisor exited before listening");
        struct stat st;
        if (!lstat(path, &st) && S_ISSOCK(st.st_mode)) return 0;
        if (now_ms() >= deadline) return fail("supervisor did not create the public socket");
        if (pause_ms(10)) break;
    }
    return fail("interrupted while waiting for socket");
}

static int connect_socket(const char *path, int64_t deadline) {
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    size_t length = strlen(path);
    if (length >= sizeof address.sun_path) return fail("test socket path too long");
    memcpy(address.sun_path, path, length + 1);
    while (!interrupted && now_ms() < deadline) {
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (fd < 0) { perror("socket"); return -1; }
        if (!connect(fd, (struct sockaddr *)&address, sizeof address)) return fd;
        int saved = errno;
        if (saved == EINPROGRESS && !wait_fd(fd, POLLOUT, deadline)) {
            socklen_t size = sizeof saved;
            if (!getsockopt(fd, SOL_SOCKET, SO_ERROR, &saved, &size) && !saved) return fd;
        }
        close(fd);
        if (saved != ECONNREFUSED && saved != ENOENT && saved != EAGAIN && saved != EINTR) {
            errno = saved; perror("connect"); return -1;
        }
        if (pause_ms(10)) break;
    }
    return fail("could not connect to supervisor");
}

/* Each request uses a new external connection, matching the former runner.
 * Exact expected error matching rejects any prematurely published ok header. */
static int request(const char *path, unsigned char tag, const char *expected_error, pid_t *worker_pid) {
    int64_t deadline = now_ms() + TIMEOUT_MS;
    int fd = connect_socket(path, deadline);
    if (fd < 0) return -1;
    int rc = -1;
    char header[256], input[] = "enc 1\n";
    if (transfer(fd, input, sizeof input - 1, 1, deadline) ||
        transfer(fd, &tag, 1, 1, deadline) || response_line(fd, header, sizeof header, deadline)) goto done;
    if (expected_error) {
        if (strcmp(header, expected_error)) {
            fprintf(stderr, "supervisor_runner: tag %c expected %sbut received %s", tag, expected_error, header);
            goto done;
        }
        rc = 0; goto done;
    }
    if (strcmp(header, "ok 1 1 1 1 2560\n")) {
        fprintf(stderr, "supervisor_runner: tag %c unexpected success header: %s", tag, header);
        goto done;
    }
    unsigned char body[BODY_BYTES];
    if (transfer(fd, body, sizeof body, 0, deadline)) goto done;
    float identity;
    memcpy(&identity, body, sizeof identity);
    if (!isfinite(identity) || (double)identity < 1 || (double)identity > INT_MAX) {
        fail("invalid fixture worker identity"); goto done;
    }
    *worker_pid = (pid_t)identity;
    if (identity != (float)*worker_pid) { fail("nonintegral fixture worker identity"); goto done; }
    for (size_t i = sizeof identity; i < sizeof body; i++) if (body[i]) {
        fail("fixture embeddings/positions changed"); goto done;
    }
    rc = 0;
done:
    close(fd);
    return rc;
}

static int check_dispatches(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    const char tags[] = "APIE";
    unsigned counts[4] = {0};
    char line[512];
    while (fgets(line, sizeof line, f)) {
        long pid;
        char tag, tail;
        if (sscanf(line, "fixture-request pid=%ld tag=%c%c", &pid, &tag, &tail) == 3 &&
            pid > 0 && tail == '\n') {
            for (unsigned i = 0; i < 4; i++) if (tag == tags[i]) counts[i]++;
        }
    }
    int failed = ferror(f);
    fclose(f);
    for (unsigned i = 0; i < 4; i++) if (counts[i] != 1) {
        fprintf(stderr, "supervisor_runner: tag %c dispatched %u times, expected exactly once\n", tags[i], counts[i]);
        failed = 1;
    }
    return failed ? -1 : 0;
}

static int pid_gone(pid_t pid, int64_t deadline) {
    while (!interrupted && now_ms() < deadline) {
        if (kill(pid, 0) < 0 && errno == ESRCH) return 0;
        (void)poll(NULL, 0, 10);
    }
    return fail("mock supervisor or worker survived client cancellation");
}

static int group_alive(pid_t group) {
    if (group <= 1) return 0;
    if (kill(-group, 0) == 0) return 1;
    return errno != ESRCH;
}

static int pid_alive(pid_t pid) {
    if (pid <= 1) return 0;
    if (kill(pid, 0) == 0) return 1;
    return errno != ESRCH;
}

static void stop_fixture_group(pid_t group, pid_t supervisor, pid_t worker) {
    if (group <= 1) return;
    (void)kill(-group, SIGTERM);
    int64_t deadline = now_ms() + TIMEOUT_MS;
    while (now_ms() < deadline &&
           (group_alive(group) || pid_alive(supervisor) || pid_alive(worker)))
        (void)poll(NULL, 0, 10);
    if (group_alive(group) || pid_alive(supervisor) || pid_alive(worker)) {
        (void)kill(-group, SIGKILL);
        if (supervisor > 1) (void)kill(supervisor, SIGKILL);
        if (worker > 1) (void)kill(worker, SIGKILL);
        deadline = now_ms() + TIMEOUT_MS;
        while (now_ms() < deadline &&
               (group_alive(group) || pid_alive(supervisor) || pid_alive(worker)))
            (void)poll(NULL, 0, 10);
    }
}

static int cancellation_case(const char *client_binary, const char *fixture,
                             const char *dir, int sig, const char *label) {
    char marker[PATH_MAX], log[PATH_MAX], socket_path[128], output[PATH_MAX];
    int nmarker = snprintf(marker, sizeof marker, "%s/%s-worker.pid", dir, label);
    int nlog = snprintf(log, sizeof log, "%s/%s-client.log", dir, label);
    int noutput = snprintf(output, sizeof output, "%s/embeddings-0.bin", dir);
    if (nmarker < 0 || (size_t)nmarker >= sizeof marker || nlog < 0 ||
        (size_t)nlog >= sizeof log || noutput < 0 || (size_t)noutput >= sizeof output)
        return fail("cancellation test path too long");
    struct process client = {0};
    int log_fd = open(log, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (log_fd < 0) { perror(log); return -1; }
    pid_t pid = fork();
    if (!pid) {
        if (setpgid(0, 0) || dup2(log_fd, STDERR_FILENO) < 0 ||
            fcntl(STDERR_FILENO, F_SETFD, 0) ||
            setenv("GPUQ_RUN_DIR", dir, 1) ||
            setenv("GPUQ_JOB_ID", "supervisor-startup-cancel", 1) ||
            setenv("FLASHVIS_DEVICE", "auto", 1) ||
            setenv("VIS_SUPERVISOR_TEST_STARTUP_MARKER", marker, 1) ||
            setenv("VIS_SUPERVISOR_TEST_STARTUP_DELAY_MS", "30000", 1)) _exit(126);
        if (log_fd != STDERR_FILENO) close(log_fd);
        char *const args[] = {(char *)client_binary, "encode", (char *)fixture,
                              "projector-placeholder", "text-placeholder", "1x1", NULL};
        execv(client_binary, args);
        _exit(127);
    }
    close(log_fd);
    if (pid < 0) { perror("fork client"); return -1; }
    client.pid = pid;
    int result = -1, status = 0;
    long worker_pid = 0, supervisor_pid = 0;
    FILE *f = NULL;
    if (setpgid(pid, pid) && errno != EACCES && errno != ESRCH) {
        perror("setpgid client"); goto done;
    }

    const int64_t deadline = now_ms() + TIMEOUT_MS;
    while (!interrupted && now_ms() < deadline) {
        f = fopen(marker, "r");
        if (f) break;
        if (poll_exit(&client, &status) != 1) {
            fail("vision_check exited before the delayed worker marker");
            goto done;
        }
        (void)poll(NULL, 0, 10);
    }
    if (!f) { fail("delayed worker marker timed out"); goto done; }
    int parsed = fscanf(f, "%ld %ld", &worker_pid, &supervisor_pid);
    fclose(f);
    f = NULL;
    if (parsed != 2 || worker_pid <= 1 || supervisor_pid <= 1 ||
        worker_pid > INT_MAX || supervisor_pid > INT_MAX) {
        fail("invalid worker/supervisor marker"); goto done;
    }
    int nsocket = snprintf(socket_path, sizeof socket_path,
                           "/tmp/vision-check-%ld.sock", (long)pid);
    if (nsocket < 0 || (size_t)nsocket >= sizeof socket_path) {
        fail("client socket path too long"); goto done;
    }
    struct stat socket_stat;
    if (!lstat(socket_path, &socket_stat)) {
        fail("public socket appeared before worker ready"); goto done;
    }

    if (kill(pid, sig)) { perror("signal vision_check"); goto done; }
    int exit_result = wait_exit(&client, &status, now_ms() + TIMEOUT_MS);
    if (exit_result) goto done;
    if ((WIFEXITED(status) && WEXITSTATUS(status) == 0) ||
        (!WIFEXITED(status) && (!WIFSIGNALED(status) || WTERMSIG(status) != sig))) {
        fail("vision_check did not report the requested cancellation"); goto done;
    }
    if (stop_process(&client, 1)) goto done;
    if (!lstat(socket_path, &socket_stat)) {
        fail("cancelled startup left its public socket"); goto done;
    }
    if (pid_gone((pid_t)worker_pid, now_ms() + TIMEOUT_MS) ||
        pid_gone((pid_t)supervisor_pid, now_ms() + TIMEOUT_MS)) goto done;
    if (!access(output, F_OK)) { fail("cancelled startup wrote embeddings"); goto done; }
    printf("{\"event\":\"vision_startup_cancel_passed\",\"signal\":\"%s\","
           "\"client_pid\":%ld,\"supervisor_pid\":%ld,\"worker_pid\":%ld}\n",
           label, (long)pid, supervisor_pid, worker_pid);
    result = 0;
done:
    if (f) fclose(f);
    if (client.pid > 0) (void)stop_process(&client, 0);
    stop_fixture_group((pid_t)supervisor_pid, (pid_t)supervisor_pid,
                       (pid_t)worker_pid);
    (void)unlink(marker);
    (void)unlink(log);
    (void)unlink(output);
    if (client.pid > 0) client.pid = 0;
    return result;
}

static void dump_log(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    fprintf(stderr, "--- fixture diagnostics: %s ---\n", path);
    char block[4096];
    size_t total = 0, n;
    while (total < 65536 && (n = fread(block, 1, sizeof block, f))) {
        (void)fwrite(block, 1, n, stderr); total += n;
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: supervisor_runner FIXTURE_BINARY [VISION_CHECK_BINARY]\n");
        return 2;
    }
    char binary[PATH_MAX];
    if (!realpath(argv[1], binary)) { perror(argv[1]); return 2; }
    char client_binary[PATH_MAX];
    if (argc == 3 && !realpath(argv[2], client_binary)) { perror(argv[2]); return 2; }
    struct sigaction action = {.sa_handler = interrupt_handler};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL) ||
        sigaction(SIGHUP, &action, NULL)) { perror("sigaction"); return 2; }
    char dir[] = "/tmp/fs96-supervisor-XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 2; }
    char startup_socket[128], startup_log[128], socket_path[128], log[128];
    snprintf(startup_socket, sizeof startup_socket, "%s/startup.sock", dir);
    snprintf(startup_log, sizeof startup_log, "%s/startup.log", dir);
    snprintf(socket_path, sizeof socket_path, "%s/supervisor.sock", dir);
    snprintf(log, sizeof log, "%s/worker.log", dir);
    struct process startup = {0}, server = {0};
    int result = 1, status;
    if (spawn_fixture(&startup, binary, startup_socket, startup_log, "cpu") ||
        wait_exit(&startup, &status, now_ms() + TIMEOUT_MS)) goto done;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 2) {
        fail("GPU-off startup must exit exactly 2"); goto done;
    }
    struct stat st;
    if (!lstat(startup_socket, &st) || errno != ENOENT) {
        fail("failed startup published an external socket"); goto done;
    }
    if (stop_process(&startup, 1)) goto done;
    if (spawn_fixture(&server, binary, socket_path, log, "auto") || wait_socket(&server, socket_path)) goto done;
    pid_t previous, current;
    if (request(socket_path, 'V', NULL, &previous) || request(socket_path, 'V', NULL, &current)) goto done;
    if (current != previous) { fail("healthy requests did not reuse the persistent worker"); goto done; }
    const struct { char tag; const char *name; const char *error; } cases[] = {
        {'A', "abort", "err GPU encoder worker terminated\n"},
        {'P', "partial_body", "err GPU encoder worker terminated\n"},
        {'I', "invalid_header", "err GPU encoder worker terminated\n"},
        {'E', "reported_error", "err fixture reported encode failure\n"}
    };
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (request(socket_path, (unsigned char)cases[i].tag, cases[i].error, NULL) ||
            request(socket_path, 'V', NULL, &current)) goto done;
        if (current == previous) { fail("failed worker was reused for the next image"); goto done; }
        previous = current;
        printf("{\"event\":\"supervisor_case_passed\",\"case\":\"%s\"}\n", cases[i].name);
    }
    if (stop_process(&server, 1) || check_dispatches(log)) goto done;
    if (argc == 3) {
        if (cancellation_case(client_binary, binary, dir, SIGINT, "startup-int") ||
            cancellation_case(client_binary, binary, dir, SIGTERM, "startup-term") ||
            cancellation_case(client_binary, binary, dir, SIGHUP, "startup-hup")) goto done;
    }
    puts("CPU-only supervisor IPC regression passed: startup exit 2, persistent worker, abort, partial body, invalid header, exact errors, fresh worker, no replay, group TERM cleanup.");
    result = 0;
done:
    (void)stop_process(&startup, 0);
    (void)stop_process(&server, 0);
    if (result) { dump_log(startup_log); dump_log(log); }
    (void)unlink(startup_socket); (void)unlink(socket_path);
    (void)unlink(startup_log); (void)unlink(log); (void)rmdir(dir);
    return interrupted ? 128 + interrupted : result;
}
