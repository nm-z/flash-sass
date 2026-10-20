#define _GNU_SOURCE
#include "vis_supervisor.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

static int read_full(int fd, void *buffer, size_t size) {
    unsigned char *p = buffer;
    while (size) {
        ssize_t n = read(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n;
        size -= (size_t)n;
    }
    return 0;
}

static int write_full(int fd, const void *buffer, size_t size) {
    const unsigned char *p = buffer;
    while (size) {
        ssize_t n = write(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n;
        size -= (size_t)n;
    }
    return 0;
}

static int read_line(int fd, char *line, size_t capacity) {
    size_t length = 0;
    for (;;) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        if (c == '\n') break;
        if (length + 1 >= capacity) return -1;
        line[length++] = c;
    }
    line[length] = '\0';
    return 0;
}

static int worker(int fd) {
    struct rlimit no_core = {0, 0};
    (void)setrlimit(RLIMIT_CORE, &no_core);
    if (write_full(fd, "ready\n", 6)) return 2;

    for (;;) {
        char line[128];
        size_t nbytes = 0;
        char extra = 0;
        if (read_line(fd, line, sizeof line)) return 0;
        if (sscanf(line, "enc %zu%c", &nbytes, &extra) != 1 || nbytes != 1) {
            static const char bad[] = "err fixture rejected request\n";
            if (write_full(fd, bad, sizeof bad - 1)) return 2;
            continue;
        }

        unsigned char payload;
        if (read_full(fd, &payload, sizeof payload)) return 0;
        fprintf(stderr, "fixture-request pid=%ld tag=%c\n", (long)getpid(), payload);
        fflush(stderr);

        if (payload == 'A') abort();
        if (payload == 'P') {
            static const char header[] = "ok 1 1 1 1 2560\n";
            const unsigned char partial_body[7] = {0};
            (void)write_full(fd, header, sizeof header - 1);
            (void)write_full(fd, partial_body, sizeof partial_body);
            shutdown(fd, SHUT_RDWR);
            close(fd);
            return 0;
        }
        if (payload == 'I') {
            static const char invalid[] = "invalid worker response\n";
            (void)write_full(fd, invalid, sizeof invalid - 1);
            shutdown(fd, SHUT_RDWR);
            close(fd);
            return 0;
        }
        if (payload == 'E') {
            /* Keep the mock alive after its in-band failure. The supervisor
             * must retire it before forwarding the error, or the next valid
             * request will expose the same worker PID. */
            static const char reported[] = "err fixture reported encode failure\n";
            if (write_full(fd, reported, sizeof reported - 1)) return 2;
            continue;
        }
        if (payload != 'V') {
            static const char bad[] = "err unknown fixture payload\n";
            if (write_full(fd, bad, sizeof bad - 1)) return 2;
            continue;
        }

        static const char header[] = "ok 1 1 1 1 2560\n";
        float embeddings[2560] = {0};
        uint32_t positions[3] = {0, 0, 0};
        embeddings[0] = (float)getpid();
        if (write_full(fd, header, sizeof header - 1) ||
            write_full(fd, embeddings, sizeof embeddings) ||
            write_full(fd, positions, sizeof positions)) return 2;
    }
}

int main(int argc, char **argv) {
    int fd = vis_supervisor_worker_fd(&argc, argv);
    if (fd < 0) return vis_supervisor_main(argc, argv);

    /* Model flashvis rejecting a CPU-only/no-GPU startup in its worker. */
    const char *device = getenv("FLASHVIS_DEVICE");
    if (device && !strcmp(device, "cpu")) return 2;

    return worker(fd);
}
