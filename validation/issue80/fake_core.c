#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int write_all(int fd, const char *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        ssize_t written = send(fd, data + offset, length - offset, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (!written) { errno = EIO; return -1; }
        offset += (size_t)written;
    }
    return 0;
}

static void handle(int fd, const char *mode, FILE *log) {
    FILE *client = fdopen(fd, "r+");
    if (!client) {
        perror("fdopen");
        _exit(1);
    }
    setvbuf(client, NULL, _IOLBF, 0);
    struct timespec wall_started;
    clock_gettime(CLOCK_REALTIME, &wall_started);
    int chat = 0;
    char line[8192];
    while (fgets(line, sizeof line, client)) {
        if (!strncmp(line, "info", 4)) {
            if (write_all(fd, "info 65536 4096\n", strlen("info 65536 4096\n")) < 0)
                _exit(1);
        }
        if (!strncmp(line, "chat ", 5)) {
            chat = 1;
            if (!strcmp(mode, "answer")) {
                const char *answer = "tok 9707\ndone 1 1 4 12.5 10.0 0 0\n";
                if (write_all(fd, answer, strlen(answer)) < 0) _exit(1);
            }
        }
    }
    struct timespec wall_finished;
    clock_gettime(CLOCK_REALTIME, &wall_finished);
    double elapsed = (double)(wall_finished.tv_sec - wall_started.tv_sec)
                   + (double)(wall_finished.tv_nsec - wall_started.tv_nsec) / 1e9;
    fprintf(log, "closed chat=%d after %.1f s\n", chat, elapsed);
    fflush(log);
    fclose(client);
    _exit(0);
}

int main(int argc, char **argv) {
    if (argc < 4 || !argv[3][0]
        || (strcmp(argv[2], "silent") && strcmp(argv[2], "answer"))) {
        fprintf(stderr, "usage: %s SOCKET silent|answer LOG\n", argv[0]);
        return 255;
    }
    const char *path = argv[1];
    const char *mode = argv[2];
    const char *log_path = argv[3];
    signal(SIGPIPE, SIG_IGN);
    unlink(path);

    int server = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server < 0) {
        fprintf(stderr, "listen %s: %s\n", path, strerror(errno));
        return 1;
    }
    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof address.sun_path) {
        fprintf(stderr, "listen %s: socket path too long\n", path);
        close(server);
        return 1;
    }
    strcpy(address.sun_path, path);
    if (bind(server, (struct sockaddr *)&address, sizeof address) < 0 || listen(server, 8) < 0) {
        fprintf(stderr, "listen %s: %s\n", path, strerror(errno));
        close(server);
        return 1;
    }
    FILE *log = fopen(log_path, "a");
    if (!log) {
        fprintf(stderr, "%s: %s\n", log_path, strerror(errno));
        close(server);
        unlink(path);
        return 1;
    }
    setvbuf(log, NULL, _IOLBF, 0);
    signal(SIGCHLD, SIG_IGN);
    for (;;) {
        int fd = accept(server, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "accept: %s\n", strerror(errno));
            break;
        }
        pid_t child = fork();
        if (child < 0) {
            fprintf(stderr, "fork: %s\n", strerror(errno));
            close(fd);
            break;
        }
        if (child == 0) {
            close(server);
            handle(fd, mode, log);
        }
        close(fd);
    }
    fclose(log);
    close(server);
    unlink(path);
    return 1;
}
