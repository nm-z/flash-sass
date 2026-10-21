#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int write_all(int fd, const char *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        ssize_t written = send(fd, data + offset, length - offset, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (!written) return -1;
        offset += (size_t)written;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SOCKET\n", argv[0]);
        return 1;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;
    if (strlen(argv[1]) >= sizeof address.sun_path) {
        fprintf(stderr, "socket path too long\n");
        close(fd);
        return 1;
    }
    strcpy(address.sun_path, argv[1]);
    if (connect(fd, (struct sockaddr *)&address, sizeof address) < 0) {
        perror("connect");
        close(fd);
        return 1;
    }
    const char *commands =
        "info\n"
        "timings 1\n"
        "cancel_on_eof 1\n"
        "stop 0 0\n"
        "sample 1 0 32 1\n"
        "chat 2 1 2\n";
    if (write_all(fd, commands, strlen(commands)) < 0 || shutdown(fd, SHUT_WR) < 0) {
        perror("write");
        close(fd);
        return 1;
    }
    char bytes[4096];
    for (;;) {
        ssize_t got = read(fd, bytes, sizeof bytes);
        if (got < 0) {
            if (errno == EINTR) continue;
            perror("read");
            close(fd);
            return 1;
        }
        if (!got) break;
        if (fwrite(bytes, 1, (size_t)got, stdout) != (size_t)got) {
            perror("stdout");
            close(fd);
            return 1;
        }
    }
    close(fd);
    return 0;
}
