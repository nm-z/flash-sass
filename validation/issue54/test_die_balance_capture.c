#define _GNU_SOURCE
#include "fixtures.h"
#include "../../kern/die_balance_capture.h"

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int listener;
    int mode;
    char commands[32][128];
    size_t command_count;
} fake_core;

typedef struct {
    const char *path;
    _Atomic int acquired;
} lock_worker;

static int checks;
static int failures;
static char directory[256];

static void *lock_thread_main(void *argument) {
    lock_worker *worker = argument;
    int fd = -1; char *error = NULL;
    if (dc_validation_lock(worker->path, NULL, NULL, &fd, &error))
        atomic_store(&worker->acquired, 1);
    free(error);
    if (fd >= 0) close(fd);
    return NULL;
}

static void expect(int condition, const char *name) {
    checks++;
    if (!condition) {
        failures++;
        fprintf(stderr, "FAIL: %s\n", name);
    }
}

static int read_line_fd(int fd, char *buffer, size_t capacity) {
    size_t used = 0;
    while (used + 1 < capacity) {
        char byte;
        ssize_t got = read(fd, &byte, 1);
        if (got <= 0) return (int)used;
        buffer[used++] = byte;
        if (byte == '\n') break;
    }
    buffer[used] = '\0';
    return (int)used;
}

static int write_all_fd(int fd, const char *text) {
    size_t length = strlen(text), offset = 0;
    while (offset < length) {
        ssize_t written = write(fd, text + offset, length - offset);
        if (written < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        offset += (size_t)written;
    }
    return 1;
}

static void record_command(fake_core *core, const char *command) {
    if (core->command_count < 32) {
        char *destination = core->commands[core->command_count++];
        size_t length = strlen(command);
        if (length >= sizeof(core->commands[0])) length = sizeof(core->commands[0]) - 1;
        memcpy(destination, command, length);
        destination[length] = '\0';
    }
}

static void *fake_core_main(void *argument) {
    fake_core *core = argument;
    int client = accept4(core->listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0) return NULL;
    char line[4096];
    for (;;) {
        int length = read_line_fd(client, line, sizeof(line));
        if (length <= 0) break;
        record_command(core, line);
        if (!strcmp(line, "info\n")) write_all_fd(client, "info 8192 0\n");
        else if (!strcmp(line, "stop 1 2\n")) write_all_fd(client, "ok: stop 1 2\n");
        else if (!strcmp(line, "sample 7 0 32 1\n")) write_all_fd(client, "ok: sample seed 7 temp 0 top_k 32 top_p 1 (greedy)\n");
        else if (!strcmp(line, "snap 3\n")) write_all_fd(client, "ok: snap 3\n");
        else if (!strncmp(line, "chat ", 5)) {
            if (core->mode == 3) {
                write_all_fd(client, "error: synthetic failure\n");
            } else if (core->mode == 4) {
                write_all_fd(client, "tok 101\ntok 102\ntok 103\ndone malformed\n");
            } else if (core->mode == 5) {
                close(client);
                return NULL;
            } else {
                /* One write exercises the relay's handling of pipelined token records. */
                write_all_fd(client, "tok 101\ntok 102\ntok 103\ndone 3 1 3 75.7 20.0 0 0\n");
                char config[2048];
                json_object *config_json = fixture_config(NULL);
                snprintf(config, sizeof(config), "%s", json_object_get_string(config_json));
                json_object_put(config_json);
                read_line_fd(client, line, sizeof(line));
                record_command(core, line);
                write_all_fd(client, config);
                for (int die = 0; die < 6; die++) {
                    read_line_fd(client, line, sizeof(line));
                    record_command(core, line);
                    json_object *profile = fixture_profile(die, 28.0, 5000.0);
                    const char *raw = json_object_get_string(json_object_object_get(profile, "raw"));
                    if (core->mode == 2 && die == 0) {
                        const char *footer = strstr(raw, "  longest:");
                        char incomplete[1024];
                        snprintf(incomplete, sizeof(incomplete), "%.*s", (int)(footer - raw), raw);
                        write_all_fd(client, incomplete);
                        json_object_put(profile);
                        close(client);
                        return NULL;
                    } else write_all_fd(client, raw);
                    json_object_put(profile);
                }
                if (core->mode == 1) {
                    int length = read_line_fd(client, line, sizeof(line));
                    if (length > 0) record_command(core, line);
                }
                char rest[256];
                while (read(client, rest, sizeof(rest)) > 0) {}
            }
        } else break;
    }
    close(client);
    return NULL;
}

static int setup_fake(const char *name, fake_core *core, pid_t expected_pid, pthread_t *worker,
                      char *upstream, size_t upstream_size, char *relay_path,
                      size_t relay_size, dc_relay **relay, char **error) {
    snprintf(upstream, upstream_size, "%s/%s-upstream.sock", directory, name);
    snprintf(relay_path, relay_size, "%s/%s-relay.sock", directory, name);
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/%s-protocol.jsonl", directory, name);
    core->listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    strcpy(address.sun_path, upstream);
    if (bind(core->listener, (struct sockaddr *)&address, sizeof(address)) != 0 || listen(core->listener, 4) != 0) return 0;
    if (pthread_create(worker, NULL, fake_core_main, core) != 0) return 0;
    *relay = dc_relay_open(relay_path, upstream, expected_pid, log_path, error);
    return *relay != NULL;
}

static int connect_relay(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    strcpy(address.sun_path, path);
    for (int attempt = 0; attempt < 50; attempt++) {
        if (connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0) return fd;
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 10000000};
        nanosleep(&delay, NULL);
    }
    close(fd);
    return -1;
}

static void teardown_fake(fake_core *core, pthread_t worker, int client,
                          dc_relay *relay, const char *upstream, char **error) {
    if (client >= 0) close(client);
    if (relay) dc_relay_close(relay, error);
    if (core->listener >= 0) close(core->listener);
    pthread_join(worker, NULL);
    unlink(upstream);
}

static void lock_tests(void) {
    char path[512], wrong_path[512], fd_text[32], *error = NULL;
    snprintf(path, sizeof(path), "%s/engine.lock", directory);
    snprintf(wrong_path, sizeof(wrong_path), "%s/wrong.lock", directory);
    int owner = open(path, O_CREAT | O_RDWR, 0600);
    expect(owner >= 0 && flock(owner, LOCK_EX | LOCK_NB) == 0, "lock owner acquires exclusive lease");
    snprintf(fd_text, sizeof(fd_text), "%d", owner);
    int duplicate = -1;
    expect(dc_validation_lock(path, "synthetic-parent", fd_text, &duplicate, &error), "inherited gpuq lease is accepted");
    expect(duplicate >= 0, "inherited lease descriptor is duplicated");
    close(duplicate);
    int probe = open(path, O_RDWR);
    expect(flock(probe, LOCK_EX | LOCK_NB) != 0, "closing child duplicate preserves parent lock");
    close(probe);
    close(owner);

    int path_fd = open(path, O_RDWR);
    int wrong_fd = open(wrong_path, O_CREAT | O_RDWR, 0600);
    snprintf(fd_text, sizeof(fd_text), "%d", wrong_fd);
    expect(!dc_validation_lock(path, "synthetic-parent", fd_text, &duplicate, &error), "descriptor for another inode is rejected");
    free(error); error = NULL;
    close(path_fd); close(wrong_fd);

    expect(!dc_validation_lock(path, "synthetic-parent", NULL, &duplicate, &error), "incomplete inherited lease is rejected");
    free(error); error = NULL;
    expect(!dc_validation_lock(path, "", NULL, &duplicate, &error), "empty inherited job ID is rejected");
    free(error); error = NULL;

    int owner2 = open(path, O_RDWR);
    flock(owner2, LOCK_EX | LOCK_NB);
    lock_worker state = {.path = path, .acquired = 0};
    pthread_t thread;
    pthread_create(&thread, NULL, lock_thread_main, &state);
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 50000000};
    nanosleep(&pause, NULL);
    expect(!atomic_load(&state.acquired), "standalone collector waits while lock is busy");
    close(owner2);
    pthread_join(thread, NULL);
    expect(atomic_load(&state.acquired), "standalone collector acquires after owner releases");
}

static void parser_tests(void) {
    const char *uuid = "GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f";
    char input[256];
    snprintf(input, sizeof(input), "0, %s, 0\n5, %s, 100\n", uuid, uuid);
    json_object *rows = NULL; char *error = NULL;
    expect(dc_nvidia_rows(input, 3, &rows, &error) && json_object_array_length(rows) == 2, "valid GPU rows parse without CSV gem");
    json_object_put(rows); rows = NULL;
    expect(dc_nvidia_rows("", 2, &rows, &error) && json_object_array_length(rows) == 0, "empty compute rows are accepted");
    json_object_put(rows); rows = NULL;
    const char *bad[] = {
        "0, GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f, N/A\n",
        "0, GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f, 101\n",
        "0, GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f\n",
        "0, GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f, 0, extra\n",
        "0, invalid-uuid, 0\n", "\"0\", GPU-1783441e-9a1a-684e-82bb-14d24e6e0c8f, 0\n", "\n"
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        free(error); error = NULL;
        expect(!dc_nvidia_rows(bad[i], 3, &rows, &error), "malformed GPU row is rejected");
        free(error); error = NULL;
    }
    free(error);

    const char *maps =
        "1000-2000 r-xp 00001000 08:13 100 /usr/lib/libc.so.6\n"
        "2000-3000 r--p 00001000 00:2b 200 /tmp/flashcore-123-gen21.so\n"
        "3000-4000 r-xp 00002000 00:2b 200 /tmp/flashcore-123-gen21.so\n"
        "4000-5000 r-xp 00002000 00:2b 300 /tmp/flashcore-124-gen1.so\n";
    char **fields = NULL;
    expect(dc_engine_mapping(123, maps, &fields, &error), "engine mapping identifies requested generation");
    expect(fields && !strcmp(fields[5], "/tmp/flashcore-123-gen21.so"), "requested generation path retained");
    expect(fields && !strcmp(fields[4], "200"), "mapped inode retained");
    if (fields) { for (int i = 0; i < 6; i++) free(fields[i]); free(fields); }
    free(error); error = NULL;
    expect(!dc_engine_mapping(125, maps, &fields, &error), "wrong process generation is rejected");
    free(error); error = NULL;
    const char *direct = "3000-4000 r-xp 00002000 00:2b 200 /private/libengine.so\n";
    expect(dc_engine_mapping(123, direct, &fields, &error) && !strcmp(fields[5], "/private/libengine.so"), "direct engine mapping remains supported");
    if (fields) { for (int i = 0; i < 6; i++) free(fields[i]); free(fields); }
    free(error); error = NULL;
    const char *duplicate = "3000-4000 r-xp 00002000 00:2b 200 /tmp/flashcore-123-gen21.so\n5000-6000 r-xp 00002000 00:2b 400 /tmp/flashcore-123-gen22.so\n";
    expect(!dc_engine_mapping(123, duplicate, &fields, &error), "multiple engine generations are rejected");
    free(error); error = NULL;
    expect(dc_stop_owned_child(getpid()), "cleanup does not signal a non-child process");
}

static void relay_success_test(void) {
    fake_core core = {.listener = -1, .mode = 1};
    pthread_t worker; char upstream[512], relay_path[512]; dc_relay *relay = NULL; char *error = NULL;
    expect(setup_fake("success", &core, getpid(), &worker, upstream, sizeof(upstream), relay_path, sizeof(relay_path), &relay, &error), "fake relay starts");
    int client = connect_relay(relay_path);
    expect(client >= 0, "front-end connects to relay");
    write_all_fd(client, "info\nstop 1 2\nsample 7 0 32 1\nsnap 3\nchat 39 1 2 3\n");
    char reply[256];
    int replies = 0;
    while (replies < 5) {
        if (read_line_fd(client, reply, sizeof(reply)) <= 0) break;
        replies++;
    }
    /* Five records are info, stop, sample, snap and the final done; tok lines are also relayed. */
    while (strcmp(reply, "done 3 1 3 75.7 20.0 0 0\n") && replies < 8) {
        if (read_line_fd(client, reply, sizeof(reply)) <= 0) break;
        replies++;
    }
    expect(!strcmp(reply, "done 3 1 3 75.7 20.0 0 0\n"), "pipelined chat returns done after six profiles");
    json_object *evidence = NULL;
    expect(dc_relay_pop(relay, &evidence, &error), "successful chat publishes evidence");
    expect(evidence && json_object_array_length(json_object_object_get(evidence, "input_tokens")) == 3, "input token IDs captured");
    expect(evidence && json_object_array_length(json_object_object_get(evidence, "output_tokens")) == 3, "output token IDs captured");
    expect(evidence && json_object_array_length(json_object_object_get(evidence, "profiles")) == 6, "all die profiles captured");
    if (evidence) json_object_put(evidence);
    free(error); error = NULL;
    teardown_fake(&core, worker, client, relay, upstream, &error);
    expect(core.command_count == 12, "one core connection serves commands and profiles");
    free(error);
}

static void relay_rejection_tests(void) {
    for (int mode = 2; mode <= 5; mode++) {
        fake_core core = {.listener = -1, .mode = mode};
        pthread_t worker; char upstream[512], relay_path[512]; dc_relay *relay = NULL; char *error = NULL;
        char name[32]; snprintf(name, sizeof(name), "failure-%d", mode);
        expect(setup_fake(name, &core, getpid(), &worker, upstream, sizeof(upstream), relay_path, sizeof(relay_path), &relay, &error), "failure relay starts");
        int client = connect_relay(relay_path);
        write_all_fd(client, "chat 39 1 2 3\n");
        json_object *evidence = NULL;
        expect(!dc_relay_pop(relay, &evidence, &error), "engine failure or incomplete profile rejects evidence");
        expect(error != NULL, "relay rejection reason is available");
        free(error); error = NULL;
        teardown_fake(&core, worker, client, relay, upstream, &error);
        free(error);
    }

    fake_core core = {.listener = -1, .mode = 0};
    pthread_t worker; char upstream[512], relay_path[512]; dc_relay *relay = NULL; char *error = NULL;
    expect(setup_fake("danger", &core, getpid(), &worker, upstream, sizeof(upstream), relay_path, sizeof(relay_path), &relay, &error), "danger relay starts");
    int client = connect_relay(relay_path);
    write_all_fd(client, "quit\n");
    json_object *evidence = NULL;
    expect(!dc_relay_pop(relay, &evidence, &error) && strstr(error, "refusing frontend command"), "dangerous command is refused");
    free(error); error = NULL;
    teardown_fake(&core, worker, client, relay, upstream, &error);
    free(error);

    core = (fake_core){.listener = -1, .mode = 0};
    expect(setup_fake("wrong-peer", &core, getpid() + 100000, &worker, upstream, sizeof(upstream), relay_path, sizeof(relay_path), &relay, &error), "peer-verification relay starts");
    client = connect_relay(relay_path);
    write_all_fd(client, "info\n");
    evidence = NULL;
    expect(!dc_relay_pop(relay, &evidence, &error) && error && strstr(error, "core peer PID"), "wrong core peer is rejected");
    expect(core.command_count == 0, "wrong peer receives no commands");
    free(error); error = NULL;
    teardown_fake(&core, worker, client, relay, upstream, &error);
    free(error);
}

static int remove_entry(const char *path, const struct stat *status, int type, struct FTW *walk) {
    (void)status; (void)type; (void)walk;
    return remove(path);
}

int main(void) {
    snprintf(directory, sizeof(directory), "/tmp/die-balance-capture-test-XXXXXX");
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 2; }
    lock_tests();
    parser_tests();
    relay_success_test();
    relay_rejection_tests();
    printf("die_balance_capture: %d checks, %d failures\n", checks, failures);
    nftw(directory, remove_entry, 16, FTW_DEPTH | FTW_PHYS);
    return failures ? 1 : 0;
}
