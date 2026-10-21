#define _XOPEN_SOURCE 700
#include <errno.h>
#include <ftw.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

static int child_status(int status) {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 125;
}

static int write_empty_marker(void) {
    const char *run_dir = getenv("GPUQ_RUN_DIR");
    char path[PATH_MAX];
    int fd;

    if (!run_dir || snprintf(path, sizeof(path), "%s/empty.once", run_dir) >= (int)sizeof(path)) {
        fputs("GPUQ_RUN_DIR is missing or too long\n", stderr);
        return 2;
    }
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        perror("create empty.once");
        return 2;
    }
    if (close(fd) != 0) {
        perror("close empty.once");
        return 2;
    }
    return 0;
}

static int run_command(char *const args[], const char *db_path,
                       const char *lock_path, const char *queue_path) {
    pid_t pid = fork();
    int status;

    if (pid < 0) {
        perror("fork");
        return 125;
    }
    if (pid == 0) {
        unsetenv("GPUQ_JOB_ID");
        unsetenv("GPUQ_LOCK_FD");
        unsetenv("GPUQ_RUN_DIR");
        if (setenv("GPUQ_DB", db_path, 1) != 0 ||
            setenv("GPUQ_LOCK", lock_path, 1) != 0 ||
            setenv("GPUQ_QUEUE", queue_path, 1) != 0) {
            perror("setenv");
            _exit(125);
        }
        execvp(args[0], args);
        perror(args[0]);
        _exit(127);
    }
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return 125;
        }
    }
    return child_status(status);
}

static int run_capture(char *const args[], const char *db_path,
                       const char *lock_path, const char *queue_path,
                       char *output, size_t output_size) {
    int pipe_fd[2];
    pid_t pid;
    int status;
    size_t used = 0;
    char discard[512];

    if (pipe(pipe_fd) != 0) {
        perror("pipe");
        return 125;
    }
    pid = fork();
    if (pid < 0) {
        perror("fork");
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return 125;
    }
    if (pid == 0) {
        close(pipe_fd[0]);
        if (dup2(pipe_fd[1], STDOUT_FILENO) < 0) {
            perror("dup2");
            _exit(125);
        }
        close(pipe_fd[1]);
        unsetenv("GPUQ_JOB_ID");
        unsetenv("GPUQ_LOCK_FD");
        unsetenv("GPUQ_RUN_DIR");
        if (setenv("GPUQ_DB", db_path, 1) != 0 ||
            setenv("GPUQ_LOCK", lock_path, 1) != 0 ||
            setenv("GPUQ_QUEUE", queue_path, 1) != 0) {
            perror("setenv");
            _exit(125);
        }
        execvp(args[0], args);
        perror(args[0]);
        _exit(127);
    }
    close(pipe_fd[1]);
    for (;;) {
        char *destination = output_size > 0 && used + 1 < output_size ? output + used : discard;
        size_t available = destination == discard ? sizeof(discard) : output_size - used - 1;
        ssize_t count = read(pipe_fd[0], destination, available);
        if (count > 0) {
            if (destination != discard) used += (size_t)count;
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        break;
    }
    close(pipe_fd[0]);
    if (output_size > 0) output[used] = '\0';
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return 125;
        }
    }
    return child_status(status);
}

static int remove_entry(const char *path, const struct stat *info, int type, struct FTW *walk) {
    (void)info;
    (void)type;
    (void)walk;
    return remove(path);
}

static int clean_tree(const char *path) {
    return nftw(path, remove_entry, 16, FTW_DEPTH | FTW_PHYS);
}

static int run_test(const char *mode, const char *repo_arg, const char *scratch_root) {
    char repo[PATH_MAX];
    char program[PATH_MAX];
    char script[PATH_MAX];
    char work_template[PATH_MAX];
    char db_path[PATH_MAX];
    char lock_path[PATH_MAX];
    char queue_path[PATH_MAX];
    char retained_json[4096] = {0};
    char logs_json[8192] = {0};
    int expected_reproduction = strcmp(mode, "--reproduce") == 0;
    int command_status, retained_status, logs_status, result = 1;

    if (!realpath(repo_arg, repo)) {
        perror("resolve repository path");
        return 2;
    }
    if (!realpath("/proc/self/exe", program)) {
        perror("resolve test executable");
        return 2;
    }
    if (snprintf(script, sizeof(script), "%s/tools/gpuq/gpuq.rb", repo) >= (int)sizeof(script) ||
        snprintf(work_template, sizeof(work_template), "%s/fs101-gpuq-empty-XXXXXX", scratch_root) >= (int)sizeof(work_template)) {
        fputs("test path is too long\n", stderr);
        return 2;
    }
    if (!mkdtemp(work_template)) {
        perror("create test directory");
        return 2;
    }
    if (snprintf(db_path, sizeof(db_path), "%s/history.sqlite3", work_template) >= (int)sizeof(db_path) ||
        snprintf(lock_path, sizeof(lock_path), "%s/queue.lock", work_template) >= (int)sizeof(lock_path) ||
        snprintf(queue_path, sizeof(queue_path), "%s/queue", work_template) >= (int)sizeof(queue_path)) {
        fputs("test database path is too long\n", stderr);
        clean_tree(work_template);
        return 2;
    }

    char *run_args[] = {
        "ruby", script, "--db", db_path, "run", "--exclude", "all",
        "--name", "fs101-empty-artifact-test", "--", program,
        "--write-empty-marker", NULL
    };
    command_status = run_command(run_args, db_path, lock_path, queue_path);

    char *retained_args[] = {
        "ruby", script, "--db", db_path, "query",
        "SELECT CASE WHEN j.status='succeeded' AND j.exit_code=0 AND COUNT(d.id)=1 THEN 1 ELSE 0 END AS retained FROM jobs j LEFT JOIN documents d ON d.job_id=j.id AND d.kind='artifact' AND d.bytes=0 WHERE j.label='fs101-empty-artifact-test' GROUP BY j.id ORDER BY j.submitted_ns DESC LIMIT 1",
        NULL
    };
    retained_status = run_capture(retained_args, db_path, lock_path, queue_path,
                                  retained_json, sizeof(retained_json));
    int retained = retained_status == 0 && strstr(retained_json, "\"retained\": 1") != NULL;

    char *logs_args[] = {"ruby", script, "--db", db_path, "logs", "latest", NULL};
    logs_status = run_capture(logs_args, db_path, lock_path, queue_path,
                              logs_json, sizeof(logs_json));
    int reproduced = logs_status == 0 &&
        strstr(logs_json, "NoMethodError: undefined method 'include?' for nil") != NULL;

    if (expected_reproduction && command_status == 70 && !retained && reproduced) {
        puts("reproduced #101: the command created an empty marker, artifact import raised NoMethodError, and no empty artifact was retained");
        result = 0;
    } else if (!expected_reproduction && command_status == 0 && retained) {
        puts("passed: the successful command retained its zero-byte artifact");
        result = 0;
    } else {
        fprintf(stderr,
                "%s: runner exit=%d, artifact_retained=%s, reported_importer_error=%s\n",
                expected_reproduction ? "expected the issue reproduction" : "expected successful empty-artifact import",
                command_status, retained ? "yes" : "no", reproduced ? "yes" : "no");
        if (retained_status != 0) fprintf(stderr, "artifact query exited %d: %s\n", retained_status, retained_json);
        if (logs_status != 0) fprintf(stderr, "logs query exited %d: %s\n", logs_status, logs_json);
    }

    if (clean_tree(work_template) != 0) {
        perror("remove test directory");
        return 2;
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--write-empty-marker") == 0) return write_empty_marker();
    if (argc != 4 || (strcmp(argv[1], "--test") != 0 && strcmp(argv[1], "--reproduce") != 0)) {
        fprintf(stderr, "usage: %s --test|--reproduce REPO_ROOT SCRATCH_ROOT\n", argv[0]);
        return 2;
    }
    return run_test(argv[1], argv[2], argv[3]);
}
