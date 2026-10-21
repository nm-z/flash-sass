#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int wait_for(pid_t pid) {
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        perror("waitpid");
        return 125;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 125;
}

static int run_to_files(char *const argv[], const char *out_path,
                        const char *err_path) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 125;
    }
    if (pid == 0) {
        int out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0) {
            perror("redirect runner output");
            _exit(125);
        }
        close(out);
        close(err);
        execvp(argv[0], argv);
        perror(argv[0]);
        _exit(127);
    }
    return wait_for(pid);
}

static int put_directory(const char *directory, const char *label) {
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        return 125;
    }

    pid_t put_pid = fork();
    if (put_pid < 0) {
        perror("fork gpuq put");
        close(pipefd[0]);
        close(pipefd[1]);
        return 125;
    }
    if (put_pid == 0) {
        close(pipefd[1]);
        if (dup2(pipefd[0], STDIN_FILENO) < 0) {
            perror("dup2 gpuq put");
            _exit(125);
        }
        close(pipefd[0]);
        execlp("gpuq", "gpuq", "put", label, (char *)NULL);
        perror("gpuq put");
        _exit(127);
    }

    pid_t tar_pid = fork();
    if (tar_pid < 0) {
        perror("fork tar");
        close(pipefd[0]);
        close(pipefd[1]);
        (void)wait_for(put_pid);
        return 125;
    }
    if (tar_pid == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) {
            perror("dup2 tar");
            _exit(125);
        }
        close(pipefd[1]);
        execlp("tar", "tar", "-czf", "-", "-C", directory, ".",
               (char *)NULL);
        perror("tar");
        _exit(127);
    }

    close(pipefd[0]);
    close(pipefd[1]);
    int tar_rc = wait_for(tar_pid);
    int put_rc = wait_for(put_pid);
    if (tar_rc != 0) fprintf(stderr, "tar exit status %d\n", tar_rc);
    if (put_rc != 0) fprintf(stderr, "gpuq put exit status %d\n", put_rc);
    return tar_rc != 0 ? tar_rc : put_rc;
}

static int remove_tree(const char *path) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork rm");
        return 125;
    }
    if (pid == 0) {
        execlp("rm", "rm", "-rf", path, (char *)NULL);
        perror("rm");
        _exit(127);
    }
    return wait_for(pid);
}

int main(void) {
    const char *job_id = getenv("GPUQ_JOB_ID");
    char work[PATH_MAX];
    char out[PATH_MAX];
    char stdout_path[PATH_MAX];
    char stderr_path[PATH_MAX];
    char label[256];
    if (!job_id || !*job_id) {
        fprintf(stderr, "GPUQ_JOB_ID is required\n");
        return 2;
    }
    if (snprintf(work, sizeof work, "/dev/shm/fs16-matrix-%s", job_id) >=
            (int)sizeof work ||
        snprintf(out, sizeof out, "%s/matrix", work) >= (int)sizeof out ||
        snprintf(stdout_path, sizeof stdout_path, "%s/runner.stdout", work) >=
            (int)sizeof stdout_path ||
        snprintf(stderr_path, sizeof stderr_path, "%s/runner.stderr", work) >=
            (int)sizeof stderr_path ||
        snprintf(label, sizeof label, "issue16.matrix.64cf.%s", job_id) >=
            (int)sizeof label) {
        fprintf(stderr, "GPUQ job identifier is too long\n");
        return 2;
    }
    if (mkdir(work, 0700) < 0) {
        perror(work);
        return 2;
    }
    if (setenv("CUDA_VISIBLE_DEVICES",
               "GPU-431371bd-b482-96b2-4b70-04f83c6d075b", 1) < 0 ||
        setenv("DIE_CUBIN", "/dev/shm/fs16-64cf/kern/die.cubin", 1) < 0 ||
        chdir("/dev/shm/fs16-64cf/kern") < 0) {
        perror("configure QSA matrix");
        (void)remove_tree(work);
        return 2;
    }

    const char *nu_bin = getenv("NU_BIN");
    if (!nu_bin || !*nu_bin) {
        const char *home = getenv("HOME");
        static char default_nu[PATH_MAX];
        if (!home || !*home) home = "/home/nate";
        if (snprintf(default_nu, sizeof default_nu, "%s/.local/bin/nu", home) >=
            (int)sizeof default_nu) {
            fprintf(stderr, "Nu executable path is too long\n");
            (void)remove_tree(work);
            return 2;
        }
        nu_bin = default_nu;
    }
    char *const command[] = {
        (char *)nu_bin, "--no-config-file", "run_qsa_gather_pair.nu",
        "--out", out, "--nv", "2000",
        "--iterations", "8", "--matrix", NULL
    };
    int run_rc = run_to_files(command, stdout_path, stderr_path);
    if (run_rc != 0)
        fprintf(stderr, "matrix runner failed with status %d; uploading diagnostics\n",
                run_rc);

    int put_rc = put_directory(work, label);
    if (put_rc == 0) {
        fprintf(stderr, "uploaded current-head matrix bundle as %s\n", label);
        int rm_rc = remove_tree(work);
        if (rm_rc != 0) fprintf(stderr, "could not remove %s (status %d)\n", work, rm_rc);
    } else {
        fprintf(stderr, "matrix bundle remains at %s after upload failure\n", work);
    }
    return run_rc != 0 ? run_rc : put_rc;
}
