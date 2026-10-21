#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
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
            perror("redirect child output");
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

static int validate_capture(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        return 1;
    }
    uint32_t ids[512];
    int ok = fread(ids, sizeof ids[0], 512, file) == 512 && fgetc(file) == EOF;
    if (fclose(file) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "captured IDs must contain exactly 512 uint32 values\n");
        return 1;
    }
    for (size_t i = 0; i < 512; ++i) {
        if (ids[i] >= 2000 || (i && ids[i - 1] >= ids[i])) {
            fprintf(stderr, "captured ID row is not sorted, unique, and in range\n");
            return 1;
        }
    }
    return 0;
}

static int csv_identity(const char *left_path, const char *right_path) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (!left || !right) {
        perror("open paired CSV");
        if (left) fclose(left);
        if (right) fclose(right);
        return 1;
    }
    char a[2048], b[2048];
    unsigned line_number = 0;
    int failed = 0;
    while (fgets(a, sizeof a, left) && fgets(b, sizeof b, right)) {
        ++line_number;
        if (line_number == 1) {
            if (strcmp(a, b) != 0) failed = 1;
            continue;
        }
        char *af[10], *bf[10];
        size_t na = 0, nb = 0;
        char *save_a = NULL, *save_b = NULL;
        for (char *v = strtok_r(a, ",\r\n", &save_a); v && na < 10;
             v = strtok_r(NULL, ",\r\n", &save_a)) af[na++] = v;
        for (char *v = strtok_r(b, ",\r\n", &save_b); v && nb < 10;
             v = strtok_r(NULL, ",\r\n", &save_b)) bf[nb++] = v;
        if (na != 10 || nb != 10) {
            failed = 1;
            continue;
        }
        for (size_t i = 0; i < 10; ++i) {
            if (i != 7 && strcmp(af[i], bf[i]) != 0) failed = 1;
        }
    }
    if (fgets(a, sizeof a, left) || fgets(b, sizeof b, right)) failed = 1;
    fclose(left);
    fclose(right);
    if (failed) fprintf(stderr, "paired CSV data differ outside GPU timing\n");
    return failed;
}

static int check_pair(const char *work, const char *name, int changing,
                      const char *csv0, const char *csv1) {
    char json0[PATH_MAX], json1[PATH_MAX], cmp[PATH_MAX], err[PATH_MAX];
    if (snprintf(json0, sizeof json0, "%s/%s-host0.json", work, name) >=
            (int)sizeof json0 ||
        snprintf(json1, sizeof json1, "%s/%s-host1.json", work, name) >=
            (int)sizeof json1 ||
        snprintf(cmp, sizeof cmp, "%s/%s-pair-check.json", work, name) >=
            (int)sizeof cmp ||
        snprintf(err, sizeof err, "%s/%s-pair-check.stderr", work, name) >=
            (int)sizeof err) {
        fprintf(stderr, "pair path is too long\n");
        return 2;
    }
    const char *filter =
        "($h0[0].preflight_verified_all == true) and "
        "($h1[0].preflight_verified_all == true) and "
        "($h0[0].source_revision == $h1[0].source_revision) and "
        "($h0[0].pattern == $h1[0].pattern) and "
        "($h0[0].position_start == $h1[0].position_start) and "
        "($h0[0].position_end == $h1[0].position_end) and "
        "($h0[0].cache_fnv1a64 == $h1[0].cache_fnv1a64) and "
        "($h0[0].selections_fnv1a64 == $h1[0].selections_fnv1a64) and "
        "($h0[0].logical_keys == $h1[0].logical_keys) and "
        "($h0[0].physical_gathered_slots == $h1[0].physical_gathered_slots) and "
        "($h0[0].selection_changes == $h1[0].selection_changes) and "
        "($h0[0].verified_output_signature_changes == $h1[0].verified_output_signature_changes)";
    char *const jq_argv[] = {"jq", "-n", "-e", "--slurpfile", "h0", json0,
                            "--slurpfile", "h1", json1, (char *)filter, NULL};
    int jq_rc = run_to_files(jq_argv, cmp, err);
    int csv_rc = csv_identity(csv0, csv1);
    if (jq_rc != 0 || csv_rc != 0) return 1;

    if (changing) {
        const char *change_filter =
            "($h0[0].selection_changes == 7) and "
            "($h1[0].selection_changes == 7) and "
            "($h0[0].verified_output_signature_changes == 7) and "
            "($h1[0].verified_output_signature_changes == 7)";
        char *const change_argv[] = {"jq", "-n", "-e", "--slurpfile", "h0",
                                     json0, "--slurpfile", "h1", json1,
                                     (char *)change_filter, NULL};
        jq_rc = run_to_files(change_argv, cmp, err);
        if (jq_rc != 0) {
            fprintf(stderr, "changing-token selections/outputs did not all change\n");
            return 1;
        }
    }
    return 0;
}

static int run_pair(const char *work, const char *pattern, int iterations,
                    const char *ids_path, int changing) {
    char iter_text[16];
    if (snprintf(iter_text, sizeof iter_text, "%d", iterations) >=
        (int)sizeof iter_text) return 2;
    char csv[2][PATH_MAX], json[2][PATH_MAX], err[2][PATH_MAX];
    for (int host = 0; host < 2; ++host) {
        char host_text[2] = {(char)('0' + host), '\0'};
        if (snprintf(csv[host], sizeof csv[host], "%s/%s-host%s.csv", work,
                     pattern, host_text) >= (int)sizeof csv[host] ||
            snprintf(json[host], sizeof json[host], "%s/%s-host%s.json", work,
                     pattern, host_text) >= (int)sizeof json[host] ||
            snprintf(err[host], sizeof err[host], "%s/%s-host%s.stderr", work,
                     pattern, host_text) >= (int)sizeof err[host]) {
            fprintf(stderr, "pair output path is too long\n");
            return 2;
        }
        if (ids_path) {
            if (setenv("GATHER_IDS", ids_path, 1) < 0) {
                perror("setenv GATHER_IDS");
                return 2;
            }
        } else {
            unsetenv("GATHER_IDS");
        }
        char *const argv[] = {"./qsa_gather_bench", "2000", "0", iter_text,
                              (char *)pattern, host_text, csv[host], "4242", NULL};
        int rc = run_to_files(argv, json[host], err[host]);
        if (rc != 0) {
            fprintf(stderr, "%s host=%s failed with status %d\n", pattern,
                    host_text, rc);
            return rc;
        }
    }
    if (unsetenv("GATHER_IDS") < 0) {
        perror("unsetenv GATHER_IDS");
        return 2;
    }
    return check_pair(work, pattern, changing, csv[0], csv[1]);
}

int main(void) {
    const char *job_id = getenv("GPUQ_JOB_ID");
    char work[PATH_MAX], ids_path[PATH_MAX], stdout_path[PATH_MAX];
    char stderr_path[PATH_MAX], label[256];
    if (!job_id || !*job_id) {
        fprintf(stderr, "GPUQ_JOB_ID is required\n");
        return 2;
    }
    if (snprintf(work, sizeof work, "/dev/shm/fs16-capture-%s", job_id) >=
            (int)sizeof work ||
        snprintf(ids_path, sizeof ids_path, "%s/op-idxsel-pos7999.ids.u32", work) >=
            (int)sizeof ids_path ||
        snprintf(stdout_path, sizeof stdout_path, "%s/idxfull.stdout", work) >=
            (int)sizeof stdout_path ||
        snprintf(stderr_path, sizeof stderr_path, "%s/idxfull.stderr", work) >=
            (int)sizeof stderr_path ||
        snprintf(label, sizeof label, "issue16.selection-pairs.64cf.%s", job_id) >=
            (int)sizeof label) {
        fprintf(stderr, "GPUQ job identifier is too long\n");
        return 2;
    }
    if (mkdir(work, 0700) < 0) {
        perror(work);
        return 2;
    }
    char origin_path[PATH_MAX];
    if (snprintf(origin_path, sizeof origin_path, "%s/selection-origin.txt", work) >=
        (int)sizeof origin_path) {
        fprintf(stderr, "selection origin path is too long\n");
        (void)remove_tree(work);
        return 2;
    }
    FILE *origin = fopen(origin_path, "wb");
    if (!origin) {
        perror(origin_path);
        (void)remove_tree(work);
        return 2;
    }
    fputs("OP_IDXSEL output from qsa_test idxfull at position 7999; synthetic query/key/pool fixture, not checkpoint-derived model selection.\n",
          origin);
    if (fclose(origin) != 0) {
        perror(origin_path);
        (void)remove_tree(work);
        return 2;
    }
    if (setenv("CUDA_VISIBLE_DEVICES",
               "GPU-431371bd-b482-96b2-4b70-04f83c6d075b", 1) < 0 ||
        setenv("DIE_CUBIN", "/dev/shm/fs16-64cf/kern/die.cubin", 1) < 0 ||
        setenv("QSA_CAPTURE_IDS", ids_path, 1) < 0 ||
        chdir("/dev/shm/fs16-64cf/kern") < 0) {
        perror("configure idxfull capture");
        (void)remove_tree(work);
        return 2;
    }
    char *const capture_argv[] = {"./qsa_test", "idxfull", "2000", "512", "0", NULL};
    int run_rc = run_to_files(capture_argv, stdout_path, stderr_path);
    if (run_rc == 0) run_rc = validate_capture(ids_path);
    if (run_rc == 0 && unsetenv("QSA_CAPTURE_IDS") < 0) {
        perror("unsetenv QSA_CAPTURE_IDS");
        run_rc = 2;
    }
    if (run_rc == 0) run_rc = run_pair(work, "captured", 8, ids_path, 0);
    if (run_rc == 0) run_rc = run_pair(work, "captured-token", 8, ids_path, 0);
    if (run_rc == 0) run_rc = run_pair(work, "changing-token", 8, NULL, 1);
    if (run_rc != 0)
        fprintf(stderr, "selection/pair validation failed with status %d; uploading available diagnostics\n",
                run_rc);

    int put_rc = put_directory(work, label);
    if (put_rc == 0) {
        fprintf(stderr, "uploaded selection/pair bundle as %s\n", label);
        int rm_rc = remove_tree(work);
        if (rm_rc != 0) fprintf(stderr, "could not remove %s (status %d)\n", work, rm_rc);
    } else {
        fprintf(stderr, "selection/pair bundle remains at %s after upload failure\n", work);
    }
    return run_rc != 0 ? run_rc : put_rc;
}
