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
            perror("redirect profiler output");
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

static int csv_identity(const char *left_path, const char *right_path) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (!left || !right) {
        perror("open paired profile CSV");
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
    if (failed) fprintf(stderr, "paired profile CSV inputs/outputs differ\n");
    return failed;
}

static int check_results(const char *work, const char *group) {
    char json0[PATH_MAX], json1[PATH_MAX], csv0[PATH_MAX], csv1[PATH_MAX];
    char check[PATH_MAX], err[PATH_MAX];
    if (snprintf(json0, sizeof json0, "%s/result-host0.json", work) >=
            (int)sizeof json0 ||
        snprintf(json1, sizeof json1, "%s/result-host1.json", work) >=
            (int)sizeof json1 ||
        snprintf(csv0, sizeof csv0, "%s/timing-host0.csv", work) >=
            (int)sizeof csv0 ||
        snprintf(csv1, sizeof csv1, "%s/timing-host1.csv", work) >=
            (int)sizeof csv1 ||
        snprintf(check, sizeof check, "%s/pair-check.json", work) >=
            (int)sizeof check ||
        snprintf(err, sizeof err, "%s/pair-check.stderr", work) >=
            (int)sizeof err) {
        fprintf(stderr, "profile validation path is too long\n");
        return 2;
    }
    const char *filter =
        "($h0[0].preflight_verified_all == true) and "
        "($h1[0].preflight_verified_all == true) and "
        "($h0[0].profile_first_launch_requested == true) and "
        "($h1[0].profile_first_launch_requested == true) and "
        "($h0[0].source_revision == $h1[0].source_revision) and "
        "($h0[0].pattern == \"captured\") and "
        "($h1[0].pattern == \"captured\") and "
        "($h0[0].cache_fnv1a64 == $h1[0].cache_fnv1a64) and "
        "($h0[0].selections_fnv1a64 == $h1[0].selections_fnv1a64) and "
        "($h0[0].logical_keys == $h1[0].logical_keys) and "
        "($h0[0].physical_gathered_slots == $h1[0].physical_gathered_slots) and "
        "($h0[0].verified_output_signature_changes == $h1[0].verified_output_signature_changes)";
    char *const jq_argv[] = {"jq", "-n", "-e", "--slurpfile", "h0", json0,
                            "--slurpfile", "h1", json1, (char *)filter, NULL};
    int jq_rc = run_to_files(jq_argv, check, err);
    int csv_rc = csv_identity(csv0, csv1);
    if (jq_rc != 0 || csv_rc != 0) {
        fprintf(stderr, "%s profile pair validation failed\n", group);
        return 1;
    }
    return 0;
}

static int run_profile_arm(const char *work, const char *metrics, int host) {
    char host_text[2] = {(char)('0' + host), '\0'};
    char profile_csv[PATH_MAX], timing_csv[PATH_MAX];
    char json[PATH_MAX], err[PATH_MAX];
    if (snprintf(profile_csv, sizeof profile_csv, "%s/nvprof-host%s.csv", work,
                 host_text) >= (int)sizeof profile_csv ||
        snprintf(timing_csv, sizeof timing_csv, "%s/timing-host%s.csv", work,
                 host_text) >= (int)sizeof timing_csv ||
        snprintf(json, sizeof json, "%s/result-host%s.json", work,
                 host_text) >= (int)sizeof json ||
        snprintf(err, sizeof err, "%s/nvprof-host%s.stderr", work,
                 host_text) >= (int)sizeof err) {
        fprintf(stderr, "profile output path is too long\n");
        return 2;
    }
    char *const argv[] = {
        "/usr/bin/sudo", "-n", "/usr/bin/env",
        "CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b",
        "DIE_CUBIN=/dev/shm/fs16-64cf/kern/die.cubin",
        "GATHER_IDS=/dev/shm/fs16-64cf/kern/op-idxsel-pos7999.ids.u32",
        "QSA_PROFILE_FIRST=1",
        "LD_LIBRARY_PATH=/dev/shm/fs16-64cf/tools/nvprof114/lib",
        "/dev/shm/fs16-64cf/tools/nvprof114/bin/nvprof",
        "--csv", "--profile-from-start", "off", "--kernels", "die",
        "--metrics", (char *)metrics, "--log-file", profile_csv,
        "./qsa_gather_bench", "2000", "0", "8", "captured", host_text,
        timing_csv, "4242", NULL
    };
    return run_to_files(argv, json, err);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: issue16_profile_launcher traffic|transactions|stalls\n");
        return 2;
    }
    const char *group = argv[1];
    const char *metrics = NULL;
    if (!strcmp(group, "traffic"))
        metrics = "sysmem_read_bytes,sysmem_write_bytes,dram_read_bytes,dram_write_bytes";
    else if (!strcmp(group, "transactions"))
        metrics = "sysmem_read_transactions,sysmem_write_transactions,dram_read_transactions,dram_write_transactions,gld_transactions,gst_transactions";
    else if (!strcmp(group, "stalls"))
        metrics = "stall_memory_dependency,stall_memory_throttle,stall_exec_dependency,stall_sync";
    else {
        fprintf(stderr, "unknown profiler group: %s\n", group);
        return 2;
    }

    const char *job_id = getenv("GPUQ_JOB_ID");
    char work[PATH_MAX], ids_path[PATH_MAX], config_path[PATH_MAX];
    char label[256];
    if (!job_id || !*job_id) {
        fprintf(stderr, "GPUQ_JOB_ID is required\n");
        return 2;
    }
    if (snprintf(work, sizeof work, "/dev/shm/fs16-profile-%s", job_id) >=
            (int)sizeof work ||
        snprintf(ids_path, sizeof ids_path,
                 "/dev/shm/fs16-64cf/kern/op-idxsel-pos7999.ids.u32") >=
            (int)sizeof ids_path ||
        snprintf(config_path, sizeof config_path, "%s/profile-config.txt", work) >=
            (int)sizeof config_path ||
        snprintf(label, sizeof label, "issue16.profile.64cf.%s.%s", group,
                 job_id) >= (int)sizeof label) {
        fprintf(stderr, "GPUQ job identifier is too long\n");
        return 2;
    }
    if (mkdir(work, 0700) < 0) {
        perror(work);
        return 2;
    }
    FILE *config = fopen(config_path, "wb");
    if (!config) {
        perror(config_path);
        (void)remove_tree(work);
        return 2;
    }
    fprintf(config,
            "source_revision=64cf729\nmetric_group=%s\nmetrics=%s\n"
            "nv=2000\ntail=0\nposition=7999\niterations=8\nseed=4242\n"
            "captured_ids=/dev/shm/fs16-64cf/kern/op-idxsel-pos7999.ids.u32\n"
            "QSA_PROFILE_FIRST=1\nnvprof_profile_from_start=off\n",
            group, metrics);
    if (fclose(config) != 0) {
        perror(config_path);
        (void)remove_tree(work);
        return 2;
    }

    if (setenv("CUDA_VISIBLE_DEVICES",
               "GPU-431371bd-b482-96b2-4b70-04f83c6d075b", 1) < 0 ||
        setenv("DIE_CUBIN", "/dev/shm/fs16-64cf/kern/die.cubin", 1) < 0 ||
        setenv("GATHER_IDS", ids_path, 1) < 0 ||
        setenv("QSA_PROFILE_FIRST", "1", 1) < 0 ||
        setenv("LD_LIBRARY_PATH", "/dev/shm/fs16-64cf/tools/nvprof114/lib", 1) < 0 ||
        chdir("/dev/shm/fs16-64cf/kern") < 0) {
        perror("configure Maxwell profile");
        (void)remove_tree(work);
        return 2;
    }

    int host0_rc = run_profile_arm(work, metrics, 0);
    int host1_rc = run_profile_arm(work, metrics, 1);
    int check_rc = (host0_rc == 0 && host1_rc == 0) ?
                   check_results(work, group) : 1;
    int run_rc = host0_rc != 0 ? host0_rc : host1_rc;
    if (check_rc != 0) run_rc = run_rc != 0 ? run_rc : check_rc;
    if (run_rc != 0)
        fprintf(stderr, "%s profile pair failed with status %d; uploading diagnostics\n",
                group, run_rc);

    int put_rc = put_directory(work, label);
    if (put_rc == 0) {
        fprintf(stderr, "uploaded %s profile bundle as %s\n", group, label);
        int rm_rc = remove_tree(work);
        if (rm_rc != 0) fprintf(stderr, "could not remove %s (status %d)\n", work, rm_rc);
    } else {
        fprintf(stderr, "profile bundle remains at %s after upload failure\n", work);
    }
    return run_rc != 0 ? run_rc : put_rc;
}
