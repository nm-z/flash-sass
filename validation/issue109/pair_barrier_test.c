#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_MAX_INPUT (64u * 1024u * 1024u)
#define TEST_PATH_CAP 4096u

static const char pair_branch[] = "@!P5 BRA PHASE_END;";
static const char pair_fence[] = "MEMBAR.GL.IVALLD;";
static const char patch_begin[] = "ISSUE109_PAIR_BODY_GRID_BARRIER_BEGIN";
static const char patch_end[] = "ISSUE109_PAIR_BODY_GRID_BARRIER_END";

static int failures;

static size_t find_bytes(const char *haystack, size_t haystack_len,
                         const char *needle, size_t needle_len, size_t start) {
    if (!needle_len || needle_len > haystack_len ||
        start > haystack_len - needle_len) return SIZE_MAX;
    for (size_t i = start; i <= haystack_len - needle_len; i++)
        if (!memcmp(haystack + i, needle, needle_len)) return i;
    return SIZE_MAX;
}

static size_t count_bytes(const char *text, size_t length,
                          const char *needle, size_t needle_len) {
    size_t count = 0, at = 0;
    for (;;) {
        size_t found = find_bytes(text, length, needle, needle_len, at);
        if (found == SIZE_MAX) return count;
        count++;
        at = found + needle_len;
    }
}

static int read_file(const char *path, char **data_out, size_t *length_out) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st;
    char *data;
    size_t length, offset = 0;
    if (fd < 0) {
        fprintf(stderr, "read %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fstat(fd, &st) < 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uint64_t)st.st_size > TEST_MAX_INPUT) {
        fprintf(stderr, "%s is not a bounded regular file\n", path);
        close(fd);
        return -1;
    }
    length = (size_t)st.st_size;
    data = malloc(length + 1u);
    if (!data) {
        fprintf(stderr, "allocate input snapshot\n");
        close(fd);
        return -1;
    }
    while (offset < length) {
        ssize_t n = read(fd, data + offset, length - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            fprintf(stderr, "read %s: short read\n", path);
            free(data);
            close(fd);
            return -1;
        }
        offset += (size_t)n;
    }
    if (close(fd) < 0) {
        fprintf(stderr, "close %s: %s\n", path, strerror(errno));
        free(data);
        return -1;
    }
    if (memchr(data, '\0', length)) {
        fprintf(stderr, "%s contains NUL bytes\n", path);
        free(data);
        return -1;
    }
    data[length] = '\0';
    *data_out = data;
    *length_out = length;
    return 0;
}

static int write_new_file(const char *path, const char *data, size_t length) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    size_t offset = 0;
    int saved = 0;
    if (fd < 0) {
        fprintf(stderr, "create %s: %s\n", path, strerror(errno));
        return -1;
    }
    while (offset < length) {
        ssize_t n = write(fd, data + offset, length - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            saved = errno ? errno : EIO;
            break;
        }
        offset += (size_t)n;
    }
    if (!saved && fsync(fd) < 0) saved = errno;
    if (close(fd) < 0 && !saved) saved = errno;
    if (saved) {
        unlink(path);
        fprintf(stderr, "write %s: %s\n", path, strerror(saved));
        return -1;
    }
    return 0;
}

static int make_path(char *out, size_t capacity, const char *directory,
                     const char *name) {
    int n = snprintf(out, capacity, "%s/%s", directory, name);
    if (n < 0 || (size_t)n >= capacity) {
        fprintf(stderr, "temporary test path is too long\n");
        return -1;
    }
    return 0;
}

static int run_patcher(const char *program, const char *input, const char *output) {
    pid_t pid = fork();
    int status;
    if (pid < 0) {
        fprintf(stderr, "fork patcher: %s\n", strerror(errno));
        return 127;
    }
    if (pid == 0) {
        char *const args[] = {(char *)program, (char *)input, (char *)output, NULL};
        execvp(program, args);
        fprintf(stderr, "exec %s: %s\n", program, strerror(errno));
        _exit(127);
    }
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fprintf(stderr, "wait for patcher: %s\n", strerror(errno));
        return 127;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128;
}

static void record_failure(const char *name, const char *reason) {
    fprintf(stderr, "FAIL %s: %s\n", name, reason);
    failures++;
}

static int file_absent(const char *path) {
    if (access(path, F_OK) < 0 && errno == ENOENT) return 1;
    return 0;
}

static int file_equals(const char *path, const char *expected, size_t expected_len) {
    char *actual = NULL;
    size_t actual_len = 0;
    int equal;
    if (read_file(path, &actual, &actual_len) < 0) return 0;
    equal = actual_len == expected_len && !memcmp(actual, expected, expected_len);
    free(actual);
    return equal;
}

static void expect_reject(const char *name, const char *program,
                          const char *input, const char *output) {
    int rc = run_patcher(program, input, output);
    if (rc == 0) record_failure(name, "patcher accepted invalid or conflicting input");
    if (!file_absent(output)) record_failure(name, "patcher created an output on rejection");
}

static int replace_once(const char *input, size_t input_len,
                        const char *old_text, const char *new_text,
                        char **output_out, size_t *output_len_out) {
    size_t old_len = strlen(old_text), new_len = strlen(new_text);
    size_t at = find_bytes(input, input_len, old_text, old_len, 0);
    size_t output_len;
    char *output;
    if (at == SIZE_MAX || old_len > input_len) return -1;
    output_len = input_len - old_len + new_len;
    output = malloc(output_len + 1u);
    if (!output) return -1;
    memcpy(output, input, at);
    memcpy(output + at, new_text, new_len);
    memcpy(output + at + new_len, input + at + old_len,
           input_len - at - old_len);
    output[output_len] = '\0';
    *output_out = output;
    *output_len_out = output_len;
    return 0;
}

static int duplicate_anchor(const char *input, size_t input_len,
                            char **output_out, size_t *output_len_out) {
    static const char duplicate[] =
        "\n01:-:-:-:6      ISETP.EQ.U32.AND P5, PT, bar_g, 0x8, PT;\n"
        "--:-:-:-:6 @!P5 BRA PHASE_END;\n"
        "--:-:-:-:6      MEMBAR.GL.IVALLD;\n";
    size_t suffix_len = sizeof duplicate - 1u;
    char *output = malloc(input_len + suffix_len + 1u);
    if (!output) return -1;
    memcpy(output, input, input_len);
    memcpy(output + input_len, duplicate, suffix_len + 1u);
    *output_out = output;
    *output_len_out = input_len + suffix_len;
    return 0;
}

static int malformed_fence(const char *input, size_t input_len,
                           char **output_out) {
    size_t branch = find_bytes(input, input_len, pair_branch, sizeof pair_branch - 1u, 0);
    size_t next_start, next_end, fence_at;
    char *output;
    if (branch == SIZE_MAX) return -1;
    next_start = branch;
    while (next_start < input_len && input[next_start] != '\n') next_start++;
    if (next_start == input_len) return -1;
    next_start++;
    next_end = next_start;
    while (next_end < input_len && input[next_end] != '\n') next_end++;
    fence_at = find_bytes(input + next_start, next_end - next_start,
                          pair_fence, sizeof pair_fence - 1u, 0);
    if (fence_at == SIZE_MAX) return -1;
    fence_at += next_start;
    output = malloc(input_len + 1u);
    if (!output) return -1;
    memcpy(output, input, input_len + 1u);
    memcpy(output + fence_at, "NOP;", 4u);
    memset(output + fence_at + 4u, ' ', sizeof pair_fence - 1u - 4u);
    *output_out = output;
    return 0;
}

static int malformed_schedule(const char *input, size_t input_len,
                              char **output_out) {
    size_t branch = find_bytes(input, input_len, pair_branch, sizeof pair_branch - 1u, 0);
    size_t line_start;
    char *output;
    if (branch == SIZE_MAX) return -1;
    line_start = branch;
    while (line_start && input[line_start - 1u] != '\n') line_start--;
    if (branch - line_start < 10u) return -1;
    output = malloc(input_len + 1u);
    if (!output) return -1;
    memcpy(output, input, input_len + 1u);
    memcpy(output + line_start, "01:-:-:-:6", 10u);
    *output_out = output;
    return 0;
}

static void cleanup_file(const char *directory, const char *name) {
    char path[TEST_PATH_CAP];
    if (make_path(path, sizeof path, directory, name) == 0) unlink(path);
}

int main(int argc, char **argv) {
    char directory_template[] = "/tmp/issue109-pair-barrier-XXXXXX";
    char *directory;
    char source_copy[TEST_PATH_CAP], positive[TEST_PATH_CAP], already[TEST_PATH_CAP];
    char missing_input[TEST_PATH_CAP], missing_output[TEST_PATH_CAP];
    char bad_fence_input[TEST_PATH_CAP], bad_fence_output[TEST_PATH_CAP];
    char bad_schedule_input[TEST_PATH_CAP], bad_schedule_output[TEST_PATH_CAP];
    char duplicate_input[TEST_PATH_CAP], duplicate_output[TEST_PATH_CAP];
    char sentinel[TEST_PATH_CAP], hardlink_alias[TEST_PATH_CAP];
    char *source = NULL, *patched = NULL, *mutation = NULL;
    size_t source_len = 0, patched_len = 0, mutation_len = 0;
    int rc, setup_failed = 0;
    static const char absent_branch[] = "@!P5 BRA PHASE_END_INVALID;";
    static const char sentinel_data[] = "KEEP-EXISTING-OUTPUT\n";
    static const char *const cleanup_names[] = {
        "source-copy.sass", "positive.sass", "already.sass",
        "missing.in", "missing.out", "bad-fence.in", "bad-fence.out",
        "bad-schedule.in", "bad-schedule.out", "duplicate.in", "duplicate.out",
        "existing-output.sass", "hardlink-alias.sass"
    };

    if (argc != 3) {
        fprintf(stderr, "usage: %s PATCHER GENERATED_DIE_SASS\n", argv[0]);
        return 2;
    }
    if (read_file(argv[2], &source, &source_len) < 0) return 1;
    directory = mkdtemp(directory_template);
    if (!directory) {
        fprintf(stderr, "create temporary test directory: %s\n", strerror(errno));
        free(source);
        return 1;
    }
    if (make_path(source_copy, sizeof source_copy, directory, "source-copy.sass") < 0 ||
        make_path(positive, sizeof positive, directory, "positive.sass") < 0 ||
        make_path(already, sizeof already, directory, "already.sass") < 0 ||
        make_path(missing_input, sizeof missing_input, directory, "missing.in") < 0 ||
        make_path(missing_output, sizeof missing_output, directory, "missing.out") < 0 ||
        make_path(bad_fence_input, sizeof bad_fence_input, directory, "bad-fence.in") < 0 ||
        make_path(bad_fence_output, sizeof bad_fence_output, directory, "bad-fence.out") < 0 ||
        make_path(bad_schedule_input, sizeof bad_schedule_input, directory, "bad-schedule.in") < 0 ||
        make_path(bad_schedule_output, sizeof bad_schedule_output, directory, "bad-schedule.out") < 0 ||
        make_path(duplicate_input, sizeof duplicate_input, directory, "duplicate.in") < 0 ||
        make_path(duplicate_output, sizeof duplicate_output, directory, "duplicate.out") < 0 ||
        make_path(sentinel, sizeof sentinel, directory, "existing-output.sass") < 0 ||
        make_path(hardlink_alias, sizeof hardlink_alias, directory, "hardlink-alias.sass") < 0) {
        free(source);
        rmdir(directory);
        return 1;
    }

    rc = run_patcher(argv[1], argv[2], positive);
    if (rc != 0) record_failure("generated-source-positive", "patcher rejected generated die.sass");
    if (read_file(positive, &patched, &patched_len) < 0) {
        record_failure("generated-source-positive", "patched output is missing or unreadable");
    } else {
        size_t branch = find_bytes(patched, patched_len, pair_branch, sizeof pair_branch - 1u, 0);
        size_t branch_end = branch;
        size_t begin_count = count_bytes(patched, patched_len, patch_begin, sizeof patch_begin - 1u);
        size_t end_count = count_bytes(patched, patched_len, patch_end, sizeof patch_end - 1u);
        size_t begin_at = find_bytes(patched, patched_len, patch_begin, sizeof patch_begin - 1u, 0);
        size_t end_at = find_bytes(patched, patched_len, patch_end, sizeof patch_end - 1u, 0);
        if (branch == SIZE_MAX) {
            record_failure("generated-source-positive", "branch anchor disappeared");
        } else {
            while (branch_end < patched_len && patched[branch_end] != '\n') branch_end++;
            if (branch_end == patched_len ||
                find_bytes(patched, patched_len, "# ISSUE109_PAIR_BODY_GRID_BARRIER_BEGIN\n",
                           sizeof "# ISSUE109_PAIR_BODY_GRID_BARRIER_BEGIN\n" - 1u,
                           branch_end + 1u) != branch_end + 1u)
                record_failure("generated-source-positive", "barrier was not inserted between A and B");
        }
        if (begin_count != 1u || end_count != 1u || begin_at == SIZE_MAX || end_at <= begin_at)
            record_failure("generated-source-positive", "barrier marker pair is missing or duplicated");
        else {
            size_t body_len = end_at - begin_at;
            if (find_bytes(patched + begin_at, body_len, "XG", 2u, 0) != SIZE_MAX ||
                find_bytes(patched + begin_at, body_len, "FLAGS_OFF", 9u, 0) != SIZE_MAX)
                record_failure("generated-source-positive", "inserted barrier contains XG/peer-flag handling");
        }
    }
    if (!file_equals(argv[2], source, source_len))
        record_failure("generated-source-positive", "patcher modified the real generated input");

    expect_reject("already-patched-input", argv[1], positive, already);
    if (patched && !file_equals(positive, patched, patched_len))
        record_failure("existing-output-preservation", "positive output changed unexpectedly");
    rc = run_patcher(argv[1], argv[2], positive);
    if (rc == 0) record_failure("existing-output-preservation", "existing output path was accepted");
    if (patched && !file_equals(positive, patched, patched_len))
        record_failure("existing-output-preservation", "existing output was overwritten");

    if (replace_once(source, source_len, pair_branch, absent_branch,
                     &mutation, &mutation_len) < 0 ||
        write_new_file(missing_input, mutation, mutation_len) < 0) {
        record_failure("missing-anchor", "could not create missing-anchor input");
        setup_failed = 1;
    } else {
        expect_reject("missing-anchor", argv[1], missing_input, missing_output);
    }
    free(mutation); mutation = NULL; mutation_len = 0;

    if (!setup_failed) {
        if (malformed_fence(source, source_len, &mutation) < 0 ||
            write_new_file(bad_fence_input, mutation, source_len) < 0) {
            record_failure("malformed-fence", "could not create malformed-fence input");
            setup_failed = 1;
        } else {
            expect_reject("malformed-fence", argv[1], bad_fence_input, bad_fence_output);
        }
        free(mutation); mutation = NULL;
    }
    if (!setup_failed) {
        if (malformed_schedule(source, source_len, &mutation) < 0 ||
            write_new_file(bad_schedule_input, mutation, source_len) < 0) {
            record_failure("malformed-schedule", "could not create malformed-schedule input");
            setup_failed = 1;
        } else {
            expect_reject("malformed-schedule", argv[1], bad_schedule_input, bad_schedule_output);
        }
        free(mutation); mutation = NULL;
    }
    if (!setup_failed) {
        if (duplicate_anchor(source, source_len, &mutation, &mutation_len) < 0 ||
            write_new_file(duplicate_input, mutation, mutation_len) < 0) {
            record_failure("duplicate-anchor", "could not create duplicate-anchor input");
            setup_failed = 1;
        } else {
            expect_reject("duplicate-anchor", argv[1], duplicate_input, duplicate_output);
        }
        free(mutation); mutation = NULL;
    }

    if (write_new_file(sentinel, sentinel_data, sizeof sentinel_data - 1u) < 0) {
        record_failure("existing-output-preservation", "could not create sentinel output");
    } else {
        rc = run_patcher(argv[1], argv[2], sentinel);
        if (rc == 0) record_failure("existing-output-preservation", "sentinel output path was accepted");
        if (!file_equals(sentinel, sentinel_data, sizeof sentinel_data - 1u))
            record_failure("existing-output-preservation", "sentinel output was changed");
    }

    if (write_new_file(source_copy, source, source_len) < 0) {
        record_failure("alias-preservation", "could not create alias input copy");
    } else {
        rc = run_patcher(argv[1], source_copy, source_copy);
        if (rc == 0) record_failure("alias-preservation", "same-path input/output was accepted");
        if (!file_equals(source_copy, source, source_len))
            record_failure("alias-preservation", "same-path alias input changed");
        if (link(source_copy, hardlink_alias) < 0) {
            record_failure("hardlink-alias-preservation", "could not create hardlink alias");
        } else {
            rc = run_patcher(argv[1], source_copy, hardlink_alias);
            if (rc == 0) record_failure("hardlink-alias-preservation", "hardlink alias was accepted");
            if (!file_equals(source_copy, source, source_len) ||
                !file_equals(hardlink_alias, source, source_len))
                record_failure("hardlink-alias-preservation", "hardlink input bytes changed");
        }
    }

    for (size_t i = 0; i < sizeof cleanup_names / sizeof cleanup_names[0]; i++)
        cleanup_file(directory, cleanup_names[i]);
    if (rmdir(directory) < 0) record_failure("cleanup", "temporary directory is not empty");
    free(source);
    free(patched);
    free(mutation);
    if (failures) {
        fprintf(stderr, "PAIR_BARRIER_TEST_FAIL checks_failed=%d\n", failures);
        return 1;
    }
    puts("PAIR_BARRIER_TEST_PASS generated_positive=1 reject_cases=5 alias_preservation=1");
    return 0;
}
