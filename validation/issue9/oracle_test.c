#define _POSIX_C_SOURCE 200809L
#include "../../kern/model_oracle.h"

#include <errno.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int digest_bytes(const void *bytes, size_t byte_count, uint8_t digest[MODEL_ORACLE_SHA256_BYTES]) {
    unsigned int digest_len = 0;
    return EVP_Digest(bytes, byte_count, digest, &digest_len, EVP_sha256(), NULL) == 1 &&
           digest_len == MODEL_ORACLE_SHA256_BYTES;
}

static int compare_case(const char *name, const model_oracle_record_t *record,
                        const model_oracle_observation_t *observations, size_t observation_count,
                        int expected_result, const char *expected_json) {
    FILE *result = tmpfile();
    char error[256] = {0};
    char output[256] = {0};
    int actual, ok = 1;
    if (!result) {
        fprintf(stderr, "%s: tmpfile failed\n", name);
        return -1;
    }
    actual = model_oracle_compare_record(record, observations, observation_count,
                                         result, error, sizeof error);
    if (fflush(result) == EOF || fseek(result, 0, SEEK_SET) != 0) ok = 0;
    size_t got = fread(output, 1, sizeof output - 1, result);
    output[got] = '\0';
    if (expected_result < 0) {
        if (actual >= 0 || strstr(output, "\"scalar_match\":true")) ok = 0;
    } else {
        if (actual != expected_result || !strstr(output, expected_json)) ok = 0;
    }
    if (fclose(result) != 0) ok = 0;
    if (ok) printf("PASS %s\n", name);
    else fprintf(stderr, "FAIL %s (result=%d, error=%s, output=%s)\n",
                 name, actual, error, output);
    return ok ? 0 : -1;
}

static void initialize_item(model_oracle_item_t *item, const char *label,
                            uint32_t value_type, uint64_t element_count,
                            uint64_t byte_count, const uint8_t *raw_bytes,
                            const uint8_t *digest) {
    memset(item, 0, sizeof *item);
    snprintf(item->label, sizeof item->label, "%s", label);
    item->layer = 0;
    item->logical_die = 0;
    item->physical_die = 3;
    item->value_type = value_type;
    item->rank = 1;
    item->dimensions[0] = element_count;
    item->byte_count = byte_count;
    memcpy(item->sha256, digest, MODEL_ORACLE_SHA256_BYTES);
    item->raw_bytes = raw_bytes;
    item->raw_byte_count = raw_bytes ? byte_count : 0;
}

static void initialize_observation(model_oracle_observation_t *observation,
                                   const char *label, uint32_t value_type,
                                   uint64_t element_count, const void *bytes,
                                   uint64_t byte_count) {
    memset(observation, 0, sizeof *observation);
    observation->label = label;
    observation->value_type = value_type;
    observation->rank = 1;
    observation->dimensions[0] = element_count;
    observation->bytes = bytes;
    observation->byte_count = byte_count;
}

static int cli_lock_checks(const char *binary) {
    char directory[] = "/tmp/flash-oracle-cli.XXXXXX";
    char records[128], ids[128], checkpoint[128], receipt[128], error_path[128], missing[128];
    const char sentinel[] = "uncommitted bytes must be preserved";
    int failed = 0, record_fd = -1;
    FILE *id_file = NULL;
    if (!mkdtemp(directory)) { perror("oracle CLI scratch"); return -1; }
    snprintf(records, sizeof records, "%s/records", directory);
    snprintf(ids, sizeof ids, "%s/ids", directory);
    snprintf(checkpoint, sizeof checkpoint, "%s/checkpoint", directory);
    snprintf(receipt, sizeof receipt, "%s/receipt", directory);
    snprintf(error_path, sizeof error_path, "%s/stderr", directory);
    snprintf(missing, sizeof missing, "%s/missing-input", directory);
    record_fd = open(records, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    id_file = fopen(ids, "w");
    if (record_fd < 0 || !id_file || write(record_fd, sentinel, sizeof sentinel) != sizeof sentinel) {
        perror("oracle CLI fixtures"); failed = -1; goto done;
    }
    if (fputs("10\n", id_file) == EOF || fclose(id_file)) {
        id_file = NULL; failed = -1; goto done;
    }
    id_file = NULL;
    struct flock lock = { .l_type = F_WRLCK, .l_whence = SEEK_SET };
    if (fcntl(record_fd, F_SETLK, &lock) < 0) { perror("oracle CLI parent lock"); failed = -1; goto done; }
    for (int stage = 2; stage >= 0; stage--) {
        int locked = stage != 0;
        lock.l_type = stage == 2 ? F_WRLCK : stage == 1 ? F_RDLCK : F_UNLCK;
        if (fcntl(record_fd, F_SETLK, &lock) < 0) { failed = -1; break; }
        pid_t child = fork();
        if (child < 0) { perror("oracle CLI fork"); failed = -1; break; }
        if (!child) {
            int err = open(error_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
            int out = open("/dev/null", O_WRONLY);
            if (err < 0 || out < 0 || dup2(err, STDERR_FILENO) < 0 || dup2(out, STDOUT_FILENO) < 0) _Exit(125);
            close(err); close(out);
            char *arguments[] = {(char *)binary, directory, missing, "cli-test", directory,
                missing, missing, missing, missing, missing, "192", ids, "1", "0,1,2,3,4,5",
                records, checkpoint, receipt, "1", NULL};
            execv(binary, arguments);
            _Exit(127);
        }
        int status = 0, finished = 0;
        struct timespec interval = { .tv_nsec = 10000000 };
        for (int attempt = 0; attempt < 500; attempt++) {
            pid_t result = waitpid(child, &status, WNOHANG);
            if (result == child) { finished = 1; break; }
            if (result < 0 && errno != EINTR) break;
            nanosleep(&interval, NULL);
        }
        if (!finished) {
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        char error[1024] = {0}, actual[sizeof sentinel + 1] = {0};
        FILE *err_file = fopen(error_path, "r");
        if (err_file) { (void)fread(error, 1, sizeof error - 1, err_file); fclose(err_file); }
        ssize_t bytes = pread(record_fd, actual, sizeof actual, 0);
        int ok = finished && WIFEXITED(status) && WEXITSTATUS(status) == 1 &&
            bytes == sizeof sentinel && !memcmp(actual, sentinel, sizeof sentinel);
        if (locked) ok = ok && strstr(error, "record stream is already in use") && access(receipt, F_OK) != 0;
        else ok = ok && strstr(error, "input verification failed");
        const char *name = stage == 2 ? "CLI excludes concurrent writers before input verification" :
            stage == 1 ? "shared reader excludes CLI writer before input verification" :
            "failed CLI verification preserves existing record bytes";
        if (ok) printf("PASS %s\n", name);
        else { fprintf(stderr, "FAIL %s: status=%d error=%s\n", name, status, error); failed = -1; }
    }
done:
    if (id_file) fclose(id_file);
    if (record_fd >= 0) close(record_fd);
    unlink(records); unlink(ids); unlink(checkpoint); unlink(receipt); unlink(error_path);
    rmdir(directory);
    return failed;
}

int main(int argc, char **argv) {
    static const uint8_t selected_bytes[8] = {1, 0, 0, 0, 2, 0, 0, 0};
    static const uint8_t scalar_bytes[4] = {0, 0, 128, 63};
    static const uint8_t changed_scalar_bytes[4] = {0, 0, 129, 63};
    model_oracle_item_t items[2];
    model_oracle_observation_t observations[3];
    model_oracle_record_t record;
    uint8_t selected_digest[MODEL_ORACLE_SHA256_BYTES];
    uint8_t scalar_digest[MODEL_ORACLE_SHA256_BYTES];
    int failed = 0;
    if (!digest_bytes(selected_bytes, sizeof selected_bytes, selected_digest) ||
        !digest_bytes(scalar_bytes, sizeof scalar_bytes, scalar_digest)) {
        fprintf(stderr, "failed to create test digests\n");
        return 1;
    }
    initialize_item(&items[0], "b0.selected_ids", MODEL_ORACLE_U32, 2,
                    sizeof selected_bytes, selected_bytes, selected_digest);
    initialize_item(&items[1], "b0.S.d0", MODEL_ORACLE_F32, 1,
                    sizeof scalar_bytes, NULL, scalar_digest);
    memset(&record, 0, sizeof record);
    record.position = 4;
    record.visible_length = 5;
    record.item_count = 2;
    record.items = items;

    /* Observation order is reversed to verify label-based matching. */
    initialize_observation(&observations[0], "b0.S.d0", MODEL_ORACLE_F32, 1,
                           scalar_bytes, sizeof scalar_bytes);
    initialize_observation(&observations[1], "b0.selected_ids", MODEL_ORACLE_U32, 2,
                           selected_bytes, sizeof selected_bytes);
    failed += compare_case("valid raw and digest comparison", &record,
                           observations, 2, 0, "\"scalar_match\":true");

    observations[0].bytes = changed_scalar_bytes;
    failed += compare_case("one-bit mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].bytes = scalar_bytes;

    observations[0].byte_count = 3;
    failed += compare_case("size mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].byte_count = sizeof scalar_bytes;

    observations[0].value_type = MODEL_ORACLE_U32;
    failed += compare_case("type mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].value_type = MODEL_ORACLE_F32;

    observations[0].dimensions[0] = 2;
    failed += compare_case("shape mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].dimensions[0] = 1;

    failed += compare_case("missing label", &record, &observations[1], 1,
                           1, "\"scalar_match\":false");

    model_oracle_observation_t duplicate[2] = {observations[0], observations[0]};
    failed += compare_case("duplicate label", &record, duplicate, 2,
                           1, "\"scalar_match\":false");

    initialize_observation(&observations[2], "extra.label", MODEL_ORACLE_F32, 1,
                           scalar_bytes, sizeof scalar_bytes);
    failed += compare_case("extra label", &record, observations, 3,
                           1, "\"scalar_match\":false");

    model_oracle_record_t empty = record;
    empty.item_count = 0;
    empty.items = NULL;
    failed += compare_case("empty record rejected", &empty, NULL, 0, -1, NULL);

    /* The producer once multiplied the ring's byte count twice by sizeof(float).
     * Reject that malformed expected record before touching its payload. */
    model_oracle_item_t ring = items[1];
    snprintf(ring.label, sizeof ring.label, "b3.ring");
    ring.layer = 3;
    ring.dimensions[0] = 128;
    ring.byte_count = 4 * 128 * sizeof(float);
    model_oracle_record_t malformed = record;
    malformed.item_count = 1;
    malformed.items = &ring;
    failed += compare_case("oversized ring byte count rejected", &malformed, NULL, 0, -1, NULL);

    if (argc == 2) failed += cli_lock_checks(argv[1]);
    else if (argc != 1) { fprintf(stderr, "usage: %s [MODEL_ORACLE_BINARY]\n", argv[0]); return 2; }

    if (failed) return 1;
    puts("oracle API checks passed");
    return 0;
}
