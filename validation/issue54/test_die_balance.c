#define _GNU_SOURCE
#include "fixtures.h"
#include "../../kern/die_balance_common.h"

#include <fcntl.h>
#include <ftw.h>
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static char directory[256];
static char analyzer[512];
static int checks;
static int failures;

static void expect(int condition, const char *name) {
    checks++;
    if (!condition) {
        failures++;
        fprintf(stderr, "FAIL: %s\n", name);
    }
}

static void path_for(char *output, size_t size, const char *name) {
    snprintf(output, size, "%s/%s", directory, name);
}

static int save_capture(const char *name, json_object *capture) {
    char path[512];
    path_for(path, sizeof(path), name);
    return fixture_write(path, capture);
}

static int invoke(int argc, char **arguments, char **output) {
    (void)argc;
    int pipes[2];
    if (pipe(pipes) != 0) return -1;
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        close(pipes[0]);
        dup2(pipes[1], STDOUT_FILENO);
        int nullfd = open("/dev/null", O_WRONLY);
        if (nullfd >= 0) dup2(nullfd, STDERR_FILENO);
        close(pipes[1]);
        execv(analyzer, arguments);
        _exit(127);
    }
    close(pipes[1]);
    size_t used = 0, capacity = 4096;
    char *buffer = malloc(capacity);
    for (;;) {
        if (used + 2048 + 1 > capacity) {
            capacity *= 2;
            buffer = realloc(buffer, capacity);
        }
        ssize_t got = read(pipes[0], buffer + used, capacity - used - 1);
        if (got <= 0) break;
        used += (size_t)got;
    }
    close(pipes[0]);
    buffer[used] = '\0';
    int status = 0;
    waitpid(child, &status, 0);
    *output = buffer;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

static int remove_entry(const char *path, const struct stat *status, int type, struct FTW *walk) {
    (void)status;
    (void)type;
    (void)walk;
    return remove(path);
}

static int run_paths(const char *baseline, const char *candidate, char **output) {
    char baseline_option[] = "--baseline";
    char *arguments[7];
    int count = 0;
    arguments[count++] = analyzer;
    arguments[count++] = baseline_option;
    arguments[count++] = (char *)baseline;
    if (candidate) arguments[count++] = (char *)candidate;
    arguments[count] = NULL;
    return invoke(count, arguments, output);
}

static void accepted_and_rejected_boundaries(void) {
    json_object *good = fixture_capture(35.9, 0.5, 28.0, "112,97,81,116,97,97");
    expect(save_capture("base.json", good), "write inclusive baseline");
    expect(save_capture("good.json", good), "write inclusive candidate");
    char baseline[512], candidate[512], *output = NULL;
    path_for(baseline, sizeof(baseline), "base.json");
    path_for(candidate, sizeof(candidate), "good.json");
    expect(run_paths(baseline, candidate, &output) == 0, "inclusive rate and spread are accepted");
    expect(output && strstr(output, "Measured die-balance analysis: ACCEPTED"), "accepted report status");
    expect(output && strstr(output, "Best measured candidate:"), "accepted candidate is selected");
    free(output);
    json_object_put(good);

    json_object *wide = fixture_capture(35.9, 0.5001, 28.0, "112,97,81,116,97,97");
    expect(save_capture("wide.json", wide), "write over-spread candidate");
    path_for(candidate, sizeof(candidate), "wide.json");
    output = NULL;
    expect(run_paths(baseline, candidate, &output) == 1, "spread over boundary rejects candidate");
    expect(output && strstr(output, "actq256 worst per-run spread exceeds 0.5 ms"), "spread rejection reason");
    free(output);
    json_object_put(wide);

    json_object *slow = fixture_capture(35.89, 0.5, 28.0, "112,97,81,116,97,97");
    expect(save_capture("slow.json", slow), "write low-rate candidate");
    path_for(candidate, sizeof(candidate), "slow.json");
    output = NULL;
    expect(run_paths(baseline, candidate, &output) == 1, "rate below boundary rejects candidate");
    expect(output && strstr(output, "arithmetic mean decode is below 35.9 tok/s"), "rate rejection reason");
    free(output);
    json_object_put(slow);
}

static void runtime_and_identity_gates(void) {
    char baseline[512], candidate[512], *output = NULL;
    path_for(baseline, sizeof(baseline), "base.json");
    json_object *candidate_data = fixture_capture(35.9, 0.5, 27.0, "100,100,100,100,100,100");
    expect(save_capture("placement.json", candidate_data), "write placement candidate");
    path_for(candidate, sizeof(candidate), "placement.json");
    expect(run_paths(baseline, candidate, &output) == 0, "placement fields may differ");
    expect(output && strstr(output, "Best measured candidate:"), "minimax candidate selected");
    free(output);
    json_object_put(candidate_data);

    candidate_data = fixture_capture(35.9, 0.5, 27.0, "100,100,100,100,100,100");
    json_object *runs = json_object_object_get(candidate_data, "runs");
    json_object *run = json_object_array_get_idx(runs, 2);
    json_object *ids = json_object_object_get(run, "output_tokens");
    json_object_array_put_idx(ids, 0, json_object_new_int(999));
    expect(save_capture("tokens.json", candidate_data), "write changed-token candidate");
    path_for(candidate, sizeof(candidate), "tokens.json");
    output = NULL;
    expect(run_paths(baseline, candidate, &output) == 1, "token mismatch rejects candidate");
    expect(output && strstr(output, "token IDs changed within capture"), "within-capture token check");
    free(output);
    json_object_put(candidate_data);

    candidate_data = fixture_capture(35.9, 0.5, 27.0, "100,100,100,100,100,100");
    json_object *manifest = json_object_object_get(candidate_data, "manifest");
    json_object_object_add(manifest, "engine_sha256", json_object_new_string("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"));
    expect(save_capture("identity.json", candidate_data), "write different-runtime candidate");
    path_for(candidate, sizeof(candidate), "identity.json");
    output = NULL;
    expect(run_paths(baseline, candidate, &output) == 1, "runtime identity mismatch rejects candidate");
    expect(output && strstr(output, "manifest engine_sha256 differs from baseline"), "manifest identity rejection");
    free(output);
    json_object_put(candidate_data);

    candidate_data = fixture_capture(35.9, 0.5, 27.0, "100,100,100,100,100,100");
    json_object *run_one = json_object_array_get_idx(json_object_object_get(candidate_data, "runs"), 1);
    json_object_object_add(run_one, "config", json_object_new_string("config changed"));
    expect(save_capture("changed-config.json", candidate_data), "write changed-config candidate");
    path_for(candidate, sizeof(candidate), "changed-config.json");
    output = NULL;
    expect(run_paths(baseline, candidate, &output) == 1, "configuration change inside capture rejects");
    expect(output && strstr(output, "runtime config changed"), "configuration rejection details");
    free(output);
    json_object_put(candidate_data);
}

static void pending_duplicate_and_cli_contract(void) {
    char baseline[512], *output = NULL;
    path_for(baseline, sizeof(baseline), "base.json");
    char *arguments[] = {analyzer, "--baseline", baseline, NULL};
    expect(invoke(3, arguments, &output) == 0, "baseline-only analysis succeeds");
    expect(output && strstr(output, "Measured die-balance analysis: PENDING"), "baseline-only stays pending");
    expect(output && strstr(output, "no acceptance claim"), "baseline-only makes no acceptance claim");
    expect(output && !strstr(output, "SPLIT_SHARES="), "baseline-only does not publish placement");
    free(output);

    char duplicate[512];
    path_for(duplicate, sizeof(duplicate), "duplicate.json");
    FILE *file = fopen(duplicate, "wb");
    fputs("{\"schema\":\"flash-sass-die-balance-v1\",\"schema\":\"bad\"}", file);
    fclose(file);
    output = NULL;
    expect(run_paths(duplicate, NULL, &output) == 1, "duplicate JSON keys reject baseline");
    expect(output && strstr(output, "duplicate JSON key"), "duplicate-key failure is reported");
    free(output);

    char report[512];
    path_for(report, sizeof(report), "report.json");
    char *json_arguments[] = {analyzer, "--baseline", baseline, "--json", report, NULL};
    output = NULL;
    expect(invoke(5, json_arguments, &output) == 0, "json report succeeds");
    json_object *json = NULL;
    char *error = NULL;
    expect(db_parse_json_file(report, &json, &error), "json report is valid JSON");
    expect(json && !strcmp(db_string(json, "status") ? db_string(json, "status") : "", "pending"), "json report records pending");
    if (json) json_object_put(json);
    free(error);
    free(output);

    file = fopen(report, "wb");
    fputs("preserve", file);
    fclose(file);
    json_arguments[5] = NULL;
    output = NULL;
    expect(invoke(5, json_arguments, &output) == 2, "existing json report path is rejected");
    file = fopen(report, "rb");
    char existing[16] = {0};
    fread(existing, 1, sizeof(existing) - 1, file);
    fclose(file);
    expect(!strcmp(existing, "preserve"), "existing report is preserved");
    free(output);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--emit-fixture")) {
        json_object *fixture = fixture_capture(35.9, 0.5, 28.0, "112,97,81,116,97,97");
        char *text = db_json_serialize(fixture, 0);
        int ok = text != NULL;
        if (ok) { fputs(text, stdout); free(text); }
        json_object_put(fixture);
        return ok ? 0 : 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: test_die_balance ANALYZER\n");
        return 2;
    }
    snprintf(analyzer, sizeof(analyzer), "%s", argv[1]);
    snprintf(directory, sizeof(directory), "/tmp/die-balance-test-XXXXXX");
    if (!mkdtemp(directory)) {
        perror("mkdtemp");
        return 2;
    }
    accepted_and_rejected_boundaries();
    runtime_and_identity_gates();
    pending_duplicate_and_cli_contract();
    printf("die_balance: %d checks, %d failures\n", checks, failures);
    nftw(directory, remove_entry, 16, FTW_DEPTH | FTW_PHYS);
    return failures ? 1 : 0;
}
