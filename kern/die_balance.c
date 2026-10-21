#define _GNU_SOURCE
#include "die_balance_common.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *usage =
    "Usage: ruby kern/die_balance.rb --baseline capture.json [--json NEW_REPORT] [candidate.json ...]\n"
    "        --baseline PATH\n"
    "        --json PATH\n"
    "    -h, --help\n";

typedef struct {
    int valid;
    int eligible;
    int rank;
    char *path;
    char *label;
    char *hostname;
    char **failures;
    size_t failure_count;
    json_object *root;
    json_object *manifest;
    json_object *config;
    json_object *input;
    json_object *output;
    int repetitions;
    double objective_ms;
    double worst_spread_ms;
    double mean_decode_tok_s;
    double per_die_max_total_ms[6];
} capture;

typedef struct {
    json_object *input;
    json_object *output;
    double rate;
    double spread;
    double max_total;
    double die_total[6];
    char signature[6][4096];
} run_metrics;

static int fail(char **error, const char *format, ...) {
    if (!error) return 0;
    va_list args;
    va_start(args, format);
    if (vasprintf(error, format, args) < 0) *error = NULL;
    va_end(args);
    return 0;
}

static int text_value(json_object *value, const char *name, char **error) {
    if (value && json_object_get_type(value) == json_type_string &&
        db_text_valid(json_object_get_string(value))) return 1;
    return fail(error, "%s: missing/empty string", name);
}

static int integer_value(json_object *value, const char *name, int positive, long long *out, char **error) {
    long long n;
    if (!db_integer(value, &n)) return fail(error, "%s: not an integer", name);
    if (positive ? n <= 0 : n < 0) return fail(error, "%s: invalid sign", name);
    if (out) *out = n;
    return 1;
}

static int number_value(json_object *value, const char *name, int positive, double *out, char **error) {
    double n;
    if (!db_number(value, &n)) return fail(error, "%s: nonfinite/nonnumeric", name);
    if (positive ? n <= 0 : n < 0) return fail(error, "%s: invalid sign", name);
    if (out) *out = n;
    return 1;
}

static int text_field(json_object *object, const char *key, const char *name, char **error) {
    return text_value(db_get(object, key), name, error);
}

static int integer_array(json_object *value, const char *name, json_object **copy, char **error) {
    if (!value || json_object_get_type(value) != json_type_array || json_object_array_length(value) == 0)
        return fail(error, "%s: missing token IDs", name);
    size_t length = json_object_array_length(value);
    json_object *result = json_object_new_array();
    if (!result) return fail(error, "out of memory");
    for (size_t i = 0; i < length; i++) {
        long long id;
        if (!integer_value(json_object_array_get_idx(value, i), name, 0, &id, error) || id >= 4294967296LL) {
            if (!*error) fail(error, "%s: token outside uint32", name);
            json_object_put(result);
            return 0;
        }
        json_object_array_add(result, json_object_new_int64(id));
    }
    *copy = result;
    return 1;
}

static int parse_float(const char *text, double *out) {
    if (!text || !*text) return 0;
    const char *p = text;
    if (*p == '+' || *p == '-') p++;
    int digits = 0;
    while (*p >= '0' && *p <= '9') { p++; digits++; }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') { p++; digits++; }
    }
    if (!digits) return 0;
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        int exponent = 0;
        while (*p >= '0' && *p <= '9') { p++; exponent++; }
        if (!exponent) return 0;
    }
    if (*p) return 0;
    errno = 0;
    double n = strtod(text, NULL);
    if (errno == ERANGE || !isfinite(n)) return 0;
    *out = n;
    return 1;
}

static int parse_done(const char *raw, size_t output_count, size_t input_count,
                      double decode_ms, char **error) {
    char *copy = strdup(raw ? raw : "");
    if (!copy) return fail(error, "out of memory");
    char *parts[10] = {0};
    size_t count = 0;
    char *save = NULL;
    for (char *part = strtok_r(copy, " \t\r\n", &save); part; part = strtok_r(NULL, " \t\r\n", &save)) {
        if (count < 10) parts[count] = part;
        count++;
    }
    if (count != 8 || strcmp(parts[0], "done")) {
        free(copy);
        return fail(error, "run: malformed done");
    }
    long long fields[8] = {0};
    for (int i = 1; i < 8; i++) {
        if (i == 4 || i == 5) continue;
        char *end = NULL;
        errno = 0;
        fields[i] = strtoll(parts[i], &end, 10);
        if (errno || !end || *end || end == parts[i] || fields[i] < 0) {
            free(copy);
            return fail(error, "run: malformed done");
        }
    }
    double run_ms = 0, first_ms = 0;
    if (!parse_float(parts[4], &run_ms) || !parse_float(parts[5], &first_ms)) {
        free(copy);
        return fail(error, "run: malformed done");
    }
    if ((size_t)fields[1] != output_count || (size_t)fields[3] != input_count || fields[2] > 1) {
        free(copy);
        return fail(error, "run: done counts/stop flag differ");
    }
    if (run_ms <= 0 || first_ms < 0) {
        free(copy);
        return fail(error, "run: invalid sign");
    }
    if (!(run_ms > first_ms) || fabs(run_ms - first_ms - decode_ms) > 0.150001) {
        free(copy);
        return fail(error, "run: raw and HTTP decode times differ");
    }
    free(copy);
    return 1;
}

static int run_validate(json_object *run, int index, const char *raw_config,
                        run_metrics *metrics, char **error) {
    long long actual_index = -1;
    if (!run || json_object_get_type(run) != json_type_object ||
        !integer_value(db_get(run, "index"), "run index", 0, &actual_index, error)) return 0;
    if (actual_index != index) return fail(error, "run %d: invalid/out-of-order index", index);
    json_object *warmup = db_get(run, "warmup");
    if (!warmup || json_object_get_type(warmup) != json_type_boolean ||
        json_object_get_boolean(warmup) != (index == 0)) return fail(error, "run %d: incorrect warmup flag", index);
    const char *config = db_string(run, "config");
    if (!config || strcmp(config, raw_config)) return fail(error, "run %d: runtime config changed", index);
    char name[80];
    json_object *input = NULL, *output = NULL;
    snprintf(name, sizeof(name), "run %d input", index);
    if (!integer_array(db_get(run, "input_tokens"), name, &input, error)) return 0;
    snprintf(name, sizeof(name), "run %d output", index);
    if (!integer_array(db_get(run, "output_tokens"), name, &output, error)) {
        json_object_put(input);
        return 0;
    }
    size_t output_count = json_object_array_length(output);
    size_t input_count = json_object_array_length(input);
    if (output_count < 2 || output_count > 40) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: require 2..40 output IDs", index);
    }
    json_object *response = db_get(run, "response");
    if (!response || json_object_get_type(response) != json_type_object || db_get(response, "error")) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: invalid HTTP response", index);
    }
    json_object *usage = db_get(response, "usage");
    json_object *timings = db_get(response, "timings");
    if (!usage || json_object_get_type(usage) != json_type_object ||
        !timings || json_object_get_type(timings) != json_type_object) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: missing usage/timings", index);
    }
    long long count = 0, predicted_n = 0, intervals = 0;
    double rate = 0, decode_ms = 0;
    if (!integer_value(db_get(usage, "prompt_tokens"), "prompt token count", 0, &count, error) || count < 0 || (size_t)count != input_count) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: prompt count differs from IDs", index);
    }
    if (!integer_value(db_get(usage, "completion_tokens"), "completion token count", 0, &count, error) ||
        (size_t)count != output_count ||
        !integer_value(db_get(timings, "predicted_n"), "predicted_n", 0, &predicted_n, error) ||
        (size_t)predicted_n != output_count) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: output count differs from IDs", index);
    }
    if (!number_value(db_get(timings, "predicted_per_second"), "predicted_per_second", 1, &rate, error) ||
        !number_value(db_get(timings, "decode_ms"), "decode_ms", 1, &decode_ms, error) ||
        !integer_value(db_get(timings, "decode_intervals"), "decode_intervals", 0, &intervals, error) ||
        (size_t)intervals != output_count - 1) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: decode intervals differ", index);
    }
    double low = (output_count - 1) * 1000.0 / (decode_ms + 0.05) - 0.005001;
    double high = decode_ms > 0.05 ? (output_count - 1) * 1000.0 / (decode_ms - 0.05) + 0.005001 : INFINITY;
    if (!(rate >= low && rate <= high)) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: inconsistent decode timing/rate", index);
    }
    const char *done = db_string(run, "done");
    if (!db_text_valid(done) || !parse_done(done, output_count, input_count, decode_ms, error)) {
        json_object_put(input); json_object_put(output);
        if (!*error) fail(error, "run %d: malformed done", index);
        return 0;
    }
    json_object *profiles = db_get(run, "profiles");
    if (!profiles || json_object_get_type(profiles) != json_type_array || json_object_array_length(profiles) != 6) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: require six profiles", index);
    }
    db_profile parsed[6];
    int found[6] = {0};
    double min_actq = INFINITY, max_actq = -INFINITY;
    for (size_t i = 0; i < 6; i++) {
        if (!db_profile_validate(json_object_array_get_idx(profiles, i), &parsed[i], error)) {
            json_object_put(input); json_object_put(output);
            return 0;
        }
        int die = parsed[i].die;
        if (found[die]) {
            json_object_put(input); json_object_put(output);
            return fail(error, "run %d: duplicate/missing die", index);
        }
        found[die] = 1;
        metrics->die_total[die] = parsed[i].total_ms;
        if (parsed[i].actq256_us < min_actq) min_actq = parsed[i].actq256_us;
        if (parsed[i].actq256_us > max_actq) max_actq = parsed[i].actq256_us;
        strcpy(metrics->signature[die], parsed[i].signature);
    }
    for (int die = 0; die < 6; die++) if (!found[die]) {
        json_object_put(input); json_object_put(output);
        return fail(error, "run %d: duplicate/missing die", index);
    }
    metrics->input = input;
    metrics->output = output;
    metrics->rate = rate;
    metrics->spread = (max_actq - min_actq) / 1000.0;
    metrics->max_total = 0;
    for (int die = 0; die < 6; die++) if (metrics->die_total[die] > metrics->max_total) metrics->max_total = metrics->die_total[die];
    return 1;
}

static int manifest_validate(json_object *manifest, char **error) {
    if (!manifest || json_object_get_type(manifest) != json_type_object) return fail(error, "manifest: missing object");
    const char *hashes[] = {"frontend_sha256", "core_exe_sha256", "engine_sha256"};
    for (size_t i = 0; i < 3; i++) {
        const char *hash = db_string(manifest, hashes[i]);
        if (!hash || strlen(hash) != 64) return fail(error, "manifest: invalid %s", hashes[i]);
        for (const char *p = hash; *p; p++) if (!isxdigit((unsigned char)*p)) return fail(error, "manifest: invalid %s", hashes[i]);
    }
    json_object *env = db_get(manifest, "frontend_env");
    if (!env || json_object_get_type(env) != json_type_object) return fail(error, "manifest: malformed frontend_env");
    json_object_object_foreach(env, key, value) {
        if (!key || json_object_get_type(value) != json_type_string) return fail(error, "manifest: malformed frontend_env");
    }
    const char *revision = db_string(manifest, "source_revision");
    if (!revision || strlen(revision) != 40) return fail(error, "manifest: malformed collector revision");
    for (const char *p = revision; *p; p++) if (!isxdigit((unsigned char)*p)) return fail(error, "manifest: malformed collector revision");
    json_object *model = db_get(manifest, "model");
    if (!model || json_object_get_type(model) != json_type_object ||
        strcmp(db_string(model, "identity_kind") ? db_string(model, "identity_kind") : "", "stat"))
        return fail(error, "manifest: missing model stat identity");
    const char *paths[] = {"path", "realpath"};
    for (size_t i = 0; i < 2; i++) {
        const char *path = db_string(model, paths[i]);
        if (!db_text_valid(path)) return fail(error, "model %s: missing/empty string", paths[i]);
        if (path[0] != '/') return fail(error, "model: path must be absolute");
    }
    const char *numeric[] = {"device", "inode", "size", "mtime_ns"};
    for (size_t i = 0; i < 4; i++) {
        long long n;
        if (!integer_value(db_get(model, numeric[i]), "model numeric identity", i != 0, &n, error)) return 0;
    }
    json_object *hardware = db_get(manifest, "hardware");
    if (!hardware || json_object_get_type(hardware) != json_type_object) return fail(error, "manifest: missing hardware");
    if (!text_field(hardware, "identity", "hardware identity: missing/empty string", error) ||
        !text_field(hardware, "pci_link_snapshot", "hardware pci_link_snapshot: missing/empty string", error)) return 0;
    return 1;
}

static int iso8601_valid(const char *text) {
    if (!text || strlen(text) < 20) return 0;
    int y, mo, d, h, mi, s, used = 0;
    if (sscanf(text, "%4d-%2d-%2dT%2d:%2d:%2d%n", &y, &mo, &d, &h, &mi, &s, &used) != 6) return 0;
    if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 60) return 0;
    const char *p = text + used;
    if (*p == '.') {
        p++;
        int digits = 0;
        while (isdigit((unsigned char)*p)) { p++; digits++; }
        if (!digits) return 0;
    }
    if (*p == 'Z') return p[1] == '\0';
    if (*p == '+' || *p == '-') {
        int hours, minutes, offset_used = 0;
        if (sscanf(p + 1, "%2d:%2d%n", &hours, &minutes, &offset_used) != 2 || hours > 23 || minutes > 59) return 0;
        return p[1 + offset_used] == '\0';
    }
    return 0;
}

static void add_failure(capture *item, const char *message) {
    char **next = realloc(item->failures, (item->failure_count + 1) * sizeof(*next));
    if (!next) return;
    item->failures = next;
    item->failures[item->failure_count++] = strdup(message);
}

static char *base_name_copy(const char *path) {
    const char *slash = strrchr(path, '/');
    return strdup(slash ? slash + 1 : path);
}

static int load_capture(const char *path, capture *item) {
    memset(item, 0, sizeof(*item));
    item->path = strdup(path);
    item->label = base_name_copy(path);
    char *error = NULL;
    if (!db_parse_json_file(path, &item->root, &error)) goto invalid;
    if (json_object_get_type(item->root) != json_type_object ||
        strcmp(db_string(item->root, "schema") ? db_string(item->root, "schema") : "", DB_SCHEMA)) {
        error = strdup("unsupported capture schema");
        goto invalid;
    }
    const char *name_fields[] = {"label", "created_at", "hostname"};
    for (size_t i = 0; i < 3; i++) if (!text_field(item->root, name_fields[i], name_fields[i], &error)) goto invalid;
    if (!iso8601_valid(db_string(item->root, "created_at"))) {
        error = strdup("invalid iso8601 timestamp");
        goto invalid;
    }
    if (!db_request_equal(db_get(item->root, "request"))) {
        error = strdup("request differs from exact linkmatrix protocol");
        goto invalid;
    }
    long long warmups = 0;
    if (!integer_value(db_get(item->root, "warmup_requests"), "warmup_requests", 0, &warmups, &error) || warmups != 1) {
        error = strdup("require exactly one warmup");
        goto invalid;
    }
    item->manifest = db_get(item->root, "manifest");
    if (!manifest_validate(item->manifest, &error)) goto invalid;
    const char *raw_config = db_string(item->manifest, "config_before");
    if (!db_text_valid(raw_config) || !db_config_validate(raw_config, &item->config, &error)) goto invalid;
    if (strcmp(db_string(item->manifest, "config_after") ? db_string(item->manifest, "config_after") : "", raw_config)) {
        error = strdup("runtime config changed during capture");
        goto invalid;
    }
    json_object *runs = db_get(item->root, "runs");
    if (!runs || json_object_get_type(runs) != json_type_array || json_object_array_length(runs) < 4) {
        error = strdup("require warmup and >=3 measured requests");
        goto invalid;
    }
    size_t run_count = json_object_array_length(runs);
    run_metrics *parsed = calloc(run_count, sizeof(*parsed));
    if (!parsed) {
        error = strdup("out of memory");
        goto invalid;
    }
    for (size_t i = 0; i < run_count; i++) {
        if (!run_validate(json_object_array_get_idx(runs, i), (int)i, raw_config, &parsed[i], &error)) {
            for (size_t j = 0; j <= i; j++) { if (parsed[j].input) json_object_put(parsed[j].input); if (parsed[j].output) json_object_put(parsed[j].output); }
            free(parsed);
            goto invalid;
        }
    }
    for (size_t i = 1; i < run_count; i++) {
        if (!json_object_equal(parsed[0].input, parsed[i].input) || !json_object_equal(parsed[0].output, parsed[i].output)) {
            error = NULL;
            asprintf(&error, "run %zu: token IDs changed within capture", i);
            for (size_t j = 0; j < run_count; j++) { json_object_put(parsed[j].input); json_object_put(parsed[j].output); }
            free(parsed);
            goto invalid;
        }
        for (int die = 0; die < 6; die++) if (strcmp(parsed[0].signature[die], parsed[i].signature[die])) {
            error = NULL;
            asprintf(&error, "run %zu: phase counts changed within capture", i);
            for (size_t j = 0; j < run_count; j++) { json_object_put(parsed[j].input); json_object_put(parsed[j].output); }
            free(parsed);
            goto invalid;
        }
    }
    item->hostname = strdup(db_string(item->root, "hostname"));
    free(item->label);
    item->label = strdup(db_string(item->root, "label"));
    item->input = json_object_get(parsed[0].input);
    item->output = json_object_get(parsed[0].output);
    item->repetitions = (int)run_count - 1;
    double rate_sum = 0;
    for (size_t i = 1; i < run_count; i++) {
        if (parsed[i].max_total > item->objective_ms) item->objective_ms = parsed[i].max_total;
        if (parsed[i].spread > item->worst_spread_ms) item->worst_spread_ms = parsed[i].spread;
        rate_sum += parsed[i].rate;
        for (int die = 0; die < 6; die++) if (parsed[i].die_total[die] > item->per_die_max_total_ms[die]) item->per_die_max_total_ms[die] = parsed[i].die_total[die];
    }
    item->mean_decode_tok_s = rate_sum / item->repetitions;
    for (size_t i = 0; i < run_count; i++) { json_object_put(parsed[i].input); json_object_put(parsed[i].output); }
    free(parsed);
    item->valid = 1;
    return 1;

invalid:
    if (item->root) { json_object_put(item->root); item->root = NULL; }
    if (item->config) { json_object_put(item->config); item->config = NULL; }
    if (!error) error = strdup("invalid capture");
    add_failure(item, error);
    free(error);
    return 0;
}

static int config_stable_equal(json_object *left, json_object *right) {
    const char *variable[] = {"split", "attn_map", "attn_dies", "cubin"};
    {
        json_object_object_foreach(left, key, value) {
            int skip = 0;
            for (size_t i = 0; i < 4; i++) if (!strcmp(key, variable[i])) skip = 1;
            if (!skip && !json_object_equal(value, db_get(right, key))) return 0;
        }
    }
    {
        json_object_object_foreach(right, key, value) {
            int skip = 0;
            for (size_t i = 0; i < 4; i++) if (!strcmp(key, variable[i])) skip = 1;
            if (!skip && !json_object_equal(value, db_get(left, key))) return 0;
        }
    }
    return 1;
}

static int gate_failures(capture *item, capture *baseline) {
    if (item->worst_spread_ms > 0.5 + 1e-9) add_failure(item, "actq256 worst per-run spread exceeds 0.5 ms");
    if (item->mean_decode_tok_s < 35.9 - 1e-9) add_failure(item, "arithmetic mean decode is below 35.9 tok/s");
    if (!json_object_equal(item->input, baseline->input) || !json_object_equal(item->output, baseline->output))
        add_failure(item, "input/output token IDs differ from baseline");
    if (!config_stable_equal(item->config, baseline->config)) add_failure(item, "runtime differs beyond placement/cubin pathname");
    if (strcmp(item->hostname, baseline->hostname)) add_failure(item, "hostname differs from baseline");
    const char *keys[] = {"frontend_sha256", "frontend_env", "core_exe_sha256", "engine_sha256", "model", "hardware", "source_revision"};
    for (size_t i = 0; i < 7; i++) if (!json_object_equal(db_get(item->manifest, keys[i]), db_get(baseline->manifest, keys[i]))) {
        char *message = NULL;
        asprintf(&message, "manifest %s differs from baseline", keys[i]);
        if (message) { add_failure(item, message); free(message); }
    }
    return 1;
}

static json_object *string_array(char **items, size_t count) {
    json_object *array = json_object_new_array();
    for (size_t i = 0; i < count; i++) json_object_array_add(array, json_object_new_string(items[i]));
    return array;
}

static json_object *public_row(capture *item, const char *role, int candidate) {
    json_object *row = json_object_new_object();
    if (item->valid) {
        json_object_object_add(row, "path", json_object_new_string(item->path));
        json_object_object_add(row, "label", json_object_new_string(item->label));
        json_object_object_add(row, "config", json_object_get(item->config));
        json_object_object_add(row, "repetitions", json_object_new_int(item->repetitions));
        json_object_object_add(row, "objective_ms", json_object_new_double(item->objective_ms));
        json_object_object_add(row, "worst_spread_ms", json_object_new_double(item->worst_spread_ms));
        json_object_object_add(row, "mean_decode_tok_s", json_object_new_double(item->mean_decode_tok_s));
        json_object *per_die = json_object_new_array();
        for (int i = 0; i < 6; i++) json_object_array_add(per_die, json_object_new_double(item->per_die_max_total_ms[i]));
        json_object_object_add(row, "per_die_max_total_ms", per_die);
        json_object_object_add(row, "failures", string_array(item->failures, item->failure_count));
        json_object_object_add(row, "valid", json_object_new_boolean(1));
        if (candidate) {
            json_object_object_add(row, "eligible", json_object_new_boolean(item->eligible));
            if (item->rank) json_object_object_add(row, "rank", json_object_new_int(item->rank));
        }
    } else {
        json_object_object_add(row, "path", json_object_new_string(item->path));
        json_object_object_add(row, "label", json_object_new_string(item->label));
        json_object_object_add(row, "valid", json_object_new_boolean(0));
        json_object_object_add(row, "failures", string_array(item->failures, item->failure_count));
        if (candidate) json_object_object_add(row, "eligible", json_object_new_boolean(0));
    }
    json_object_object_add(row, "role", json_object_new_string(role));
    return row;
}

static int candidate_compare(const void *a, const void *b) {
    const capture *left = *(const capture *const *)a;
    const capture *right = *(const capture *const *)b;
    if (left->objective_ms < right->objective_ms) return -1;
    if (left->objective_ms > right->objective_ms) return 1;
    return strcmp(left->path, right->path);
}

static json_object *make_report(capture *baseline, capture *candidates, size_t count,
                                capture **selected) {
    *selected = NULL;
    for (size_t i = 0; i < count; i++) {
        if (baseline->valid) {
            if (candidates[i].valid) gate_failures(&candidates[i], baseline);
        } else {
            add_failure(&candidates[i], "baseline is invalid");
        }
        candidates[i].eligible = baseline->valid && candidates[i].valid && candidates[i].failure_count == 0;
    }
    capture **ranked = calloc(count ? count : 1, sizeof(*ranked));
    size_t ranked_count = 0;
    for (size_t i = 0; i < count; i++) if (candidates[i].eligible) ranked[ranked_count++] = &candidates[i];
    qsort(ranked, ranked_count, sizeof(*ranked), candidate_compare);
    for (size_t i = 0; i < ranked_count; i++) ranked[i]->rank = (int)i + 1;
    if (ranked_count) *selected = ranked[0];

    json_object *report = json_object_new_object();
    json_object_object_add(report, "schema", json_object_new_string("flash-sass-die-balance-analysis-v1"));
    const char *status = !baseline->valid ? "rejected" : count == 0 ? "pending" : ranked_count ? "accepted" : "rejected";
    json_object_object_add(report, "status", json_object_new_string(status));
    json_object_object_add(report, "objective", json_object_new_string("minimize max_run(max_die(profile header total_ms)); measured requests only"));
    json_object *limits = json_object_new_array();
    json_object_array_add(limits, json_object_new_string(DB_LIMITATION));
    json_object_array_add(limits, json_object_new_string("Model identity uses path/stat metadata, not a complete content hash."));
    json_object_object_add(report, "limits", limits);
    json_object_object_add(report, "baseline", public_row(baseline, "baseline", 0));
    json_object *rows = json_object_new_array();
    for (size_t i = 0; i < count; i++) json_object_array_add(rows, public_row(&candidates[i], "candidate", 1));
    json_object_object_add(report, "candidates", rows);
    json_object *selected_json = NULL;
    if (*selected) {
        selected_json = json_object_new_object();
        json_object_object_add(selected_json, "path", json_object_new_string((*selected)->path));
        json_object_object_add(selected_json, "label", json_object_new_string((*selected)->label));
        json_object_object_add(selected_json, "objective_ms", json_object_new_double((*selected)->objective_ms));
        json_object_object_add(selected_json, "split_shares", json_object_get(db_get((*selected)->config, "split")));
        json_object_object_add(selected_json, "attn_map", json_object_get(db_get((*selected)->config, "attn_map")));
    }
    json_object_object_add(report, "selected", selected_json);
    free(ranked);
    return report;
}

static void print_report(json_object *report) {
    const char *status = db_string(report, "status");
    char upper_status[32] = "UNKNOWN";
    if (status) {
        snprintf(upper_status, sizeof(upper_status), "%s", status);
        for (char *p = upper_status; *p; p++) *p = (char)toupper((unsigned char)*p);
    }
    printf("Measured die-balance analysis: %s\n", upper_status);
    printf("%s\n%s\n", db_string(report, "objective"), DB_LIMITATION);
    printf("%-10s %-24s %5s %12s %12s %12s %s\n", "ROLE", "LABEL", "RUNS", "MAX TOTAL ms", "SPREAD ms", "MEAN tok/s", "RESULT");
    json_object *rows = json_object_new_array();
    json_object_array_add(rows, json_object_get(db_get(report, "baseline")));
    json_object *candidates = db_get(report, "candidates");
    for (size_t i = 0; i < json_object_array_length(candidates); i++) json_object_array_add(rows, json_object_get(json_object_array_get_idx(candidates, i)));
    for (size_t i = 0; i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *role = db_string(row, "role");
        const char *label = db_string(row, "label");
        char upper_role[32] = "UNKNOWN";
        if (role) {
            snprintf(upper_role, sizeof(upper_role), "%s", role);
            for (char *p = upper_role; *p; p++) *p = (char)toupper((unsigned char)*p);
        }
        long long reps = 0;
        json_object *repetitions = db_get(row, "repetitions");
        int has_reps = db_integer(repetitions, &reps);
        double objective = 0, spread = 0, mean = 0;
        int has_objective = db_number(db_get(row, "objective_ms"), &objective);
        int has_spread = db_number(db_get(row, "worst_spread_ms"), &spread);
        int has_mean = db_number(db_get(row, "mean_decode_tok_s"), &mean);
        char objective_text[64], spread_text[64], mean_text[64], reps_text[32];
        if (has_objective) snprintf(objective_text, sizeof(objective_text), "%.6f", objective); else strcpy(objective_text, "-");
        if (has_spread) snprintf(spread_text, sizeof(spread_text), "%.6f", spread); else strcpy(spread_text, "-");
        if (has_mean) snprintf(mean_text, sizeof(mean_text), "%.6f", mean); else strcpy(mean_text, "-");
        if (has_reps) snprintf(reps_text, sizeof(reps_text), "%lld", reps); else strcpy(reps_text, "-");
        const char *result;
        if (!strcmp(role, "baseline")) result = "REFERENCE / NOT SELECTED";
        else if (json_object_get_boolean(db_get(row, "eligible"))) {
            long long rank = 0;
            db_integer(db_get(row, "rank"), &rank);
            static char eligible_result[80];
            snprintf(eligible_result, sizeof(eligible_result), "ELIGIBLE rank=%lld", rank);
            result = eligible_result;
        } else result = "REJECTED";
        printf("%-10s %-24s %5s %12s %12s %12s %s\n", upper_role, label,
               reps_text, objective_text, spread_text, mean_text, result);
        json_object *failures = db_get(row, "failures");
        for (size_t j = 0; j < json_object_array_length(failures); j++)
            printf("  failure: %s\n", json_object_get_string(json_object_array_get_idx(failures, j)));
    }
    json_object *selected = db_get(report, "selected");
    if (selected && json_object_get_type(selected) == json_type_object) {
        printf("Best measured candidate: %s\n", db_string(selected, "path"));
        printf("SPLIT_SHARES=%s\nATTN_MAP=%s\n", db_string(selected, "split_shares"), db_string(selected, "attn_map"));
    } else {
        printf("No measured candidate selected; no acceptance claim.\n");
    }
    json_object_put(rows);
}

static int write_json_exclusive(const char *path, json_object *value) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) return 0;
    char *json = db_json_serialize(value, 1);
    if (!json) { close(fd); return 0; }
    size_t length = strlen(json);
    ssize_t written = write(fd, json, length);
    int ok = written == (ssize_t)length && write(fd, "\n", 1) == 1 && fsync(fd) == 0;
    int saved = errno;
    close(fd);
    free(json);
    errno = saved;
    return ok;
}

static int cli(int argc, char **argv) {
    char *baseline_path = NULL;
    char *json_path = NULL;
    char **candidate_paths = calloc((size_t)argc, sizeof(*candidate_paths));
    size_t candidate_count = 0;
    if (!candidate_paths) { fprintf(stderr, "out of memory\n"); return 2; }
    for (int i = 1; i < argc; i++) {
        const char *argument = argv[i];
        if (!strcmp(argument, "--")) {
            for (i++; i < argc; i++) candidate_paths[candidate_count++] = argv[i];
            break;
        }
        char option_buffer[64];
        const char *option = argument;
        const char *value = NULL;
        const char *equals = strchr(argument, '=');
        if (equals) {
            size_t length = (size_t)(equals - argument);
            if (length >= sizeof(option_buffer)) {
                fprintf(stderr, "invalid option: %s\n", argument); free(candidate_paths); return 2;
            }
            memcpy(option_buffer, argument, length);
            option_buffer[length] = '\0';
            option = option_buffer;
            value = equals + 1;
        }
        if (!strcmp(option, "-h") || !strcmp(option, "--help")) { fputs(usage, stdout); free(candidate_paths); return 0; }
        else if (!strcmp(option, "--baseline") || !strcmp(option, "--json")) {
            int is_baseline = !strcmp(option, "--baseline");
            if (!value) {
                if (i + 1 >= argc || argv[i + 1][0] == '-') {
                    fprintf(stderr, "missing argument: %s\n", option); free(candidate_paths); return 2;
                }
                value = argv[++i];
            }
            if (is_baseline) baseline_path = (char *)value; else json_path = (char *)value;
        } else if (argument[0] == '-') {
            fprintf(stderr, "invalid option: %s\n", argument); free(candidate_paths); return 2;
        } else candidate_paths[candidate_count++] = argv[i];
    }
    if (!baseline_path) {
        fprintf(stderr, "missing argument: --baseline\n"); free(candidate_paths); return 2;
    }
    capture baseline;
    load_capture(baseline_path, &baseline);
    capture *candidates = calloc(candidate_count ? candidate_count : 1, sizeof(*candidates));
    if (!candidates) { fprintf(stderr, "out of memory\n"); free(candidate_paths); return 2; }
    for (size_t i = 0; i < candidate_count; i++) load_capture(candidate_paths[i], &candidates[i]);
    capture *selected = NULL;
    json_object *report = make_report(&baseline, candidates, candidate_count, &selected);
    print_report(report);
    int result = strcmp(db_string(report, "status"), "rejected") == 0 ? 1 : 0;
    if (json_path && !write_json_exclusive(json_path, report)) {
        if (errno == EEXIST) fprintf(stderr, "File exists @ rb_sysopen - %s\n", json_path);
        else if (errno == ENOENT) fprintf(stderr, "No such file or directory @ rb_sysopen - %s\n", json_path);
        else fprintf(stderr, "%s\n", strerror(errno));
        result = 2;
    }
    json_object_put(report);
    for (size_t i = 0; i < candidate_count; i++) {
        if (candidates[i].root) json_object_put(candidates[i].root);
        if (candidates[i].config) json_object_put(candidates[i].config);
        if (candidates[i].input) json_object_put(candidates[i].input);
        if (candidates[i].output) json_object_put(candidates[i].output);
        for (size_t j = 0; j < candidates[i].failure_count; j++) free(candidates[i].failures[j]);
        free(candidates[i].failures); free(candidates[i].path); free(candidates[i].label); free(candidates[i].hostname);
    }
    if (baseline.root) json_object_put(baseline.root);
    if (baseline.config) json_object_put(baseline.config);
    if (baseline.input) json_object_put(baseline.input);
    if (baseline.output) json_object_put(baseline.output);
    for (size_t j = 0; j < baseline.failure_count; j++) free(baseline.failures[j]);
    free(baseline.failures); free(baseline.path); free(baseline.label); free(baseline.hostname);
    free(candidates); free(candidate_paths);
    return result;
}

int main(int argc, char **argv) {
    return cli(argc, argv);
}
