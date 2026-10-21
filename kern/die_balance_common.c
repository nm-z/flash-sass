#define _GNU_SOURCE
#include "die_balance_common.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
} db_json_buffer;

static int json_append(db_json_buffer *buffer, const char *text, size_t length) {
    if (buffer->length + length + 1 > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 256;
        while (capacity < buffer->length + length + 1) capacity *= 2;
        char *next = realloc(buffer->text, capacity);
        if (!next) return 0;
        buffer->text = next;
        buffer->capacity = capacity;
    }
    memcpy(buffer->text + buffer->length, text, length);
    buffer->length += length;
    buffer->text[buffer->length] = '\0';
    return 1;
}

static int json_indent(db_json_buffer *buffer, int level) {
    for (int i = 0; i < level * 2; i++) if (!json_append(buffer, " ", 1)) return 0;
    return 1;
}

static int json_string(db_json_buffer *buffer, const char *text, size_t length) {
    static const char hex[] = "0123456789abcdef";
    if (!json_append(buffer, "\"", 1)) return 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char byte = (unsigned char)text[i];
        const char *escape = NULL;
        switch (byte) {
            case '\"': escape = "\\\""; break;
            case '\\': escape = "\\\\"; break;
            case '\b': escape = "\\b"; break;
            case '\f': escape = "\\f"; break;
            case '\n': escape = "\\n"; break;
            case '\r': escape = "\\r"; break;
            case '\t': escape = "\\t"; break;
            default: break;
        }
        if (escape) {
            if (!json_append(buffer, escape, 2)) return 0;
        } else if (byte < 0x20) {
            char encoded[6] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 0xf]};
            if (!json_append(buffer, encoded, sizeof(encoded))) return 0;
        } else if (!json_append(buffer, (const char *)&text[i], 1)) return 0;
    }
    return json_append(buffer, "\"", 1);
}

static int json_double(db_json_buffer *buffer, double value) {
    char candidate[128], best[128] = {0};
    for (int precision = 1; precision <= 17; precision++) {
        snprintf(candidate, sizeof(candidate), "%.*g", precision, value);
        char *end = NULL;
        double parsed = strtod(candidate, &end);
        if (end && *end == '\0' && parsed == value) {
            strcpy(best, candidate);
            break;
        }
    }
    if (!*best) snprintf(best, sizeof(best), "%.17g", value);
    if (!strchr(best, '.') && !strchr(best, 'e') && !strchr(best, 'E')) {
        size_t length = strlen(best);
        if (length + 2 < sizeof(best)) { best[length] = '.'; best[length + 1] = '0'; best[length + 2] = '\0'; }
    }
    return json_append(buffer, best, strlen(best));
}

static int json_value(db_json_buffer *buffer, json_object *value, int level, int pretty) {
    if (!value) return json_append(buffer, "null", 4);
    enum json_type type = json_object_get_type(value);
    if (type == json_type_null) return json_append(buffer, "null", 4);
    if (type == json_type_boolean) {
        const char *text = json_object_get_boolean(value) ? "true" : "false";
        return json_append(buffer, text, strlen(text));
    }
    if (type == json_type_int) {
        char number[64];
        snprintf(number, sizeof(number), "%lld", (long long)json_object_get_int64(value));
        return json_append(buffer, number, strlen(number));
    }
    if (type == json_type_double) return json_double(buffer, json_object_get_double(value));
    if (type == json_type_string)
        return json_string(buffer, json_object_get_string(value), (size_t)json_object_get_string_len(value));
    if (type == json_type_array) {
        size_t count = json_object_array_length(value);
        if (!json_append(buffer, "[", 1)) return 0;
        if (count) {
            if (pretty && !json_append(buffer, "\n", 1)) return 0;
            for (size_t i = 0; i < count; i++) {
                if (pretty && !json_indent(buffer, level + 1)) return 0;
                if (!json_value(buffer, json_object_array_get_idx(value, i), level + 1, pretty)) return 0;
                if (i + 1 < count && !json_append(buffer, ",", 1)) return 0;
                if (pretty && !json_append(buffer, "\n", 1)) return 0;
            }
            if (pretty && !json_indent(buffer, level)) return 0;
        }
        return json_append(buffer, "]", 1);
    }
    if (type == json_type_object) {
        size_t count = json_object_object_length(value);
        if (!json_append(buffer, "{", 1)) return 0;
        if (count) {
            if (pretty && !json_append(buffer, "\n", 1)) return 0;
            size_t index = 0;
            json_object_object_foreach(value, key, member) {
                if (pretty && !json_indent(buffer, level + 1)) return 0;
                if (!json_string(buffer, key, strlen(key))) return 0;
                if (!json_append(buffer, pretty ? ": " : ":", pretty ? 2 : 1)) return 0;
                if (!json_value(buffer, member, level + 1, pretty)) return 0;
                if (++index < count && !json_append(buffer, ",", 1)) return 0;
                if (pretty && !json_append(buffer, "\n", 1)) return 0;
            }
            if (pretty && !json_indent(buffer, level)) return 0;
        }
        return json_append(buffer, "}", 1);
    }
    return 0;
}

char *db_json_serialize(json_object *value, int pretty) {
    db_json_buffer buffer = {0};
    if (!json_value(&buffer, value, 0, pretty)) {
        free(buffer.text);
        return NULL;
    }
    if (!buffer.text) return strdup("");
    return buffer.text;
}

static const char *required_config[] = {
    "rev", "kv_host", "ple_disk", "maxt", "layers", "nblk", "dies",
    "attn_dies", "split", "attn_map", "img_rows", "two_position",
    "prefill_chunk", "cubin", "cubin_fnv"
};

int db_fail(char **error, const char *format, ...) {
    if (error) {
        va_list args;
        va_start(args, format);
        if (vasprintf(error, format, args) < 0) *error = NULL;
        va_end(args);
    }
    return 0;
}

int db_text_valid(const char *value) {
    if (!value) return 0;
    while (*value && isspace((unsigned char)*value)) value++;
    if (!*value) return 0;
    while (*value && isspace((unsigned char)*value)) value++;
    return *value == '\0' || value != NULL;
}

json_object *db_get(json_object *object, const char *key) {
    json_object *value = NULL;
    return object && json_object_get_type(object) == json_type_object &&
                   json_object_object_get_ex(object, key, &value)
               ? value
               : NULL;
}

const char *db_string(json_object *object, const char *key) {
    json_object *value = db_get(object, key);
    return value && json_object_get_type(value) == json_type_string
               ? json_object_get_string(value)
               : NULL;
}

int db_number(json_object *value, double *out) {
    if (!value) return 0;
    enum json_type type = json_object_get_type(value);
    if (type != json_type_int && type != json_type_double) return 0;
    double n = json_object_get_double(value);
    if (!isfinite(n)) return 0;
    if (out) *out = n;
    return 1;
}

int db_integer(json_object *value, long long *out) {
    if (!value || json_object_get_type(value) != json_type_int) return 0;
    if (out) *out = json_object_get_int64(value);
    return 1;
}

static const char *skip_space(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *scan_string(const char *p) {
    if (*p != '"') return NULL;
    p++;
    while (*p) {
        if (*p == '"') return p + 1;
        if ((unsigned char)*p < 0x20) return NULL;
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
            if (*p == 'u') {
                for (int i = 0; i < 4; i++) {
                    p++;
                    if (!isxdigit((unsigned char)*p)) return NULL;
                }
            } else if (!strchr("\"\\/bfnrt", *p)) {
                return NULL;
            }
        }
        p++;
    }
    return NULL;
}

static json_object *decode_json_string(const char *start, const char *end) {
    size_t length = (size_t)(end - start);
    char *text = strndup(start, length);
    if (!text) return NULL;
    json_tokener *tokener = json_tokener_new();
    json_object *value = json_tokener_parse_ex(tokener, text, (int)length);
    enum json_tokener_error code = json_tokener_get_error(tokener);
    json_tokener_free(tokener);
    free(text);
    if (code != json_tokener_success || !value || json_object_get_type(value) != json_type_string) {
        if (value) json_object_put(value);
        return NULL;
    }
    return value;
}

static int scan_value(const char **cursor, unsigned depth, char **error);

static int scan_object(const char **cursor, unsigned depth, char **error) {
    const char *p = skip_space(*cursor + 1);
    char **keys = NULL;
    size_t *lengths = NULL;
    size_t count = 0;
    if (*p == '}') {
        *cursor = p + 1;
        return 1;
    }
    for (;;) {
        const char *start = p;
        const char *end = scan_string(p);
        if (!end) goto malformed;
        json_object *key = decode_json_string(start, end);
        if (!key) goto malformed;
        const char *decoded = json_object_get_string(key);
        size_t decoded_len = (size_t)json_object_get_string_len(key);
        for (size_t i = 0; i < count; i++) {
            if (lengths[i] == decoded_len && memcmp(keys[i], decoded, decoded_len) == 0) {
                db_fail(error, "duplicate JSON key \"%.*s\"", (int)decoded_len, decoded);
                json_object_put(key);
                goto done_fail;
            }
        }
        char **new_keys = realloc(keys, (count + 1) * sizeof(*keys));
        size_t *new_lengths = realloc(lengths, (count + 1) * sizeof(*lengths));
        if (!new_keys || !new_lengths) {
            free(new_keys);
            free(new_lengths);
            json_object_put(key);
            db_fail(error, "out of memory while checking JSON keys");
            goto done_fail;
        }
        keys = new_keys;
        lengths = new_lengths;
        keys[count] = strndup(decoded, decoded_len);
        lengths[count] = decoded_len;
        count++;
        json_object_put(key);
        p = skip_space(end);
        if (*p++ != ':') goto malformed;
        p = skip_space(p);
        if (!scan_value(&p, depth + 1, error)) goto done_fail;
        p = skip_space(p);
        if (*p == '}') {
            *cursor = p + 1;
            for (size_t i = 0; i < count; i++) free(keys[i]);
            free(keys);
            free(lengths);
            return 1;
        }
        if (*p++ != ',') goto malformed;
        p = skip_space(p);
    }

malformed:
    db_fail(error, "malformed JSON object");
done_fail:
    for (size_t i = 0; i < count; i++) free(keys[i]);
    free(keys);
    free(lengths);
    return 0;
}

static int scan_array(const char **cursor, unsigned depth, char **error) {
    const char *p = skip_space(*cursor + 1);
    if (*p == ']') {
        *cursor = p + 1;
        return 1;
    }
    for (;;) {
        if (!scan_value(&p, depth + 1, error)) return 0;
        p = skip_space(p);
        if (*p == ']') {
            *cursor = p + 1;
            return 1;
        }
        if (*p++ != ',') return db_fail(error, "malformed JSON array");
        p = skip_space(p);
    }
}

static int scan_value(const char **cursor, unsigned depth, char **error) {
    if (depth > 256) return db_fail(error, "JSON nesting exceeds limit");
    const char *p = skip_space(*cursor);
    if (*p == '{') {
        *cursor = p;
        return scan_object(cursor, depth, error);
    }
    if (*p == '[') {
        *cursor = p;
        return scan_array(cursor, depth, error);
    }
    if (*p == '"') {
        const char *end = scan_string(p);
        if (!end) return db_fail(error, "malformed JSON string");
        *cursor = end;
        return 1;
    }
    while (*p && *p != ',' && *p != ']' && *p != '}' && !isspace((unsigned char)*p)) p++;
    if (p == *cursor) return db_fail(error, "malformed JSON value");
    *cursor = p;
    return 1;
}

int db_parse_json_file(const char *path, json_object **out, char **error) {
    FILE *file = fopen(path, "rb");
    if (!file) return db_fail(error, "%s", strerror(errno));
    if (fseek(file, 0, SEEK_END) != 0) {
        int saved = errno;
        fclose(file);
        return db_fail(error, "%s", strerror(saved));
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        int saved = errno;
        fclose(file);
        return db_fail(error, "%s", strerror(saved));
    }
    char *text = malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        return db_fail(error, "out of memory");
    }
    size_t got = fread(text, 1, (size_t)length, file);
    int read_error = ferror(file);
    fclose(file);
    if (read_error || got != (size_t)length) {
        free(text);
        return db_fail(error, "cannot read JSON file");
    }
    text[got] = '\0';
    const char *cursor = text;
    if (!scan_value(&cursor, 0, error) || *skip_space(cursor) != '\0') {
        free(text);
        if (error && !*error) db_fail(error, "invalid JSON document");
        return 0;
    }
    json_tokener *tokener = json_tokener_new();
    json_object *value = json_tokener_parse_ex(tokener, text, (int)got);
    enum json_tokener_error code = json_tokener_get_error(tokener);
    json_tokener_free(tokener);
    free(text);
    if (code != json_tokener_success || !value) {
        if (value) json_object_put(value);
        return db_fail(error, "invalid JSON document");
    }
    *out = value;
    return 1;
}

static int valid_digits(const char *s, int positive) {
    if (!s || !*s || (positive && *s == '0')) return 0;
    for (; *s; s++) if (!isdigit((unsigned char)*s)) return 0;
    return 1;
}

static int valid_hex(const char *s, size_t min, size_t max) {
    size_t n = 0;
    if (!s) return 0;
    for (; *s; s++, n++) if (!isxdigit((unsigned char)*s)) return 0;
    return n >= min && n <= max;
}

static int parse_int_list(const char *text, long *values, size_t max, size_t *count) {
    if (!text || !*text) return 0;
    size_t n = 0;
    const char *p = text;
    while (*p) {
        if (n == max || !isdigit((unsigned char)*p)) return 0;
        errno = 0;
        char *end = NULL;
        long value = strtol(p, &end, 10);
        if (errno || end == p) return 0;
        values[n++] = value;
        if (*end == '\0') {
            p = end;
            break;
        }
        if (*end != ',' || end[1] == '\0') return 0;
        p = end + 1;
    }
    *count = n;
    return n > 0;
}

int db_config_validate(const char *raw, json_object **out, char **error) {
    if (!db_text_valid(raw)) return db_fail(error, "config: missing/empty string");
    char *copy = strdup(raw);
    if (!copy) return db_fail(error, "out of memory");
    json_object *parsed = json_object_new_object();
    if (!parsed) {
        free(copy);
        return db_fail(error, "out of memory");
    }
    char *save = NULL;
    char *token = strtok_r(copy, " \t\r\n", &save);
    if (!token || strcmp(token, "config") != 0) {
        free(copy);
        json_object_put(parsed);
        return db_fail(error, "config: missing prefix");
    }
    while ((token = strtok_r(NULL, " \t\r\n", &save))) {
        char *equals = strchr(token, '=');
        if (!equals || equals == token || !equals[1]) {
            free(copy);
            json_object_put(parsed);
            return db_fail(error, "config: malformed/duplicate field");
        }
        *equals = '\0';
        if (db_get(parsed, token)) {
            free(copy);
            json_object_put(parsed);
            return db_fail(error, "config: malformed/duplicate field");
        }
        json_object_object_add(parsed, token, json_object_new_string(equals + 1));
    }
    free(copy);
    for (size_t i = 0; i < sizeof(required_config) / sizeof(required_config[0]); i++) {
        if (!db_get(parsed, required_config[i])) {
            json_object_put(parsed);
            return db_fail(error, "config: missing required fields");
        }
    }
    const char *rev = db_string(parsed, "rev");
    size_t rev_len = rev ? strlen(rev) : 0;
    int dirty = rev && rev_len > 6 && strcmp(rev + rev_len - 6, "-dirty") == 0;
    char *base_rev = dirty ? strndup(rev, rev_len - 6) : NULL;
    const char *checked_rev = dirty ? base_rev : rev;
    int valid_revision = checked_rev && valid_hex(checked_rev, 7, 40);
    free(base_rev);
    if (!valid_revision) {
        json_object_put(parsed);
        return db_fail(error, "config: malformed source revision");
    }
    if (!valid_hex(db_string(parsed, "cubin_fnv"), 16, 16)) {
        json_object_put(parsed);
        return db_fail(error, "config: missing cubin identity");
    }
    if (strcmp(db_string(parsed, "layers"), "48/48") || strcmp(db_string(parsed, "dies"), "6") || strcmp(db_string(parsed, "nblk"), "512")) {
        json_object_put(parsed);
        return db_fail(error, "config: requires six dies, 48/48 layers, nblk=512");
    }
    if (!valid_digits(db_string(parsed, "maxt"), 1)) {
        json_object_put(parsed);
        return db_fail(error, "config: invalid maxt");
    }
    const char *bool_fields[] = {"kv_host", "ple_disk", "two_position"};
    for (size_t i = 0; i < 3; i++) {
        const char *value = db_string(parsed, bool_fields[i]);
        if (!value || (strcmp(value, "0") && strcmp(value, "1"))) {
            json_object_put(parsed);
            return db_fail(error, "config: invalid %s", bool_fields[i]);
        }
    }
    if (!valid_digits(db_string(parsed, "img_rows"), 0) || !valid_digits(db_string(parsed, "prefill_chunk"), 0)) {
        json_object_put(parsed);
        return db_fail(error, "config: invalid numeric field");
    }
    long shares[6], attn[12], dies[6];
    size_t share_count = 0, attn_count = 0, die_count = 0;
    if (!parse_int_list(db_string(parsed, "split"), shares, 6, &share_count) || share_count != 6) {
        json_object_put(parsed);
        return db_fail(error, "split: require six positive shares");
    }
    for (size_t i = 0; i < share_count; i++) if (shares[i] <= 0) {
        json_object_put(parsed);
        return db_fail(error, "split: require six positive shares");
    }
    if (!parse_int_list(db_string(parsed, "attn_map"), attn, 12, &attn_count) || attn_count != 12) {
        json_object_put(parsed);
        return db_fail(error, "attn_map: require twelve valid dies");
    }
    for (size_t i = 0; i < attn_count; i++) if (attn[i] < 0 || attn[i] >= 6) {
        json_object_put(parsed);
        return db_fail(error, "attn_map: require twelve valid dies");
    }
    if (!parse_int_list(db_string(parsed, "attn_dies"), dies, 6, &die_count)) {
        json_object_put(parsed);
        return db_fail(error, "attn_dies: inconsistent map");
    }
    long expected[6];
    size_t expected_count = 0;
    for (size_t i = 0; i < attn_count; i++) {
        int found = 0;
        for (size_t j = 0; j < expected_count; j++) if (expected[j] == attn[i]) found = 1;
        if (!found) expected[expected_count++] = attn[i];
    }
    for (size_t i = 0; i < expected_count; i++) for (size_t j = i + 1; j < expected_count; j++) if (expected[j] < expected[i]) {
        long tmp = expected[i]; expected[i] = expected[j]; expected[j] = tmp;
    }
    for (size_t i = 0; i < die_count; i++) for (size_t j = i + 1; j < die_count; j++) if (dies[j] < dies[i]) {
        long tmp = dies[i]; dies[i] = dies[j]; dies[j] = tmp;
    }
    int duplicate_dies = 0;
    for (size_t i = 1; i < die_count; i++) if (dies[i] == dies[i - 1]) duplicate_dies = 1;
    if (duplicate_dies || die_count != expected_count || memcmp(dies, expected, die_count * sizeof(long)) != 0) {
        json_object_put(parsed);
        return db_fail(error, "attn_dies: inconsistent map");
    }
    *out = parsed;
    return 1;
}

static char *trim(char *text) {
    while (*text && isspace((unsigned char)*text)) text++;
    size_t length = strlen(text);
    while (length && isspace((unsigned char)text[length - 1])) text[--length] = '\0';
    return text;
}

int db_profile_validate(json_object *entry, db_profile *out, char **error) {
    if (!entry || json_object_get_type(entry) != json_type_object) return db_fail(error, "profile: expected object");
    long long die = 0;
    if (!db_integer(db_get(entry, "die"), &die)) return db_fail(error, "profile die: not an integer");
    if (die < 0 || die >= 6) return db_fail(error, "profile: invalid die");
    const char *raw = db_string(entry, "raw");
    if (!db_text_valid(raw)) return db_fail(error, "profile raw: missing/empty string");
    char *copy = strdup(raw);
    if (!copy) return db_fail(error, "out of memory");
    char *save = NULL;
    char *line = strtok_r(copy, "\n", &save);
    int phases = 0, consumed = 0;
    double total = 0;
    if (!line || sscanf(trim(line), "prof: %d phases, %lf ms between the first and last timestamp%n", &phases, &total, &consumed) != 2 ||
        *trim(trim(line) + consumed) != '\0') {
        free(copy);
        return db_fail(error, "die %lld: malformed profile header", die);
    }
    if (phases <= 1 || !isfinite(total) || total <= 0) {
        free(copy);
        return db_fail(error, "die %lld: incomplete profile", die);
    }
    char names[256][128];
    int counts[256];
    double totals[256];
    size_t operations = 0;
    int footer = 0;
    while ((line = strtok_r(NULL, "\n", &save))) {
        line = trim(line);
        if (!*line) continue;
        if (strncmp(line, "longest:", 8) == 0) {
            footer = 1;
            break;
        }
        char name[128] = {0};
        int count = 0;
        double us = 0, average = 0;
        consumed = 0;
        if (sscanf(line, "%127s %d phases %lf us (%lf us each)%n", name, &count, &us, &average, &consumed) != 4 ||
            *trim(line + consumed) != '\0' || !*name || count <= 0 || operations >= 256) {
            free(copy);
            return db_fail(error, "die %lld: malformed operation row", die);
        }
        for (const char *p = name; *p; p++) if (!(isalnum((unsigned char)*p) || strchr("_.:+-", *p))) {
            free(copy);
            return db_fail(error, "die %lld: malformed operation row", die);
        }
        for (size_t i = 0; i < operations; i++) if (strcmp(names[i], name) == 0) {
            free(copy);
            return db_fail(error, "die %lld: duplicate operation %s", die, name);
        }
        if (!isfinite(us) || us < 0 || !isfinite(average) || average < 0) {
            free(copy);
            return db_fail(error, "die %lld: nonfinite/nonnumeric operation time", die);
        }
        if (fabs(average - us / count) > 0.050001 + 0.050001 / count) {
            free(copy);
            return db_fail(error, "die %lld: inconsistent operation average", die);
        }
        strcpy(names[operations], name);
        counts[operations] = count;
        totals[operations] = us;
        operations++;
    }
    if (!footer) {
        free(copy);
        return db_fail(error, "die %lld: incomplete profile", die);
    }
    int has_actq = 0;
    long total_phases = 0;
    double sum_us = 0;
    for (size_t i = 0; i < operations; i++) {
        if (strcmp(names[i], "actq256") == 0) {
            has_actq = 1;
            out->actq256_us = totals[i];
        }
        total_phases += counts[i];
        sum_us += totals[i];
    }
    free(copy);
    if (!has_actq) return db_fail(error, "die %lld: missing actq256", die);
    if (total_phases != phases - 1) return db_fail(error, "die %lld: incomplete phase counts", die);
    if (fabs(sum_us / 1000.0 - total) > 0.005001 + operations * 0.000051)
        return db_fail(error, "die %lld: operation times do not reconcile to header", die);
    size_t order[256];
    for (size_t i = 0; i < operations; i++) order[i] = i;
    for (size_t i = 0; i < operations; i++) for (size_t j = i + 1; j < operations; j++) {
        if (strcmp(names[order[i]], names[order[j]]) > 0) {
            size_t temporary = order[i]; order[i] = order[j]; order[j] = temporary;
        }
    }
    size_t signature_offset = 0;
    out->signature[0] = '\0';
    for (size_t i = 0; i < operations; i++) {
        int written = snprintf(out->signature + signature_offset, sizeof(out->signature) - signature_offset,
                               "%s%s:%d", i ? "," : "", names[order[i]], counts[order[i]]);
        if (written < 0 || (size_t)written >= sizeof(out->signature) - signature_offset)
            return db_fail(error, "profile signature too long");
        signature_offset += (size_t)written;
    }
    out->die = (int)die;
    out->phases = phases;
    out->total_ms = total;
    return 1;
}

int db_request_equal(json_object *request) {
    if (!request || json_object_get_type(request) != json_type_object || json_object_object_length(request) != 3) return 0;
    json_object *messages = db_get(request, "messages");
    if (!messages || json_object_get_type(messages) != json_type_array || json_object_array_length(messages) != 1) return 0;
    json_object *message = json_object_array_get_idx(messages, 0);
    if (!message || json_object_get_type(message) != json_type_object || json_object_object_length(message) != 2) return 0;
    if (strcmp(db_string(message, "role") ? db_string(message, "role") : "", "user") != 0 ||
        strcmp(db_string(message, "content") ? db_string(message, "content") : "", "What is the capital of France? Answer in one word.") != 0) return 0;
    long long max_tokens = 0;
    double temperature = -1;
    return db_integer(db_get(request, "max_tokens"), &max_tokens) && max_tokens == 40 &&
           db_number(db_get(request, "temperature"), &temperature) && temperature == 0.0;
}
