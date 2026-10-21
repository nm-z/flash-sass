#ifndef DIE_BALANCE_COMMON_H
#define DIE_BALANCE_COMMON_H

#include <json-c/json.h>

#include <stddef.h>

#define DB_SCHEMA "flash-sass-die-balance-v1"
#define DB_LIMITATION "The last-token profiler omits phase 0; parallel die spans are never added."
#define DB_FLOAT_PATTERN "[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?"

typedef struct {
    int die;
    int phases;
    double total_ms;
    double actq256_us;
    char signature[4096];
} db_profile;

int db_parse_json_file(const char *path, json_object **out, char **error);
char *db_json_serialize(json_object *value, int pretty);
int db_config_validate(const char *raw, json_object **out, char **error);
int db_profile_validate(json_object *entry, db_profile *out, char **error);
int db_request_equal(json_object *request);
int db_text_valid(const char *value);
int db_number(json_object *value, double *out);
int db_integer(json_object *value, long long *out);
const char *db_string(json_object *object, const char *key);
json_object *db_get(json_object *object, const char *key);
int db_fail(char **error, const char *format, ...);

#endif
