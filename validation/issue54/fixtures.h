#ifndef DIE_BALANCE_FIXTURES_H
#define DIE_BALANCE_FIXTURES_H

#include <json-c/json.h>

json_object *fixture_config(const char *split);
json_object *fixture_profile(int die, double total_ms, double actq_us);
json_object *fixture_capture(double rate, double spread, double total_ms, const char *split);
void fixture_rate(json_object *run, double rate);
int fixture_write(const char *path, json_object *value);

#endif
