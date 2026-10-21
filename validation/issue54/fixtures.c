#define _GNU_SOURCE
#include "fixtures.h"
#include "../../kern/die_balance_common.h"

#include <stdio.h>
#include <string.h>

static json_object *string(const char *value) {
    return json_object_new_string(value);
}

static void put_string(json_object *object, const char *key, const char *value) {
    json_object_object_add(object, key, string(value));
}

json_object *fixture_config(const char *split) {
    char raw[1024];
    snprintf(raw, sizeof(raw),
             "config rev=abcdef1234567 kv_host=0 ple_disk=1 maxt=8192 layers=48/48 "
             "nblk=512 dies=6 attn_dies=1,4,5 split=%s "
             "attn_map=1,1,1,1,4,4,4,4,5,5,5,5 img_rows=0 two_position=0 "
             "prefill_chunk=1 cubin=/synthetic/die.cubin cubin_fnv=0123456789abcdef\n",
             split ? split : "112,97,81,116,97,97");
    return string(raw);
}

json_object *fixture_profile(int die, double total_ms, double actq_us) {
    char raw[1024];
    double sync_us = total_ms * 1000.0 - actq_us;
    snprintf(raw, sizeof(raw),
             "prof: 11 phases, %.2f ms between the first and last timestamp\n"
             "  actq256      6 phases %9.1f us (%6.1f us each)\n"
             "  sync         4 phases %9.1f us (%6.1f us each)\n"
             "  longest: SYNTHETIC CPU fixture\n",
             total_ms, actq_us, actq_us / 6.0, sync_us, sync_us / 4.0);
    json_object *profile = json_object_new_object();
    json_object_object_add(profile, "die", json_object_new_int(die));
    put_string(profile, "raw", raw);
    return profile;
}

void fixture_rate(json_object *run, double rate) {
    double decode_ms = (double)((long)((2000.0 / rate) * 10.0 + 0.5)) / 10.0;
    json_object *response = json_object_object_get(run, "response");
    json_object *timings = json_object_new_object();
    json_object_object_add(timings, "predicted_n", json_object_new_int(3));
    json_object_object_add(timings, "predicted_per_second", json_object_new_double(rate));
    json_object_object_add(timings, "decode_intervals", json_object_new_int(2));
    json_object_object_add(timings, "decode_ms", json_object_new_double(decode_ms));
    json_object_object_add(response, "timings", timings);
    char done[128];
    snprintf(done, sizeof(done), "done 3 1 3 %.1f 20.0 0 0\n", decode_ms + 20.0);
    put_string(run, "done", done);
}

static json_object *token_array(const int *values, size_t count) {
    json_object *array = json_object_new_array();
    for (size_t i = 0; i < count; i++) json_object_array_add(array, json_object_new_int(values[i]));
    return array;
}

static json_object *fixture_run(int index, double rate, double spread, double total_ms,
                                const char *config) {
    static const int input[] = {1, 2, 3};
    static const int output[] = {101, 102, 103};
    json_object *run = json_object_new_object();
    json_object_object_add(run, "index", json_object_new_int(index));
    json_object_object_add(run, "warmup", json_object_new_boolean(index == 0));
    put_string(run, "config", config);
    json_object_object_add(run, "input_tokens", token_array(input, 3));
    json_object_object_add(run, "output_tokens", token_array(output, 3));
    json_object *response = json_object_new_object();
    json_object *choices = json_object_new_array();
    json_object *choice = json_object_new_object();
    put_string(choice, "finish_reason", "stop");
    json_object *message = json_object_new_object();
    put_string(message, "content", "synthetic");
    json_object_object_add(choice, "message", message);
    json_object_array_add(choices, choice);
    json_object_object_add(response, "choices", choices);
    json_object *usage = json_object_new_object();
    json_object_object_add(usage, "prompt_tokens", json_object_new_int(3));
    json_object_object_add(usage, "completion_tokens", json_object_new_int(3));
    json_object_object_add(response, "usage", usage);
    json_object_object_add(run, "response", response);
    json_object *profiles = json_object_new_array();
    for (int die = 0; die < 6; die++) {
        double actq = 5000.0 + (die == 5 ? spread * 1000.0 : 0.0);
        json_object_array_add(profiles, fixture_profile(die, total_ms - die * 0.1, actq));
    }
    json_object_object_add(run, "profiles", profiles);
    fixture_rate(run, rate);
    return run;
}

json_object *fixture_capture(double rate, double spread, double total_ms, const char *split) {
    json_object *capture = json_object_new_object();
    put_string(capture, "schema", "flash-sass-die-balance-v1");
    put_string(capture, "label", "SYNTHETIC CPU fixture");
    put_string(capture, "created_at", "2026-01-01T00:00:00Z");
    put_string(capture, "hostname", "synthetic");
    json_object *request = json_object_new_object();
    json_object *messages = json_object_new_array();
    json_object *message = json_object_new_object();
    put_string(message, "role", "user");
    put_string(message, "content", "What is the capital of France? Answer in one word.");
    json_object_array_add(messages, message);
    json_object_object_add(request, "messages", messages);
    json_object_object_add(request, "max_tokens", json_object_new_int(40));
    json_object_object_add(request, "temperature", json_object_new_int(0));
    json_object_object_add(capture, "request", request);
    json_object_object_add(capture, "warmup_requests", json_object_new_int(1));
    json_object *config_value = fixture_config(split);
    const char *config = json_object_get_string(config_value);
    json_object *manifest = json_object_new_object();
    put_string(manifest, "config_before", config);
    put_string(manifest, "config_after", config);
    put_string(manifest, "frontend_sha256", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    json_object_object_add(manifest, "frontend_env", json_object_new_object());
    put_string(manifest, "core_exe_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    put_string(manifest, "engine_sha256", "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    put_string(manifest, "source_revision", "dddddddddddddddddddddddddddddddddddddddd");
    json_object *model = json_object_new_object();
    put_string(model, "identity_kind", "stat");
    put_string(model, "path", "/synthetic/model.gguf");
    put_string(model, "realpath", "/synthetic/model.gguf");
    json_object_object_add(model, "device", json_object_new_int(1));
    json_object_object_add(model, "inode", json_object_new_int(2));
    json_object_object_add(model, "size", json_object_new_int(3));
    json_object_object_add(model, "mtime_ns", json_object_new_int(4));
    json_object_object_add(manifest, "model", model);
    json_object *hardware = json_object_new_object();
    put_string(hardware, "identity", "SYNTHETIC GPU identities");
    put_string(hardware, "pci_link_snapshot", "SYNTHETIC PCI links");
    json_object_object_add(manifest, "hardware", hardware);
    json_object_object_add(capture, "manifest", manifest);
    json_object *runs = json_object_new_array();
    for (int index = 0; index < 4; index++)
        json_object_array_add(runs, fixture_run(index, rate, spread, total_ms, config));
    json_object_object_add(capture, "runs", runs);
    json_object_put(config_value);
    return capture;
}

int fixture_write(const char *path, json_object *value) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    char *text = db_json_serialize(value, 0);
    if (!text) { fclose(file); return 0; }
    int ok = fputs(text, file) >= 0;
    if (fclose(file) != 0) ok = 0;
    free(text);
    return ok;
}
