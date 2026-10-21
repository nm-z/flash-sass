/* CPU integration regression: no CUDA entry point reaches a driver. */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "prog.h"

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static unsigned bench_context_calls, bench_copy_calls;
static unsigned bench_fail_allocation, bench_program_allocations;
static void *bench_calloc(size_t count, size_t bytes);

static CUresult CUDAAPI bench_context(CUcontext context)
{
    (void)context;
    ++bench_context_calls;
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI bench_copy(void *destination, CUdeviceptr source,
                                   size_t bytes)
{
    ++bench_copy_calls;
    CHECK(destination != NULL && source != 0);
    memcpy(destination, (const void *)(uintptr_t)source, bytes);
    return CUDA_SUCCESS;
}

#undef cuCtxSetCurrent
#undef cuMemcpyDtoH
#define cuCtxSetCurrent bench_context
#define cuMemcpyDtoH bench_copy
#define calloc bench_calloc
#include "engine.c"
#undef calloc
#undef cuCtxSetCurrent
#undef cuMemcpyDtoH

static void *bench_calloc(size_t count, size_t bytes)
{
    if (count == MAXD && bytes == sizeof(prog_t)) {
        ++bench_program_allocations;
        if (bench_program_allocations == bench_fail_allocation)
            return NULL;
    }
    return calloc(count, bytes);
}

static uint64_t bench_uint(const char *text)
{
    char *end;
    CHECK(text != NULL && *text >= '0' && *text <= '9');
    uint64_t value = strtoull(text, &end, 10);
    CHECK(end != text);
    CHECK(*end == '\0' || *end == ' ' || *end == '\n' || *end == '\r');
    return value;
}

static uint64_t bench_field(const char *line, const char *name)
{
    const char *field = strstr(line, name);
    CHECK(field != NULL);
    return bench_uint(field + strlen(name));
}

static void bench_session_error(char *commands, const char *status)
{
    char *output = NULL;
    size_t output_size = 0;
    FILE *input = fmemopen(commands, strlen(commands), "r");
    FILE *result = open_memstream(&output, &output_size);
    CHECK(input != NULL && result != NULL);
    trace_valid = 7;
    CHECK(engine_session(input, result) == 1);
    CHECK(fclose(input) == 0);
    CHECK(fclose(result) == 0);
    CHECK(trace_request == requestmeta.request && trace_valid == 0);
    CHECK(bench_context_calls == 0 && bench_copy_calls == 0);

    unsigned metadata_lines = 0;
    char *save = NULL;
    for (char *line = strtok_r(output, "\n", &save);
         line != NULL; line = strtok_r(NULL, "\n", &save)) {
        if (strncmp(line, "requestmeta ", 12) != 0 &&
            strstr(line, "no completed host trace") == NULL &&
            strstr(line, "requires FLASH_TRACE") == NULL)
            continue;
        CHECK(bench_field(line, "request=") == requestmeta.request);
        const char *field = strstr(line, "status=");
        CHECK(field != NULL && strncmp(field + 7, status, strlen(status)) == 0);
        ++metadata_lines;
    }
    CHECK(metadata_lines == 3);
    free(output);
}

static void bench_invalid_requests(void)
{
    uint64_t initial = requestmeta.request;
    trace_n = trace_valid = 7;
    trace_request = UINT64_C(777);
    CHECK(engine_run(NULL, 0, 0, NULL, NULL, 0, NULL, 0, NULL, NULL) == -1);
    CHECK(requestmeta.request == initial + 1);
    CHECK(trace_request == requestmeta.request && trace_valid == 0);
    CHECK(strcmp(trace_status, "argument_error") == 0);
    CHECK(bench_context_calls == 0 && bench_copy_calls == 0);

    char commands[] =
        "chat 2147483648 42\nrequestmeta\nhosttrace\ntrace 0\nquit\n";
    bench_session_error(commands, "argument_error");
    CHECK(requestmeta.request == initial + 2);

    CHECK(engine_run(NULL, 0, 0, NULL, NULL, 0, NULL, 0, NULL, NULL) == -1);
    CHECK(requestmeta.request == initial + 3);
    CHECK(trace_request == requestmeta.request && trace_valid == 0);
    CHECK(strcmp(trace_status, "argument_error") == 0);
    CHECK(bench_context_calls == 0 && bench_copy_calls == 0);
}

static void bench_allocation_failures(void)
{
    static uint32_t prompt_buffer[34];
    const uint32_t prompt[] = {42};
    h_ids = prompt_buffer;
    two_position = prefill_chunk = 1;
    for (unsigned allocation = 1; allocation <= 3; ++allocation) {
        uint64_t initial = requestmeta.request;
        bench_fail_allocation = allocation;
        bench_program_allocations = 0;
        trace_valid = 7;
        CHECK(engine_run(prompt, 1, 1, NULL, NULL, 0, NULL, 0, NULL, NULL) == -4);
        CHECK(bench_program_allocations == 3);
        CHECK(requestmeta.request == initial + 1);
        CHECK(trace_request == requestmeta.request && trace_valid == 0);
        CHECK(strcmp(requestmeta.status, "allocation_error") == 0);
        CHECK(strcmp(trace_status, "allocation_error") == 0);
        CHECK(requestmeta.build_start_ns > 0);
        CHECK(requestmeta.build_end_ns >= requestmeta.build_start_ns);
        CHECK(requestmeta.run_start_ns == 0 && requestmeta.completed == 0);
        CHECK(bench_context_calls == 0 && bench_copy_calls == 0);
    }
    bench_program_allocations = 0;
    bench_fail_allocation = 2;
    char commands[] = "chat 1 42\nrequestmeta\nhosttrace\ntrace 0\nquit\n";
    bench_session_error(commands, "allocation_error");
    CHECK(bench_program_allocations == 3);
    CHECK(requestmeta.run_start_ns == 0 && requestmeta.completed == 0);
    bench_fail_allocation = bench_program_allocations = 0;
    two_position = prefill_chunk = 0;
    h_ids = NULL;
}

static void bench_trace_export(const char *output_path)
{
    const unsigned tokens = 129, phases = 2;
    uint64_t *raw = calloc(tokens * phases * TRACE_WORDS, sizeof(*raw));
    CHECK(raw != NULL);
    begin_request(1, 128);
    trace_n = trace_valid = tokens;
    trace_first = 0;
    trace_ctas = 1;
    trace_completed = trace_requested = tokens;
    trace_ended = trace_aborted = 0;
    requestmeta.completed = requestmeta.requested = tokens;
    requestmeta.prompt = 1;
    finish_request("completed");
    nblk_qsa = 512;
    prof_nph[0] = phases;
    prof_index[0][0] = 0;
    prof_index[0][1] = 2;
    prof_layer[0][0] = 3;
    prof_layer[0][1] = 31; /* Must not be exported as compact phase 1. */
    prof_layer[0][2] = 7;
    for (unsigned phase = 0; phase < phases; ++phase) {
        prof_op[0][phase] = 0;
        prof_units[0][phase] = 1;
        prof_flags[0][phase] = 0;
    }
    for (unsigned token = 0; token < tokens; ++token) {
        for (unsigned phase = 0; phase < phases; ++phase) {
            uint64_t origin = UINT64_C(9007199254741000) +
                              token * UINT64_C(1000) + phase * UINT64_C(100);
            uint64_t *z = raw + (token * phases + phase) * TRACE_WORDS;
            z[0] = origin;
            z[1] = origin + 1;
            z[2] = origin + 2;
            z[8] = origin + 3;
            z[3] = origin + 4;
            z[4] = origin + 5;
            z[7] = phase + 1 == phases ? origin + 6 : 0;
        }
    }
    dTRACE[0] = (CUdeviceptr)(uintptr_t)raw;
    char *output = NULL;
    size_t output_size = 0;
    FILE *result = open_memstream(&output, &output_size);
    CHECK(result != NULL);
    trace_dump(result, 0);
    CHECK(fclose(result) == 0);
    CHECK(bench_context_calls == 1 && bench_copy_calls == 1);

    if (output_path != NULL) {
        FILE *file = fopen(output_path, "wb");
        CHECK(file != NULL);
        CHECK(fwrite(output, 1, output_size, file) == output_size);
        CHECK(fclose(file) == 0);
    }

    FILE *input = fmemopen(output, output_size, "r");
    CHECK(input != NULL);
    char *line = NULL;
    size_t capacity = 0;
    unsigned rows = 0, headers = 0, footers = 0;
    while (getline(&line, &capacity, input) >= 0) {
        if (strncmp(line, "trace_begin ", 12) == 0) {
            CHECK(bench_field(line, "request=") == requestmeta.request);
            CHECK(bench_field(line, "count=") == tokens);
            CHECK(bench_field(line, "nph=") == phases);
            CHECK(bench_field(line, "ctas=") == 1);
            CHECK(strstr(line, "schema=2") != NULL);
            CHECK(strstr(line, "status=completed") != NULL);
            ++headers;
            continue;
        }
        if (strncmp(line, "trace_end", 9) == 0) {
            ++footers;
            continue;
        }
        if (*line < '0' || *line > '9')
            continue;
        char *fields[20], *remaining = line;
        for (unsigned column = 0; column < 20; ++column) {
            fields[column] = strsep(&remaining, ",");
            CHECK(fields[column] != NULL && *fields[column] != '\0');
        }
        CHECK(remaining == NULL && rows < tokens * phases);
        unsigned token = rows / phases, phase = rows % phases;
        uint64_t origin = UINT64_C(9007199254741000) +
                          token * UINT64_C(1000) + phase * UINT64_C(100);
        CHECK(bench_uint(fields[0]) == token);
        CHECK(bench_uint(fields[1]) == phase);
        CHECK(bench_uint(fields[2]) == (phase == 0 ? 3 : 7));
        CHECK(bench_uint(fields[4]) == 1 && bench_uint(fields[5]) == 0);
        CHECK(bench_uint(fields[7]) == origin);
        CHECK(bench_uint(fields[8]) == origin + 1);
        CHECK(bench_uint(fields[9]) == origin + 2);
        CHECK(bench_uint(fields[10]) == origin + 4);
        CHECK(bench_uint(fields[11]) == origin + 5);
        CHECK(bench_uint(fields[12]) == 0 && bench_uint(fields[13]) == 0);
        CHECK(bench_uint(fields[14]) == (phase == 1 ? origin + 6 : 0));
        CHECK(bench_uint(fields[15]) == 0);
        CHECK(bench_uint(fields[16]) == origin + 3);
        CHECK(bench_uint(fields[17]) == 0 && bench_uint(fields[18]) == 0);
        CHECK(bench_uint(fields[19]) == 0);
        ++rows;
    }
    CHECK(!ferror(input));
    CHECK(headers == 1 && footers == 1 && rows == tokens * phases);
    CHECK(fclose(input) == 0);
    free(line);
    free(output);
    free(raw);
    dTRACE[0] = 0;
}

int main(int argc, char **argv)
{
    CHECK(argc <= 2);
    static core_t fixture_core;
    core = &fixture_core;
    nd = 1;
    MAXT = 32;
    bench_invalid_requests();
    bench_allocation_failures();
    bench_trace_export(argc == 2 ? argv[1] : NULL);
    free(s_ids);
    free(s_toks);
    s_ids = NULL;
    s_toks = NULL;
    puts("bench_engine_trace_test: ok");
    return 0;
}
