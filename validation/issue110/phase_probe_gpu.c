/* Actual-kernel test for the issue110 dispatch probe. */
#include <cuda.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "phase_probe.h"

CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

#define CONTROL_BYTES (0x100u + 4u * 4096u)
#define CONTROL_WORDS 64u
#define XCOPY_UNITS 10u
#define XCOPY_CHUNK 4096u
#define XCOPY_BYTES ((size_t)XCOPY_UNITS * XCOPY_CHUNK)
#define GUARD_BYTES 64u
#define BF16_K 320u
#define BF16_ROWS 16u
#define BF16_GUARD 16u

static unsigned failures;
static int failure_mode;

static void launch_failure(CUresult result, const char *expression, int line) {
    if (failure_mode == 1) {
        /* A launch/configuration failure is not evidence for this check. The
         * child must have executed the dispatch store, then faulted on its
         * deliberately unmapped XCOPY source. No CUDA calls follow the error. */
        if (result != CUDA_ERROR_ILLEGAL_ADDRESS) {
            fprintf(stderr, "FAULT_FIXTURE unexpected_cuda_error=%d\n", (int)result);
            fflush(NULL); _Exit(73);
        }
        unsigned observed = 0;
        const uint64_t expected = ((uint64_t)PHASE_PROBE_TOKEN_FIRST << 32) | 32u;
        for (unsigned cta = 0; cta < PHASE_PROBE_CTAS; cta++)
            observed += phase_probe_observe(0, cta) == expected;
        if (!observed) {
            fprintf(stderr, "FAULT_FIXTURE no_expected_host_observation\n");
            fflush(NULL); _Exit(74);
        }
        fprintf(stderr, "FAULT_FIXTURE deliberate_invalid_read=1 error=%d host_observations=%u cuda_calls_after_error=0\n",
                (int)result, observed);
    }
    phase_probe_fault(result, expression, line);
}

static int failure_children(const char *executable, const char *cubin) {
    const char *modes[] = {"--fault-child", "--limit-child"};
    const int expected[] = {70, 72};
    for (unsigned i = 0; i < 2; i++) {
        fflush(NULL);
        pid_t child = fork();
        if (child < 0) { perror("fork failure fixture"); return 1; }
        if (!child) {
            execlp(executable, executable, modes[i], cubin, (char *)NULL);
            perror("exec failure fixture"); _Exit(127);
        }
        int status;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        if (result != child || !WIFEXITED(status) || WEXITSTATUS(status) != expected[i]) {
            fprintf(stderr, "failure fixture mode=%s unexpected_child_result=%d status=%d\n",
                    modes[i], (int)result, result == child ? status : -1);
            return 1;
        }
        printf("phase_probe_failure case=%s exit=%d PASS\n", modes[i], expected[i]);
    }
    printf("phase_probe_failure: deliberate-CUDA-error and synthetic-host-limit paths passed\n");
    return 0;
}

static int cu_ok(CUresult result, const char *expression) {
    if (result == CUDA_SUCCESS) return 1;
    const char *message = "unknown CUDA driver error";
    (void)cuGetErrorString(result, &message);
    fprintf(stderr, "CUDA_ERROR expression=%s result=%d message=%s\n",
            expression, (int)result, message ? message : "unknown");
    return 0;
}

#define CU_OK(call) cu_ok((call), #call)

static int check_probe(const prog_t *program, uint32_t first, uint32_t end,
                       unsigned expected_mode, uint32_t decoded_token,
                       uint32_t physical_phase, unsigned expected_opcode,
                       unsigned expected_units, const char *label) {
    phase_probe_descriptor descriptors[2];
    phase_probe_record record;
    if (!program || program->n < 1 || program->n > 2) {
        fprintf(stderr, "%s: unexpected descriptor count\n", label);
        return 0;
    }
    for (int i = 0; i < program->n; i++) {
        descriptors[i] = (phase_probe_descriptor){
            program->p[i].w[0], program->p[i].w[1], program->p[i].w[30]
        };
    }
    uint64_t expected = ((uint64_t)first << 32) | (UINT64_C(1) << 5) | expected_mode;
    unsigned bad = 0;
    for (unsigned cta = 0; cta < PHASE_PROBE_CTAS; cta++) {
        uint64_t observed = phase_probe_observe(0, cta);
        if (observed != expected) {
            if (bad < 4)
                fprintf(stderr, "%s: CTA%u tag=%016llx expected=%016llx\n",
                        label, cta, (unsigned long long)observed,
                        (unsigned long long)expected);
            bad++;
            continue;
        }
        if (phase_probe_decode(observed, descriptors, (size_t)program->n,
                               first, end, &record) ||
            record.base_token != first || record.token != decoded_token ||
            record.primary_index != 0 || record.physical_index != physical_phase ||
            record.mode != expected_mode || record.descriptor.opcode != expected_opcode ||
            record.descriptor.units != expected_units) {
            if (bad < 4)
                fprintf(stderr, "%s: CTA%u tag decode did not match expected descriptor\n",
                        label, cta);
            bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "%s: failed CTA tag checks=%u\n", label, bad);
        failures++;
        return 0;
    }
    printf("phase_probe case=%s tag=%016llx ctas=%u decoded_token=%u physical_phase=%u mode=%u PASS\n",
           label, (unsigned long long)expected, PHASE_PROBE_CTAS,
           decoded_token, physical_phase, expected_mode);
    return 1;
}

static int launch_program(CUfunction function, CUdeviceptr device_program,
                          CUdeviceptr device_control, const prog_t *program,
                          uint32_t first, uint32_t end, uint64_t request) {
    uint32_t control[CONTROL_WORDS] = {0};
    if (!program || program->n < 1 || program->n > 2 || first >= end) {
        fprintf(stderr, "invalid launch program or token segment\n");
        return 0;
    }
    unsigned nph = (unsigned)program->n;
    unsigned ntok = end;
    control[2] = first;
    control[9] = end - first;
    if (phase_probe_enabled()) {
        CUdeviceptr mapped = phase_probe_pointer(0);
        if (!mapped) {
            fprintf(stderr, "phase probe enabled without a mapped pointer\n");
            return 0;
        }
        memcpy((unsigned char *)control + PHASE_PROBE_CTL_OFFSET,
               &mapped, sizeof mapped);
    }
    phase_probe_upload_begin();
    if (phase_probe_program(0, program)) {
        fprintf(stderr, "phase_probe_program rejected test descriptor\n");
        return 0;
    }
    if (!CU_OK(cuMemsetD8(device_control, 0, CONTROL_BYTES)) ||
        !CU_OK(cuMemcpyHtoD(device_control, control, sizeof control)) ||
        !CU_OK(cuMemcpyHtoD(device_program, program->p,
                            (size_t)program->n * sizeof *program->p)))
        return 0;
    phase_probe_launch(request, first, end);
    void *arguments[] = { &device_program, &nph, &device_control, &ntok };
    CUresult result = cuLaunchKernel(function, 16, 1, 1, 256, 1, 1,
                                     0, 0, arguments, NULL);
    if (result != CUDA_SUCCESS) {
        launch_failure(result, "cuLaunchKernel", __LINE__);
        (void)cu_ok(result, "cuLaunchKernel");
        return 0;
    }
    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        launch_failure(result, "cuCtxSynchronize", __LINE__);
        (void)cu_ok(result, "cuCtxSynchronize");
        return 0;
    }
    if (failure_mode == 1) {
        fprintf(stderr, "FAULT_FIXTURE invalid read unexpectedly succeeded\n");
        fflush(NULL); _Exit(75);
    }
    if (failure_mode == 2) {
        fprintf(stderr, "LIMIT_FIXTURE synthetic_host_limit_call=1 actual_kernel_expiry=not-tested\n");
        phase_probe_limit(__LINE__);
    }
    phase_probe_done();
    return 1;
}

static int copy_case(CUfunction function, CUdeviceptr device_program,
                     CUdeviceptr device_control, unsigned positions,
                     uint32_t first, int expect_probe, unsigned mode,
                     uint32_t decoded_token, uint32_t physical_phase,
                     const char *label, uint64_t request) {
    CUdeviceptr source_device[2] = {0, 0}, output_device[2] = {0, 0};
    unsigned char *source_host[2] = {NULL, NULL};
    unsigned char *output_host[2] = {NULL, NULL};
    prog_t scalar[2], paired;
    int scalar_init[2] = {0, 0}, paired_init = 0;
    int ok = 0;
    size_t total = XCOPY_BYTES + 2u * GUARD_BYTES;
    if (positions < 1 || positions > 2) return 0;
    prog_init(&scalar[0]); scalar_init[0] = 1;
    if (positions == 2) { prog_init(&scalar[1]); scalar_init[1] = 1; }
    prog_init(&paired); paired_init = 1;

    for (unsigned position = 0; position < positions; position++) {
        source_host[position] = malloc(total);
        output_host[position] = malloc(total);
        if (!source_host[position] || !output_host[position]) {
            fprintf(stderr, "%s: host allocation failed\n", label);
            goto done;
        }
        memset(source_host[position], 0x3c, total);
        memset(output_host[position], 0xa5, total);
        for (size_t i = 0; i < XCOPY_BYTES; i++)
            source_host[position][GUARD_BYTES + i] =
                (unsigned char)((i * 29u + 17u * position + 3u) & 0xffu);
        if (!CU_OK(cuMemAlloc(&source_device[position], total)) ||
            !CU_OK(cuMemAlloc(&output_device[position], total)) ||
            !CU_OK(cuMemcpyHtoD(source_device[position], source_host[position], total)) ||
            !CU_OK(cuMemcpyHtoD(output_device[position], output_host[position], total)))
            goto done;
        phase_t *phase = prog_add(&scalar[position], OP_XCOPY, XCOPY_UNITS);
        if (!phase) {
            fprintf(stderr, "%s: descriptor allocation failed\n", label);
            goto done;
        }
        ph_ptr(phase, 0x140, source_device[position] + GUARD_BYTES);
        ph_ptr(phase, 0x148, output_device[position] + GUARD_BYTES);
    }

    const prog_t *program = &scalar[0];
    if (positions == 2) {
        if (prog_pair(&paired, &scalar[0], &scalar[1])) {
            fprintf(stderr, "%s: prog_pair rejected copy descriptors\n", label);
            goto done;
        }
        program = &paired;
    }
    if (!launch_program(function, device_program, device_control, program,
                        first, first + positions, request))
        goto done;
    if (expect_probe && !check_probe(program, first, first + positions, mode,
                                     decoded_token, physical_phase, OP_XCOPY,
                                     XCOPY_UNITS, label))
        goto done;
    if (!expect_probe)
        for (unsigned cta = 0; cta < PHASE_PROBE_CTAS; cta++)
            if (phase_probe_observe(0, cta)) {
                fprintf(stderr, "%s: filtered CTA%u was recorded\n", label, cta);
                failures++;
                goto done;
            }

    for (unsigned position = 0; position < positions; position++) {
        if (!CU_OK(cuMemcpyDtoH(output_host[position], output_device[position], total)))
            goto done;
        int guard_bad = 0;
        for (size_t i = 0; i < GUARD_BYTES; i++)
            guard_bad |= output_host[position][i] != 0xa5u ||
                output_host[position][GUARD_BYTES + XCOPY_BYTES + i] != 0xa5u;
        if (guard_bad || memcmp(output_host[position] + GUARD_BYTES,
                                source_host[position] + GUARD_BYTES, XCOPY_BYTES)) {
            fprintf(stderr, "%s: position%u copy or guard mismatch\n", label, position);
            failures++;
            goto done;
        }
    }
    printf("phase_probe case=%s positions=%u copy_bytes=%zu guards=PASS\n",
           label, positions, XCOPY_BYTES);
    ok = 1;

done:
    if (paired_init) prog_free(&paired);
    for (unsigned position = 0; position < 2; position++) {
        if (scalar_init[position]) prog_free(&scalar[position]);
        if (source_device[position]) (void)cuMemFree(source_device[position]);
        if (output_device[position]) (void)cuMemFree(output_device[position]);
        free(source_host[position]);
        free(output_host[position]);
    }
    return ok;
}

static int fused_bf16_case(CUfunction function, CUdeviceptr device_program,
                           CUdeviceptr device_control, uint32_t first,
                           uint64_t request) {
    const size_t weight_bytes = (size_t)BF16_K * BF16_ROWS * sizeof(uint16_t);
    const size_t input_bytes = (size_t)BF16_K * sizeof(float);
    const size_t output_elems = BF16_ROWS + 2u * BF16_GUARD;
    const size_t output_bytes = output_elems * sizeof(float);
    uint16_t *weights = NULL;
    float *input[2] = {NULL, NULL}, *output[2] = {NULL, NULL};
    CUdeviceptr weight_device = 0, input_device[2] = {0, 0}, output_device[2] = {0, 0};
    prog_t scalar[2], paired;
    int scalar_init[2] = {0, 0}, paired_init = 0;
    int ok = 0;

    weights = malloc(weight_bytes);
    input[0] = malloc(input_bytes); input[1] = malloc(input_bytes);
    output[0] = malloc(output_bytes); output[1] = malloc(output_bytes);
    if (!weights || !input[0] || !input[1] || !output[0] || !output[1]) {
        fprintf(stderr, "fused BF16: host allocation failed\n");
        goto done;
    }
    for (size_t i = 0; i < (size_t)BF16_K * BF16_ROWS; i++) weights[i] = 0x3f00u;
    for (unsigned position = 0; position < 2; position++) {
        for (unsigned i = 0; i < BF16_K; i++) input[position][i] = 1.0f;
        for (size_t i = 0; i < output_elems; i++) output[position][i] = -12345.25f;
        if (!CU_OK(cuMemAlloc(&input_device[position], input_bytes)) ||
            !CU_OK(cuMemAlloc(&output_device[position], output_bytes)) ||
            !CU_OK(cuMemcpyHtoD(input_device[position], input[position], input_bytes)) ||
            !CU_OK(cuMemcpyHtoD(output_device[position], output[position], output_bytes)))
            goto done;
    }
    if (!CU_OK(cuMemAlloc(&weight_device, weight_bytes)) ||
        !CU_OK(cuMemcpyHtoD(weight_device, weights, weight_bytes)))
        goto done;

    prog_init(&scalar[0]); scalar_init[0] = 1;
    prog_init(&scalar[1]); scalar_init[1] = 1;
    prog_init(&paired); paired_init = 1;
    ph_bf16(&scalar[0], OP_BF16K320, weight_device, input_device[0],
            output_device[0] + (CUdeviceptr)BF16_GUARD * sizeof(float), BF16_ROWS, 0);
    ph_bf16(&scalar[1], OP_BF16K320, weight_device, input_device[1],
            output_device[1] + (CUdeviceptr)BF16_GUARD * sizeof(float), BF16_ROWS, 0);
    if (prog_pair(&paired, &scalar[0], &scalar[1]) ||
        prog_pair_fuse_bf16(&paired, 1) != 1) {
        fprintf(stderr, "fused BF16: pair builder did not fuse the K320 pair\n");
        goto done;
    }
    if (!launch_program(function, device_program, device_control, &paired,
                        first, first + 2u, request))
        goto done;
    if (!check_probe(&paired, first, first + 2u, 24u, first, 0u,
                     OP_BF16K320P, 16u, "fused_bf16"))
        goto done;

    for (unsigned position = 0; position < 2; position++) {
        if (!CU_OK(cuMemcpyDtoH(output[position], output_device[position], output_bytes)))
            goto done;
        for (unsigned i = 0; i < BF16_GUARD; i++) {
            if (output[position][i] != -12345.25f ||
                output[position][BF16_GUARD + BF16_ROWS + i] != -12345.25f) {
                fprintf(stderr, "fused BF16: position%u output guard changed\n", position);
                failures++;
                goto done;
            }
        }
        for (unsigned row = 0; row < BF16_ROWS; row++) {
            float value = output[position][BF16_GUARD + row];
            if (!isfinite(value) || value != 160.0f) {
                fprintf(stderr, "fused BF16: position%u row%u got %.9g expected 160\n",
                        position, row, value);
                failures++;
                goto done;
            }
        }
    }
    printf("phase_probe case=fused_bf16 positions=2 rows=%u value=160 guards=PASS\n",
           BF16_ROWS);
    ok = 1;

done:
    if (paired_init) prog_free(&paired);
    for (unsigned position = 0; position < 2; position++) {
        if (scalar_init[position]) prog_free(&scalar[position]);
        if (input_device[position]) (void)cuMemFree(input_device[position]);
        if (output_device[position]) (void)cuMemFree(output_device[position]);
        free(input[position]);
        free(output[position]);
    }
    if (weight_device) (void)cuMemFree(weight_device);
    free(weights);
    return ok;
}

int main(int argc, char **argv) {
    CUdevice device;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr device_program = 0, device_control = 0;
    int rc = 1;

    const char *cubin = argc == 2 ? argv[1] : argc == 3 ? argv[2] : NULL;
    if (argc == 3 && !strcmp(argv[1], "--failure-paths"))
        return failure_children(argv[0], cubin);
    if (argc == 3 && !strcmp(argv[1], "--fault-child")) failure_mode = 1;
    if (argc == 3 && !strcmp(argv[1], "--limit-child")) failure_mode = 2;
    if (!cubin || (argc != 2 && !failure_mode)) {
        fprintf(stderr, "usage: %s [--failure-paths] INSTRUMENTED_DIE_CUBIN\n", argv[0]);
        return 2;
    }
    if (!CU_OK(cuInit(0)) || !CU_OK(cuDeviceGet(&device, 0)) ||
        !CU_OK(cuCtxCreate_v2(&context, CU_CTX_SCHED_SPIN, device)))
        goto done;
    if (!CU_OK(cuModuleLoad(&module, cubin)) ||
        !CU_OK(cuModuleGetFunction(&function, module, "die")) ||
        !CU_OK(cuMemAlloc(&device_program, 128u * PHASE_PROBE_PHASES)) ||
        !CU_OK(cuMemAlloc(&device_control, CONTROL_BYTES)))
        goto done;

    if (failure_mode) {
        if (phase_probe_create(&context, 1) != CUDA_SUCCESS) {
            fprintf(stderr, "failure fixture: phase_probe_create failed\n");
            goto done;
        }
        if (failure_mode == 1) {
            CUdeviceptr destination = 0;
            prog_t invalid;
            prog_init(&invalid);
            if (!CU_OK(cuMemAlloc(&destination, XCOPY_BYTES))) goto done;
            phase_t *phase = prog_add(&invalid, OP_XCOPY, XCOPY_UNITS);
            if (!phase) goto done;
            ph_ptr(phase, 0x140, 0); /* Intentionally unmapped read; no invalid write. */
            ph_ptr(phase, 0x148, destination);
            (void)launch_program(function, device_program, device_control, &invalid,
                                 PHASE_PROBE_TOKEN_FIRST, PHASE_PROBE_TOKEN_FIRST + 1, 90);
            fprintf(stderr, "failure fixture: CUDA fault path returned\n");
            fflush(NULL); _Exit(76);
        }
        (void)copy_case(function, device_program, device_control, 1, PHASE_PROBE_TOKEN_FIRST,
                        1, 0, PHASE_PROBE_TOKEN_FIRST, 0, "host_limit", 91);
        fprintf(stderr, "failure fixture: host limit path returned\n");
        fflush(NULL); _Exit(77);
    }

    if (!copy_case(function, device_program, device_control, 1, PHASE_PROBE_TOKEN_FIRST,
                   0, 0, 0, 0, "null_pointer_copy", 1))
        goto done;
    if (phase_probe_create(&context, 1) != CUDA_SUCCESS) {
        fprintf(stderr, "phase_probe_create failed\n");
        goto done;
    }
    if (!copy_case(function, device_program, device_control, 1, 7,
                   0, 0, 0, 0, "filtered_before_window", 2) ||
        !copy_case(function, device_program, device_control, 1, PHASE_PROBE_TOKEN_END,
                   0, 0, 0, 0, "filtered_after_window", 3) ||
        !copy_case(function, device_program, device_control, 2, PHASE_PROBE_TOKEN_FIRST - 1,
                   0, 0, 0, 0, "filtered_pair_base", 7) ||
        !copy_case(function, device_program, device_control, 1, PHASE_PROBE_TOKEN_FIRST,
                   1, 0, PHASE_PROBE_TOKEN_FIRST, 0, "scalar", 4) ||
        !copy_case(function, device_program, device_control, 1, PHASE_PROBE_TOKEN_END - 1,
                   1, 0, PHASE_PROBE_TOKEN_END - 1, 0, "scalar_window_last", 8) ||
        !copy_case(function, device_program, device_control, 2, PHASE_PROBE_TOKEN_FIRST,
                   1, 0, PHASE_PROBE_TOKEN_FIRST + 1, 1, "generic_pair_B", 5) ||
        !fused_bf16_case(function, device_program, device_control, PHASE_PROBE_TOKEN_FIRST, 6))
        goto done;

    printf("phase_probe_gpu: null-pointer/filtered/scalar/generic-pair/fused-BF16 checks passed\n");
    rc = 0;

done:
    phase_probe_destroy();
    if (device_control) (void)cuMemFree(device_control);
    if (device_program) (void)cuMemFree(device_program);
    if (module) (void)cuModuleUnload(module);
    if (context) (void)cuCtxDestroy(context);
    if (rc) fprintf(stderr, "phase_probe_gpu: failures=%u\n", failures + 1u);
    return rc;
}
