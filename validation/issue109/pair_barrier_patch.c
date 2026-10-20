#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PAIR_PATCH_MAX_INPUT (64u * 1024u * 1024u)

static const char pair_branch[] = "@!P5 BRA PHASE_END;";
static const char pair_mode_test[] = "ISETP.EQ.U32.AND P5, PT, bar_g, 0x8, PT;";
static const char pair_after_branch[] = "MEMBAR.GL.IVALLD;";
static const char patch_marker[] = "ISSUE109_PAIR_BODY_GRID_BARRIER_BEGIN";

/* Test-only GEN barrier. It deliberately omits FLAGS_OFF/XG_OFF handling:
 * the generic paired bodies share one exchange publication after both bodies. */
static const char pair_grid_barrier[] =
    "# ISSUE109_PAIR_BODY_GRID_BARRIER_BEGIN\n"
    "--:-:-:-:6      MOV bar_cp0, ctl0;\n"
    "--:-:-:-:6      MOV bar_cp1, ctl1;\n"
    "--:-:-:-:6      ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;\n"
    "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, tid, RZ, PT;\n"
    "--:-:1:-:1      LDS bar_g, [RZ+0xbf78];\n"
    "--:-:-:-:6      SHL bar_t, lane, 0x2;\n"
    "--:-:-:-:6      SHL bar_g2, cta, 0x2;\n"
    "--:-:-:-:6      ISETP.LT.U32.AND P3, PT, lane, 0x10, PT;\n"
    "--:-:-:-:6  @P6 BRA ISSUE109_PAIR_BODY_GRID_DONE;\n"
    "04:-:-:-:6      IADD xa0.CC, bar_cp0, bar_t;\n"
    "--:-:-:-:6      IADD.X xa1, bar_cp1, RZ;\n"
    "--:-:-:-:6      IADD xb0.CC, bar_cp0, bar_g2;\n"
    "--:-:-:-:6      IADD.X xb1, bar_cp1, RZ;\n"
    "--:-:-:-:6      MEMBAR.GL.IVALLD;\n"
    "01:-:-:-:6      IADD bar_g, bar_g, 0x1;\n"
    "--:-:-:-:6      MOV32I bar_g2, 0xffffffff;\n"
    "--:2:-:-:6  @!P5 STG.E.WT [xb+0x80], bar_g;\n"
    "ISSUE109_PAIR_BODY_GRID_SPIN:\n"
    "--:-:1:-:2  @P3 LDG.E.CV bar_g2, [xa+0x80];\n"
    "01:-:-:-:6      ISETP.GE.U32.AND P4, PT, bar_g2, bar_g, PT;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      VOTE.ALL P4, P4;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "02:-:6:-:2      S2R bar_t, SR_GLOBALTIMERHI;\n"
    "--:-:6:-:2      LDS bar_old, [RZ+0xbf5c];\n"
    "20:-:-:-:6      IADD bar_t, bar_t, -bar_old;\n"
    "--:-:-:-:6      ISETP.GE.U32.AND P5, PT, bar_t, 0x46, PT;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      VOTE.ANY P5, P5;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6  @P5 BRA ISSUE109_PAIR_BODY_GRID_TIMEOUT;\n"
    "--:-:-:-:6 @!P4 BRA ISSUE109_PAIR_BODY_GRID_SPIN;\n"
    "--:-:-:-:6      MEMBAR.GL.IVALLD;\n"
    "02:-:-:-:6      IADD bar_t, cta, tid;\n"
    "--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_t, RZ, PT;\n"
    "--:2:-:-:6      STS [RZ+0xbf78], bar_g;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:2:-:-:6  @P4 STG.E [bar_cp+0x4], bar_g;\n"
    "--:-:-:-:6      BRA ISSUE109_PAIR_BODY_GRID_DONE;\n"
    "ISSUE109_PAIR_BODY_GRID_TIMEOUT:\n"
    "--:-:-:-:6      MOV32I bar_old, 0x1;\n"
    "--:2:-:-:6      STS [RZ+0xbf60], bar_old;\n"
    "ISSUE109_PAIR_BODY_GRID_DONE:\n"
    "02:-:-:-:6      BAR.SYNC 0x1;\n"
    "--:-:1:-:2      LDS bar_old, [RZ+0xbf60];\n"
    "01:-:-:-:6      ISETP.NE.U32.AND P5, PT, bar_old, RZ, PT;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6      NOP;\n"
    "--:-:-:-:6  @P5 BRA EXPIRED;\n"
    "# ISSUE109_PAIR_BODY_GRID_BARRIER_END\n";

static size_t find_bytes(const char *haystack, size_t haystack_len,
                         const char *needle, size_t needle_len, size_t start) {
    if (!needle_len || needle_len > haystack_len || start > haystack_len - needle_len) return SIZE_MAX;
    for (size_t i = start; i <= haystack_len - needle_len; i++)
        if (!memcmp(haystack + i, needle, needle_len)) return i;
    return SIZE_MAX;
}

static int line_has_instruction(const char *text, size_t start, size_t end,
                                const char *instruction, int fixed_schedule) {
    static const char schedule[] = "--:-:-:-:6";
    size_t instruction_len = strlen(instruction);
    size_t at;

    if (fixed_schedule) {
        size_t schedule_len = sizeof schedule - 1u;
        if (end - start < schedule_len ||
            memcmp(text + start, schedule, schedule_len) != 0) return 0;
        at = start + schedule_len;
    } else {
        size_t schedule_len = 10u;
        if (end - start < schedule_len || text[start + 2u] != ':' ||
            text[start + 4u] != ':' || text[start + 6u] != ':' ||
            text[start + 8u] != ':') return 0;
        at = start + schedule_len;
    }
    while (at < end && (text[at] == ' ' || text[at] == '\t')) at++;
    if (end - at < instruction_len ||
        memcmp(text + at, instruction, instruction_len) != 0) return 0;
    at += instruction_len;
    while (at < end && (text[at] == ' ' || text[at] == '\t' || text[at] == '\r')) at++;
    return at == end || text[at] == '#';
}

static int read_text_file(const char *path, char **text_out, size_t *length_out) {
    FILE *f = fopen(path, "rb");
    long end;
    char *text;
    size_t length;
    if (!f) { fprintf(stderr, "%s: %s\n", path, strerror(errno)); return -1; }
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0 ||
        (uint64_t)end > PAIR_PATCH_MAX_INPUT || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: invalid or oversized generated SASS input\n", path);
        fclose(f); return -1;
    }
    length = (size_t)end;
    text = malloc(length + 1u);
    if (!text) { fprintf(stderr, "allocate generated SASS input\n"); fclose(f); return -1; }
    if (fread(text, 1, length, f) != length || ferror(f)) {
        fprintf(stderr, "%s: read failed\n", path);
        free(text); fclose(f); return -1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "%s: close failed: %s\n", path, strerror(errno));
        free(text); return -1;
    }
    if (memchr(text, '\0', length)) {
        fprintf(stderr, "%s: generated SASS contains NUL bytes\n", path);
        free(text); return -1;
    }
    text[length] = '\0';
    *text_out = text;
    *length_out = length;
    return 0;
}

static int validate_anchor(const char *text, size_t length, size_t branch_at,
                           size_t *insert_at) {
    size_t line_start = branch_at, line_end, next_start, next_end, previous_start;
    size_t mode_at, mode_line_start, mode_line_end, mode_window = 512u;
    while (line_start && text[line_start - 1u] != '\n') line_start--;
    line_end = branch_at;
    while (line_end < length && text[line_end] != '\n') line_end++;
    if (line_end == length) {
        fprintf(stderr, "pair branch anchor is not newline-terminated\n");
        return -1;
    }
    if (!line_has_instruction(text, line_start, line_end, pair_branch, 1)) {
        fprintf(stderr, "pair branch anchor does not match the expected SASS instruction line\n");
        return -1;
    }
    next_start = line_end + 1u;
    next_end = next_start;
    while (next_end < length && text[next_end] != '\n') next_end++;
    if (next_end == length ||
        !line_has_instruction(text, next_start, next_end, pair_after_branch, 1)) {
        fprintf(stderr, "pair branch is not followed by the expected grid-barrier fence\n");
        return -1;
    }
    previous_start = line_start > mode_window ? line_start - mode_window : 0;
    mode_at = find_bytes(text + previous_start, line_start - previous_start,
                         pair_mode_test, sizeof pair_mode_test - 1u, 0);
    if (mode_at == SIZE_MAX) {
        fprintf(stderr, "pair branch lacks the generic-mode test anchor\n");
        return -1;
    }
    mode_at += previous_start;
    mode_line_start = mode_at;
    while (mode_line_start && text[mode_line_start - 1u] != '\n') mode_line_start--;
    mode_line_end = mode_at;
    while (mode_line_end < line_start && text[mode_line_end] != '\n') mode_line_end++;
    if (!line_has_instruction(text, mode_line_start, mode_line_end, pair_mode_test, 0)) {
        fprintf(stderr, "generic-mode test anchor does not match its SASS instruction line\n");
        return -1;
    }
    *insert_at = line_end + 1u;
    return 0;
}

static int write_new_file(const char *path, const char *prefix, size_t prefix_len,
                          const char *insert, size_t insert_len,
                          const char *suffix, size_t suffix_len) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    const char *parts[3] = {prefix, insert, suffix};
    size_t lengths[3] = {prefix_len, insert_len, suffix_len};
    int failed = 0;
    if (fd < 0) { fprintf(stderr, "%s: %s\n", path, strerror(errno)); return -1; }
    for (unsigned part = 0; part < 3 && !failed; part++) {
        size_t offset = 0;
        while (offset < lengths[part]) {
            ssize_t n = write(fd, parts[part] + offset, lengths[part] - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { failed = 1; break; }
            offset += (size_t)n;
        }
    }
    if (!failed && fsync(fd) < 0) failed = 1;
    if (close(fd) < 0) failed = 1;
    if (failed) {
        int saved = errno ? errno : EIO;
        unlink(path);
        fprintf(stderr, "%s: write failed: %s\n", path, strerror(saved));
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    char *input = NULL;
    size_t input_len = 0;
    size_t branch_at, insert_at;
    size_t occurrences = 0;
    struct stat input_stat, output_stat;
    if (argc != 3) {
        fprintf(stderr, "usage: %s GENERATED_DIE_SASS OUTPUT_PAIR_BARRIER_SASS\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], argv[2])) {
        fprintf(stderr, "input and output paths must differ\n");
        return 2;
    }
    if (stat(argv[1], &input_stat) < 0) {
        fprintf(stderr, "%s: input is not a regular file: %s\n", argv[1], strerror(errno));
        return 1;
    }
    if (!S_ISREG(input_stat.st_mode)) {
        fprintf(stderr, "%s: input is not a regular file\n", argv[1]);
        return 1;
    }
    if (stat(argv[2], &output_stat) == 0 &&
        input_stat.st_dev == output_stat.st_dev && input_stat.st_ino == output_stat.st_ino) {
        fprintf(stderr, "input and output refer to the same file\n");
        return 2;
    }
    if (read_text_file(argv[1], &input, &input_len) < 0) return 1;
    if (find_bytes(input, input_len, patch_marker, sizeof patch_marker - 1u, 0) != SIZE_MAX) {
        fprintf(stderr, "input already contains the issue109 patch marker\n");
        free(input); return 1;
    }
    {
        static const char *const labels[] = {
            "ISSUE109_PAIR_BODY_GRID_SPIN:",
            "ISSUE109_PAIR_BODY_GRID_TIMEOUT:",
            "ISSUE109_PAIR_BODY_GRID_DONE:"
        };
        for (size_t i = 0; i < sizeof labels / sizeof labels[0]; i++) {
            if (find_bytes(input, input_len, labels[i], strlen(labels[i]), 0) != SIZE_MAX) {
                fprintf(stderr, "input already contains issue109 barrier label %s\n", labels[i]);
                free(input); return 1;
            }
        }
    }
    branch_at = 0;
    for (;;) {
        size_t found = find_bytes(input, input_len, pair_branch,
                                  sizeof pair_branch - 1u, branch_at);
        if (found == SIZE_MAX) break;
        occurrences++;
        branch_at = found + sizeof pair_branch - 1u;
        insert_at = found;
        if (validate_anchor(input, input_len, found, &insert_at) < 0) {
            free(input); return 1;
        }
        branch_at = found + sizeof pair_branch - 1u;
    }
    if (occurrences != 1u) {
        fprintf(stderr, "expected exactly one generic-pair branch anchor; found %zu\n", occurrences);
        free(input); return 1;
    }
    /* Recompute the sole insertion point after the uniqueness pass. */
    branch_at = find_bytes(input, input_len, pair_branch, sizeof pair_branch - 1u, 0);
    if (branch_at == SIZE_MAX || validate_anchor(input, input_len, branch_at, &insert_at) < 0) {
        free(input); return 1;
    }
    if (write_new_file(argv[2], input, insert_at,
                       pair_grid_barrier, sizeof pair_grid_barrier - 1u,
                       input + insert_at, input_len - insert_at) < 0) {
        free(input); return 1;
    }
    printf("PAIR_BARRIER_PATCHED mode=generic-pair anchor_count=1 output=%s\n", argv[2]);
    free(input);
    return 0;
}
