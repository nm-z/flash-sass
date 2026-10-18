#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Numerical comparison only: this program cannot attest execution provenance.
 */
#define ABS_LIMIT 0.10
#define NRMS_LIMIT 0.01
#define RMS_FLOOR 1e-8

typedef struct {
  FILE *file;
  const char *path;
  struct stat before;
} input;
typedef struct {
  long double error_sq, reference_sq;
  double max_abs, max_relative;
  uint64_t count;
} metrics;
static FILE *report;
static const char *created_report;

static _Noreturn void invalid(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  fputs("qsa_logits_check: ", stderr);
  vfprintf(stderr, fmt, args);
  fputc('\n', stderr);
  va_end(args);
  if (report)
    fclose(report);
  if (created_report)
    unlink(created_report);
  exit(2);
}
static void emit(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int result = vfprintf(report, fmt, args);
  va_end(args);
  if (result < 0)
    invalid("report write failed: %s", strerror(errno));
}
static uint64_t positive(const char *text, const char *name) {
  uint64_t value = 0;
  if (!*text)
    invalid("%s must be a positive decimal integer", name);
  for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
    if (*p < '0' || *p > '9')
      invalid("invalid %s: %s", name, text);
    unsigned digit = (unsigned)(*p - '0');
    if (value > (UINT64_MAX - digit) / 10)
      invalid("%s overflows uint64", name);
    value = value * 10 + digit;
  }
  if (!value)
    invalid("%s must be positive", name);
  return value;
}
static uint64_t product(uint64_t a, uint64_t b, const char *name) {
  if (a && b > UINT64_MAX / a)
    invalid("%s size overflow", name);
  return a * b;
}
static input open_input(const char *path, uint64_t expected) {
  input in = {0};
  in.path = path;
  in.file = fopen(path, "rb");
  if (!in.file)
    invalid("cannot open %s: %s", path, strerror(errno));
  if (fstat(fileno(in.file), &in.before))
    invalid("cannot stat %s: %s", path, strerror(errno));
  if (!S_ISREG(in.before.st_mode) || in.before.st_size < 0)
    invalid("%s must be a regular file", path);
  if ((uintmax_t)in.before.st_size != (uintmax_t)expected)
    invalid("%s has %ju bytes; expected %" PRIu64, path,
            (uintmax_t)in.before.st_size, expected);
  return in;
}
static void read_exact(input *in, void *buf, size_t count) {
  if (fread(buf, 1, count, in->file) != count)
    invalid("short read or read error in %s", in->path);
}
static void finish_input(input *in) {
  if (fgetc(in->file) != EOF || ferror(in->file))
    invalid("trailing data or read error in %s", in->path);
  struct stat after;
  if (fstat(fileno(in->file), &after))
    invalid("cannot restat %s", in->path);
  if (after.st_dev != in->before.st_dev || after.st_ino != in->before.st_ino ||
      after.st_size != in->before.st_size ||
      after.st_mtim.tv_sec != in->before.st_mtim.tv_sec ||
      after.st_mtim.tv_nsec != in->before.st_mtim.tv_nsec ||
      after.st_ctim.tv_sec != in->before.st_ctim.tv_sec ||
      after.st_ctim.tv_nsec != in->before.st_ctim.tv_nsec)
    invalid("input changed during comparison: %s", in->path);
  if (fclose(in->file))
    invalid("input close failed: %s", in->path);
  in->file = NULL;
}
static uint32_t u32le(const unsigned char *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}
static float f32le(const unsigned char *p) {
  uint32_t bits = u32le(p);
  float value;
  memcpy(&value, &bits, sizeof value);
  return value;
}
static void observe(metrics *m, double ref, double got) {
  double error = got - ref;
  double absolute = fabs(error);
  double relative = absolute / fmax(fabs(ref), RMS_FLOOR);
  m->error_sq += (long double)error * (long double)error;
  m->reference_sq += (long double)ref * (long double)ref;
  if (absolute > m->max_abs)
    m->max_abs = absolute;
  if (relative > m->max_relative)
    m->max_relative = relative;
  ++m->count;
}
static double rms_error(const metrics *m) {
  return (double)sqrtl(m->error_sq / (long double)m->count);
}
static double rms_reference(const metrics *m) {
  return (double)sqrtl(m->reference_sq / (long double)m->count);
}
static double nrms(const metrics *m) {
  return rms_error(m) / fmax(rms_reference(m), RMS_FLOOR);
}
static int passes(const metrics *m) {
  return m->max_abs <= ABS_LIMIT && nrms(m) <= NRMS_LIMIT;
}
static void emit_metrics(const metrics *m) {
  emit("\"max_abs\":%.17g,\"max_relative\":%.17g,"
       "\"rms\":%.17g,\"reference_rms\":%.17g,\"normalized_rms\":%.17g",
       m->max_abs, m->max_relative, rms_error(m), rms_reference(m), nrms(m));
}
int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: %s REF_IDS CAND_IDS REF_LOGITS CAND_LOGITS ROWS VOCAB "
            "OUT.json\n",
            argv[0]);
    return 2;
  }
  if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
      FLT_MAX_EXP != 128 || FLT_MIN_EXP != -125)
    invalid("IEEE binary32 float is required");
  uint64_t rows = positive(argv[5], "ROWS");
  uint64_t vocab = positive(argv[6], "VOCAB");
  if (vocab > UINT32_MAX)
    invalid("VOCAB exceeds uint32 token-ID range");
  uint64_t id_bytes = product(rows, 4, "IDs");
  uint64_t cells = product(rows, vocab, "logits");
  uint64_t logit_bytes = product(cells, 4, "logits");
  uint64_t row_bytes64 = product(vocab, 4, "logit row");
  if (row_bytes64 > SIZE_MAX)
    invalid("logit row exceeds addressable size");
  size_t row_bytes = (size_t)row_bytes64;
  input ref_ids = open_input(argv[1], id_bytes);
  input got_ids = open_input(argv[2], id_bytes);
  input ref_logits = open_input(argv[3], logit_bytes);
  input got_logits = open_input(argv[4], logit_bytes);
  unsigned char *ref_row = malloc(row_bytes), *got_row = malloc(row_bytes);
  if (!ref_row || !got_row)
    invalid("cannot allocate logit rows");
  int fd = open(argv[7], O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    invalid("cannot create new report %s: %s", argv[7], strerror(errno));
  created_report = argv[7];
  report = fdopen(fd, "w");
  if (!report) {
    close(fd);
    invalid("cannot open report stream");
  }
  emit("{\n\"schema\":\"flash-sass.qsa-logits-numerical.v1\",\n"
       "\"comparison_scope\":\"supplied teacher-forced arrays only\",\n"
       "\"independent_checkpoint_evidence\":false,\n"
       "\"provenance_status\":\"not_verified\",\n"
       "\"byte_order\":\"little\",\n"
       "\"rows\":%" PRIu64 ",\"vocab\":%" PRIu64 ",\n"
       "\"limits\":{\"max_abs\":0.10,\"normalized_rms\":0.01,"
       "\"rms_denominator_floor\":1e-8,\"relative_denominator_floor\":1e-8,"
       "\"scope\":\"each row and aggregate\"},\n\"per_row\":[\n",
       rows, vocab);
  metrics aggregate = {0};
  uint64_t matching_top1 = 0, failing_rows = 0;
  double max_row_nrms = 0;
  for (uint64_t t = 0; t < rows; ++t) {
    unsigned char ref_id_bytes[4], got_id_bytes[4];
    read_exact(&ref_ids, ref_id_bytes, sizeof ref_id_bytes);
    read_exact(&got_ids, got_id_bytes, sizeof got_id_bytes);
    uint32_t ref_id = u32le(ref_id_bytes), got_id = u32le(got_id_bytes);
    if (ref_id != got_id)
      invalid("input IDs differ at row %" PRIu64 ": %" PRIu32
              " versus %" PRIu32,
              t, ref_id, got_id);
    if ((uint64_t)ref_id >= vocab)
      invalid("input ID %" PRIu32 " outside vocabulary at row %" PRIu64, ref_id,
              t);
    read_exact(&ref_logits, ref_row, row_bytes);
    read_exact(&got_logits, got_row, row_bytes);
    metrics row = {0};
    uint64_t ref_top = 0, got_top = 0;
    double ref_best = -INFINITY, got_best = -INFINITY;
    for (size_t j = 0; j < (size_t)vocab; ++j) {
      double ref = f32le(ref_row + 4 * j), got = f32le(got_row + 4 * j);
      if (!isfinite(ref) || !isfinite(got))
        invalid("nonfinite logit at row %" PRIu64 ", token %zu", t, j);
      observe(&row, ref, got);
      if (ref > ref_best) {
        ref_best = ref;
        ref_top = (uint64_t)j;
      }
      if (got > got_best) {
        got_best = got;
        got_top = (uint64_t)j;
      }
    }
    aggregate.error_sq += row.error_sq;
    aggregate.reference_sq += row.reference_sq;
    aggregate.count += row.count;
    if (row.max_abs > aggregate.max_abs)
      aggregate.max_abs = row.max_abs;
    if (row.max_relative > aggregate.max_relative)
      aggregate.max_relative = row.max_relative;
    double row_nrms = nrms(&row);
    if (row_nrms > max_row_nrms)
      max_row_nrms = row_nrms;
    int pass = passes(&row);
    if (!pass)
      ++failing_rows;
    if (ref_top == got_top)
      ++matching_top1;
    emit("%s{\"position\":%" PRIu64 ",\"input_id\":%" PRIu32 ",",
         t ? ",\n" : "", t, ref_id);
    emit_metrics(&row);
    emit(",\"reference_top1\":%" PRIu64 ",\"candidate_top1\":%" PRIu64
         ",\"top1_equal\":%s,\"limits_pass\":%s}",
         ref_top, got_top, ref_top == got_top ? "true" : "false",
         pass ? "true" : "false");
  }
  finish_input(&ref_ids);
  finish_input(&got_ids);
  finish_input(&ref_logits);
  finish_input(&got_logits);
  int pass = !failing_rows && passes(&aggregate);
  emit("\n],\n\"aggregate\":{");
  emit_metrics(&aggregate);
  emit(",\"max_row_normalized_rms\":%.17g,\"matching_top1_rows\":%" PRIu64
       ",\"failing_rows\":%" PRIu64 "},\n\"numerical_limits_pass\":%s,\n"
       "\"complete\":true\n}\n",
       max_row_nrms, matching_top1, failing_rows, pass ? "true" : "false");
  FILE *completed = report;
  report = NULL;
  if (fclose(completed))
    invalid("report close failed: %s", strerror(errno));
  created_report = NULL;
  free(ref_row);
  free(got_row);
  fprintf(stderr,
          "%s: %" PRIu64
          " rows; max_abs=%.9g normalized_rms=%.9g; provenance unverified\n",
          pass ? "NUMERICAL_PASS" : "NUMERICAL_FAIL", rows, aggregate.max_abs,
          nrms(&aggregate));
  return pass ? 0 : 3;
}
