/* C-only clients and numerical checks for flashvis. No GPU/FA opt-out. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <openssl/evp.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(float) == 4 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
               "F32 little-endian protocol");
static pid_t child;
static char socket_path[100];
static char pending_output[4096];
static volatile sig_atomic_t stop_signal;
static double transfer_deadline;
static double now(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t))
    abort();
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void stop(void) {
  if (child > 0) {
    kill(-child, SIGTERM);
    double end = now() + 5;
    int status;
    for (;;) {
      pid_t reaped = waitpid(child, &status, WNOHANG);
      if (reaped == child || (reaped < 0 && errno == ECHILD))
        break;
      if (reaped < 0 && errno != EINTR)
        break;
      if (now() > end) {
        kill(-child, SIGKILL);
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
        break;
      }
      usleep(10000);
    }
    child = 0;
  }
  if (*socket_path)
    unlink(socket_path);
  if (*pending_output) {
    unlink(pending_output);
    *pending_output = 0;
  }
}
static void interrupted(int sig) { stop_signal = sig; }
static void check_interrupt(void) {
  if (stop_signal)
    exit(128 + stop_signal);
}
static void fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}
static void quote(const char *s) {
  putchar('"');
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    if (*p == '"' || *p == '\\')
      printf("\\%c", *p);
    else if (*p < 32)
      printf("\\u%04x", *p);
    else
      putchar(*p);
  }
  putchar('"');
}
static void wait_io(int fd, short events) {
  for (;;) {
    check_interrupt();
    if (transfer_deadline && now() >= transfer_deadline)
      fail("socket transfer deadline");
    struct pollfd p = {.fd = fd, .events = events};
    int ready = poll(&p, 1, 100);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      fail("poll: %s", strerror(errno));
    }
    if (p.revents & POLLNVAL)
      fail("invalid transfer descriptor");
    if (ready > 0 && (p.revents & (events | POLLHUP | POLLERR)))
      return;
  }
}
static void full_write(int fd, const void *buf, size_t n) {
  const unsigned char *p = buf;
  while (n) {
    wait_io(fd, POLLOUT);
    ssize_t k = write(fd, p, n);
    if (k < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (k <= 0)
      fail("write: %s", strerror(errno));
    p += k;
    n -= (size_t)k;
  }
}
static void full_read(int fd, void *buf, size_t n) {
  unsigned char *p = buf;
  while (n) {
    wait_io(fd, POLLIN);
    ssize_t k = read(fd, p, n);
    if (k < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (k <= 0)
      fail("short/failed read: %s", k < 0 ? strerror(errno) : "EOF");
    p += k;
    n -= (size_t)k;
  }
}
static void line_read(int fd, char *line, size_t cap) {
  size_t n = 0;
  for (;;) {
    char c;
    full_read(fd, &c, 1);
    if (c == '\n')
      break;
    if (!c || n + 1 >= cap)
      fail("invalid protocol line");
    line[n++] = c;
  }
  line[n] = 0;
}
static EVP_MD_CTX *digest(void) {
  EVP_MD_CTX *c = EVP_MD_CTX_new();
  if (!c || EVP_DigestInit_ex(c, EVP_sha256(), NULL) != 1)
    fail("SHA256 init");
  return c;
}
static void digest_end(EVP_MD_CTX *c, char hex[65]) {
  unsigned char b[EVP_MAX_MD_SIZE];
  unsigned n = 0;
  if (EVP_DigestFinal_ex(c, b, &n) != 1 || n != 32)
    fail("SHA256 final");
  EVP_MD_CTX_free(c);
  for (unsigned i = 0; i < 32; i++)
    sprintf(hex + 2 * i, "%02x", b[i]);
}
static void hash_memory(const void *p, size_t n, char hex[65]) {
  EVP_MD_CTX *c = digest();
  if (EVP_DigestUpdate(c, p, n) != 1)
    fail("SHA256 update");
  digest_end(c, hex);
}
static void hash_file(const char *path, char hex[65]) {
  FILE *f = fopen(path, "rb");
  if (!f)
    fail("open %s: %s", path, strerror(errno));
  EVP_MD_CTX *c = digest();
  unsigned char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)))
    if (EVP_DigestUpdate(c, buf, n) != 1)
      fail("SHA256 update");
  if (ferror(f) || fclose(f))
    fail("read %s", path);
  digest_end(c, hex);
}
struct image {
  unsigned char *bytes;
  size_t size;
  unsigned w, h;
};
static void le32(unsigned char *p, uint32_t v) {
  for (unsigned i = 0; i < 4; i++)
    p[i] = (unsigned char)(v >> (8 * i));
}
static struct image bitmap(unsigned w, unsigned h, int stripes) {
  uint64_t stride = ((uint64_t)w * 3 + 3) / 4 * 4, size = 54 + stride * h;
  if (!w || !h || size > (64u << 20))
    fail("fixture dimensions too large");
  struct image im = {
      .bytes = calloc(1, (size_t)size), .size = (size_t)size, .w = w, .h = h};
  if (!im.bytes)
    fail("allocate fixture");
  unsigned char *p = im.bytes;
  p[0] = 'B';
  p[1] = 'M';
  le32(p + 2, (uint32_t)size);
  le32(p + 10, 54);
  le32(p + 14, 40);
  le32(p + 18, w);
  le32(p + 22, h);
  p[26] = 1;
  p[28] = 24;
  le32(p + 34, (uint32_t)(size - 54));
  if (!stripes)
    memset(p + 54, 0x80, (size_t)size - 54);
  else
    for (unsigned y = 0; y < h; y++)
      for (unsigned x = 0; x < w; x++) {
        size_t off = 54 + (size_t)y * (size_t)stride + (size_t)x * 3;
        p[off] = (unsigned char)(x % 251);
        p[off + 1] = (unsigned char)((x / 16) % 251);
        p[off + 2] = 127;
      }
  return im;
}
static struct image load_image(const char *arg) {
  unsigned w, h;
  char extra;
  if (sscanf(arg, "%ux%u%c", &w, &h, &extra) == 2)
    return bitmap(w, h, 1);
  struct stat st;
  if (stat(arg, &st) || st.st_size <= 0 || st.st_size > (64 << 20))
    fail("invalid image file %s", arg);
  struct image im = {.size = (size_t)st.st_size};
  im.bytes = malloc(im.size);
  if (!im.bytes)
    fail("allocate image");
  int f = open(arg, O_RDONLY);
  if (f < 0)
    fail("open image %s", arg);
  full_read(f, im.bytes, im.size);
  close(f);
  return im;
}
static const char *run_dir(void) {
  const char *d = getenv("GPUQ_RUN_DIR");
  if (!d || !getenv("GPUQ_JOB_ID"))
    fail("run GPU clients through gpuq");
  return d;
}
static void path_join(char *out, size_t cap, const char *dir,
                      const char *name) {
  int n = snprintf(out, cap, "%s/%s", dir, name);
  if (n < 0 || (size_t)n >= cap)
    fail("path too long");
}
static void start(const char *binary, const char *projector, const char *model,
                  const char *device, int hidden, const char *fault,
                  const char *mode, const char *log) {
  if (child)
    fail("worker already running");
  snprintf(socket_path, sizeof socket_path, "/tmp/vision-check-%ld.sock",
           (long)getpid());
  unlink(socket_path);
  sigset_t blocked, saved;
  sigemptyset(&blocked);
  sigaddset(&blocked, SIGINT);
  sigaddset(&blocked, SIGTERM);
  sigaddset(&blocked, SIGHUP);
  check_interrupt();
  if (sigprocmask(SIG_BLOCK, &blocked, &saved))
    fail("block startup signals");
  child = fork();
  if (child < 0)
    fail("fork");
  if (!child) {
    if (setpgid(0, 0))
      _exit(127);
    struct sigaction normal = {.sa_handler = SIG_DFL};
    sigemptyset(&normal.sa_mask);
    if (sigaction(SIGINT, &normal, NULL) || sigaction(SIGTERM, &normal, NULL) ||
        sigaction(SIGHUP, &normal, NULL) ||
        sigprocmask(SIG_SETMASK, &saved, NULL))
      _exit(127);
    setenv("FLASHVIS_DEVICE", device, 1);
    if (hidden)
      setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
    if (fault)
      setenv("LD_PRELOAD", fault, 1);
    else
      unsetenv("LD_PRELOAD");
    if (mode) {
      char marker[4096], name[100];
      snprintf(name, sizeof name, "%s.once", mode);
      path_join(marker, sizeof marker, run_dir(), name);
      setenv("FLASHVIS_TEST_FAULT", mode, 1);
      setenv("FLASHVIS_TEST_FAULT_ONCE", marker, 1);
    } else {
      unsetenv("FLASHVIS_TEST_FAULT");
      unsetenv("FLASHVIS_TEST_FAULT_ONCE");
    }
    if (log) {
      int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (fd < 0 || dup2(fd, 2) < 0)
        _exit(127);
      close(fd);
    }
    execl(binary, binary, projector, model, socket_path, "10", (char *)NULL);
    _exit(127);
  }
  if (setpgid(child, child) && errno != EACCES && errno != ESRCH)
    fail("set worker process group");
  if (sigprocmask(SIG_SETMASK, &saved, NULL))
    fail("restore startup signals");
  check_interrupt();
}
static int exited(double seconds) {
  double end = now() + seconds;
  int status;
  for (;;) {
    check_interrupt();
    pid_t p = waitpid(child, &status, WNOHANG);
    if (p == child) {
      child = 0;
      return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    }
    if (p < 0)
      fail("waitpid");
    if (now() > end)
      fail("startup exit deadline");
    usleep(25000);
  }
}
static void ready(void) {
  double end = now() + 120;
  struct stat st;
  int status;
  for (;;) {
    check_interrupt();
    if (!stat(socket_path, &st) && S_ISSOCK(st.st_mode))
      return;
    pid_t p = waitpid(child, &status, WNOHANG);
    if (p == child) {
      child = 0;
      fail("encoder exited before listening");
    }
    if (p < 0 || now() > end)
      fail("encoder startup deadline");
    usleep(25000);
  }
}
struct response {
  unsigned tokens, nx, ny, npos, dim;
  char error[512], sha[65];
};
static struct response request(const struct image *im, const char *output,
                               unsigned seconds) {
  struct response r = {0};
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd < 0)
    fail("socket");
  struct timeval tv = {.tv_sec = seconds};
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) ||
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv))
    fail("socket timeout");
  struct sockaddr_un sa = {.sun_family = AF_UNIX};
  strcpy(sa.sun_path, socket_path);
  transfer_deadline = now() + seconds;
  check_interrupt();
  if (connect(fd, (struct sockaddr *)&sa, sizeof sa)) {
    if (errno != EINPROGRESS)
      fail("connect: %s", strerror(errno));
    wait_io(fd, POLLOUT);
    int error = 0;
    socklen_t length = sizeof error;
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) || error)
      fail("connect: %s", strerror(error ? error : errno));
  }
  char line[512];
  int n = snprintf(line, sizeof line, "enc %zu\n", im->size);
  full_write(fd, line, (size_t)n);
  full_write(fd, im->bytes, im->size);
  line_read(fd, line, sizeof line);
  if (!strncmp(line, "err ", 4)) {
    strcpy(r.error, line);
    close(fd);
    transfer_deadline = 0;
    return r;
  }
  char extra;
  if (sscanf(line, "ok %u %u %u %u %u%c", &r.tokens, &r.nx, &r.ny, &r.npos,
             &r.dim, &extra) != 5 ||
      !r.tokens || r.tokens > 16384 || !r.nx || !r.ny || r.dim != 2560 ||
      (uint64_t)r.nx * r.ny != r.tokens ||
      r.npos != (r.nx > r.ny ? r.nx : r.ny))
    fail("invalid response header: %s", line);
  if (im->w && im->w % 32 == 0 && im->h % 32 == 0 &&
      (uint64_t)im->w * im->h >= 65536 &&
      (r.nx != im->w / 32 || r.ny != im->h / 32))
    fail("fixture resolution changed");
  if (output) {
    int size = snprintf(pending_output, sizeof pending_output, "%s.tmp-%ld",
                        output, (long)getpid());
    if (size < 0 || (size_t)size >= sizeof pending_output)
      fail("temporary output path too long");
  }
  FILE *file = output ? fopen(pending_output, "wb") : NULL;
  if (output && !file)
    fail("output %s", output);
  float *buf = malloc(1 << 20);
  if (!buf)
    fail("allocate response chunk");
  EVP_MD_CTX *md = digest();
  size_t left = (size_t)r.tokens * r.dim;
  while (left) {
    size_t k = left > (1u << 18) ? (1u << 18) : left;
    full_read(fd, buf, k * 4);
    for (size_t i = 0; i < k; i++)
      if (!isfinite(buf[i]))
        fail("non-finite embedding");
    if (file && fwrite(buf, 4, k, file) != k)
      fail("write output");
    if (EVP_DigestUpdate(md, buf, k * 4) != 1)
      fail("SHA256 update");
    left -= k;
  }
  free(buf);
  digest_end(md, r.sha);
  uint32_t pos[3];
  for (unsigned i = 0; i < r.tokens; i++) {
    full_read(fd, pos, 12);
    if (pos[0] || pos[1] != i / r.nx || pos[2] != i % r.nx)
      fail("M-RoPE mismatch at %u", i);
  }
  if (file && fclose(file))
    fail("close output");
  if (output) {
    if (rename(pending_output, output))
      fail("publish validated output");
    *pending_output = 0;
  }
  close(fd);
  transfer_deadline = 0;
  return r;
}
static char *read_text(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    fail("open log %s", path);
  if (fseek(f, 0, SEEK_END))
    fail("seek log");
  long n = ftell(f);
  if (n < 0 || n > 16 * 1024 * 1024)
    fail("log size");
  rewind(f);
  char *s = calloc((size_t)n + 1, 1);
  if (!s || fread(s, 1, (size_t)n, f) != (size_t)n)
    fail("read log");
  fclose(f);
  return s;
}
static void encode(int argc, char **argv) {
  if (argc < 4)
    fail("encode BINARY PROJECTOR TEXT IMAGE|WIDTHxHEIGHT ...");
  const char *dir = run_dir();
  start(argv[0], argv[1], argv[2], "auto", 0, NULL, NULL, NULL);
  ready();
  for (int i = 3; i < argc; i++) {
    struct image im = load_image(argv[i]);
    char path[4096], name[60], sha[65];
    snprintf(name, sizeof name, "embeddings-%d.bin", i - 3);
    path_join(path, sizeof path, dir, name);
    hash_memory(im.bytes, im.size, sha);
    double t = now();
    struct response r = request(&im, path, 3600);
    if (*r.error)
      fail("%s", r.error);
    printf("{\"event\":\"vision_encode_passed\",\"input\":");
    quote(argv[i]);
    printf(",\"dimensions\":[%u,%u,%u,%u,%u],\"input_sha256\":\"%s\","
           "\"embeddings_sha256\":\"%s\",\"roundtrip_seconds\":%.9f}\n",
           r.tokens, r.nx, r.ny, r.npos, r.dim, sha, r.sha, now() - t);
    free(im.bytes);
  }
  stop();
}
static void recover(int argc, char **argv) {
  if (argc != 5)
    fail("recover BINARY PROJECTOR TEXT IMAGE FAULT_SO");
  const char *dir = run_dir();
  struct image image = load_image(argv[3]);
  char fault[4096];
  if (!realpath(argv[4], fault))
    fail("fault path");
  const char *bad[] = {"cpu", "not-a-real-device", "auto"};
  for (unsigned i = 0; i < 3; i++) {
    char path[4096], name[80];
    snprintf(name, sizeof name, "invalid-%u.diag", i);
    path_join(path, sizeof path, dir, name);
    start(argv[0], argv[1], argv[2], bad[i], i == 2, NULL, NULL, path);
    int code = exited(120);
    char *s = read_text(path);
    if (code != 2 || (!strstr(s, "not an available GPU") &&
                      !strstr(s, "GPU Flash Attention is required")))
      fail("invalid GPU startup was not rejected");
    free(s);
    printf("{\"event\":\"vision_failure_check\",\"case\":\"unavailable-%s\","
           "\"exit\":2}\n",
           bad[i]);
  }
  const char *modes[] = {"malformed", "oversized", "error", "nan",
                         "native-abort"};
  const char *errors[] = {
      "err cannot decode the image",
      "err image exceeds the model limit of 16777216 pixels",
      "err required GPU Flash Attention did not execute",
      "err encoder returned invalid embeddings",
      "err GPU encoder worker terminated"};
  for (unsigned i = 0; i < 5; i++) {
    char path[4096], name[100];
    snprintf(name, sizeof name, "%s.diag", modes[i]);
    path_join(path, sizeof path, dir, name);
    start(argv[0], argv[1], argv[2], "auto", 0, i < 2 ? NULL : fault, modes[i],
          path);
    ready();
    struct image input = image, following = image;
    int allocated = 0;
    if (i == 0) {
      input.bytes = (unsigned char *)"not an image";
      input.size = 12;
      input.w = input.h = 0;
    }
    if (i == 1) {
      input = bitmap(4097, 4096, 0);
      allocated = 1;
    }
    if (i == 4) {
      input = bitmap(256, 256, 0);
      following = input;
      allocated = 1;
    }
    struct response rejected = request(&input, NULL, 120);
    if (strcmp(rejected.error, errors[i]))
      fail("wrong %s rejection: %s", modes[i], rejected.error);
    if (i == 4) {
      char marker[4096];
      path_join(marker, sizeof marker, dir, "native-abort.once");
      if (access(marker, F_OK))
        fail("native fault did not fire");
    }
    struct response accepted = request(&following, NULL, 120);
    if (*accepted.error)
      fail("following request: %s", accepted.error);
    stop();
    char *s = read_text(path);
    if (!strstr(s, "observed_gpu=27") || !strstr(s, "backend=CUDA") ||
        strstr(s, "backend CPU") || strstr(s, "switching to CPU"))
      fail("GPU FA diagnostic check");
    free(s);
    printf("{\"event\":\"vision_failure_check\",\"case\":\"%s\",\"rejection\":",
           modes[i]);
    quote(rejected.error);
    printf(",\"subsequent_dimensions\":[%u,%u,%u,%u,%u],\"subsequent_backend\":"
           "\"GPU FA\"}\n",
           accepted.tokens, accepted.nx, accepted.ny, accepted.npos,
           accepted.dim);
    if (allocated)
      free(input.bytes);
  }
  free(image.bytes);
}
static void compare(const char *a, const char *b) {
  struct stat sa_info, sb_info;
  if (stat(a, &sa_info) || stat(b, &sb_info) || sa_info.st_size <= 0 ||
      sa_info.st_size != sb_info.st_size || sa_info.st_size % (2560 * 4))
    fail("comparison file dimensions differ");
  FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
  if (!fa || !fb)
    fail("open comparison inputs");
  float x[2560], y[2560];
  double dot = 0, aa = 0, bb = 0, error = 0, maximum = 0, mincos = 1,
         maxrel = 0;
  size_t rows = 0;
  for (;;) {
    size_t na = fread(x, 4, 2560, fa), nb = fread(y, 4, 2560, fb);
    if (!na && !nb)
      break;
    if (na != 2560 || nb != 2560)
      fail("comparison dimension mismatch");
    double rd = 0, ra = 0, rb = 0, re = 0;
    for (unsigned i = 0; i < 2560; i++) {
      double u = x[i], v = y[i], d = u - v;
      if (!isfinite(u) || !isfinite(v))
        fail("non-finite comparison");
      rd += u * v;
      ra += u * u;
      rb += v * v;
      re += d * d;
      if (fabs(d) > maximum)
        maximum = fabs(d);
    }
    if (ra <= 0 || rb <= 0)
      fail("zero norm embedding");
    double c = rd / sqrt(ra * rb), r = sqrt(re / ra);
    if (c < mincos)
      mincos = c;
    if (r > maxrel)
      maxrel = r;
    dot += rd;
    aa += ra;
    bb += rb;
    error += re;
    rows++;
  }
  if (ferror(fa) || ferror(fb) || !rows)
    fail("invalid comparison file");
  fclose(fa);
  fclose(fb);
  char sa[65], sb[65];
  hash_file(a, sa);
  hash_file(b, sb);
  printf("{\"event\":\"embedding_numerical_validation\",\"tokens\":%zu,"
         "\"global_cosine\":%.15g,\"global_relative_l2\":%.15g,\"max_abs_"
         "error\":%.15g,\"min_token_cosine\":%.15g,\"max_token_relative_l2\":%."
         "15g,\"reference_sha256\":\"%s\",\"gpu_sha256\":\"%s\"}\n",
         rows, dot / sqrt(aa * bb), sqrt(error / aa), maximum, mincos, maxrel,
         sa, sb);
  if (mincos < 0.999 || maxrel > 0.02)
    fail("numerical tolerance exceeded");
}
static void retained(const char *dir) {
  const unsigned tokens[] = {602, 4096, 16384};
  const char *hashes[] = {
      "d9b7be627eb9828045805efc651e373127d29850acb11152a7d4c188f1beebab",
      "eb0a3449d67a8265c93ccea9b4206f2af4a2b0c9098e961ac908b8fc6d0c188f",
      "807b537caf63c6e4e9c898e544e36b199f332814e435070f411cc31eb91ba93a"};
  float buf[16384];
  for (unsigned i = 0; i < 3; i++) {
    char path[4096], name[40], sha[65];
    snprintf(name, sizeof name, "embeddings-%u.bin", i);
    path_join(path, sizeof path, dir, name);
    struct stat st;
    if (stat(path, &st) ||
        (uint64_t)st.st_size != (uint64_t)tokens[i] * 2560 * 4)
      fail("retained size mismatch");
    hash_file(path, sha);
    if (strcmp(sha, hashes[i]))
      fail("retained hash mismatch");
    FILE *f = fopen(path, "rb");
    if (!f)
      fail("retained open");
    size_t n, count = 0;
    while ((n = fread(buf, 4, 16384, f))) {
      for (size_t j = 0; j < n; j++)
        if (!isfinite(buf[j]))
          fail("non-finite retained value");
      count += n;
    }
    if (ferror(f))
      fail("retained read");
    fclose(f);
    printf("{\"event\":\"retained_embeddings_verified\",\"tokens\":%u,\"finite_"
           "values\":%zu,\"bytes\":%llu,\"sha256\":\"%s\"}\n",
           tokens[i], count, (unsigned long long)st.st_size, sha);
  }
}
int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  atexit(stop);
  signal(SIGPIPE, SIG_IGN);
  struct sigaction action = {.sa_handler = interrupted};
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, NULL) || sigaction(SIGINT, &action, NULL) ||
      sigaction(SIGHUP, &action, NULL))
    fail("install signal handlers");
  if (argc < 2)
    fail("vision_check encode|recover|compare|retained|fixture ...");
  if (!strcmp(argv[1], "encode"))
    encode(argc - 2, argv + 2);
  else if (!strcmp(argv[1], "recover"))
    recover(argc - 2, argv + 2);
  else if (!strcmp(argv[1], "compare") && argc == 4)
    compare(argv[2], argv[3]);
  else if (!strcmp(argv[1], "retained") && argc == 3)
    retained(argv[2]);
  else if (!strcmp(argv[1], "fixture") && (argc == 3 || argc == 4)) {
    struct image im = load_image(argv[2]);
    if (argc == 4) {
      FILE *f = fopen(argv[3], "wb");
      if (!f || fwrite(im.bytes, 1, im.size, f) != im.size || fclose(f))
        fail("fixture output");
    }
    char sha[65];
    hash_memory(im.bytes, im.size, sha);
    const char *expected = NULL;
    if (im.w == 2048 && im.h == 2048)
      expected =
          "dd6386bf409f18a4f6e2012358462816d241949e3796274a952e2b021a1f0fc4";
    if (im.w == 4096 && im.h == 4096)
      expected =
          "18f2a50c6601fa49201838172663ccfffa1675a2eb8ed3236ac071205d3e1464";
    if (expected && strcmp(sha, expected))
      fail("fixture differs from the recorded full-resolution input");
    printf("{\"event\":\"fixture_verified\",\"width\":%u,\"height\":%u,"
           "\"bytes\":%zu,\"sha256\":\"%s\"}\n",
           im.w, im.h, im.size, sha);
    free(im.bytes);
  } else
    fail("invalid command or arguments");
  check_interrupt();
  return 0;
}
