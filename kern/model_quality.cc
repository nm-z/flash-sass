#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include "json.h"
}
namespace fs = std::filesystem;
using S = std::string;
using U = uint32_t;
using D = std::vector<double>;
static arena *A;
static const char *SCHEMA = "flash-sass.frozen-logits.v1";
static void need(bool x, const S &s) {
  if (!x)
    throw std::runtime_error(s);
}
static jv *obj() { return jv_new(A, J_DICT); }
static jv *list() { return jv_new(A, J_LIST); }
static jv *str(const S &s) { return jv_str(A, s.data(), s.size()); }
static jv *num(double x) {
  if (!std::isfinite(x))
    return jv_new(A, J_NULL);
  jv *v = jv_new(A, J_NUM);
  v->d = x;
  return v;
}
static void set(jv *v, const char *k, jv *x) { jv_set(A, v, k, strlen(k), x); }
static void ns(jv *v, const char *k, double x) { set(v, k, num(x)); }
static void ss(jv *v, const char *k, const S &s) { set(v, k, str(s)); }
static void bs(jv *v, const char *k, bool x) { set(v, k, jv_bool(A, x)); }
static jv *get(const jv *v, const char *k) {
  return v && (v->k == J_DICT || v->k == J_NS) ? jv_getz(v, k) : nullptr;
}
static S text(const jv *v) {
  return v && v->k == J_STR ? S(v->s.s, v->s.n) : S();
}
static bool yes(const jv *v) { return v && v->k == J_BOOL && v->b; }
static uint64_t integer(const jv *v) {
  need(v && v->k == J_INT && v->i > 0, "invalid positive integer");
  return uint64_t(v->i);
}
static jv *size_value(uint64_t n) {
  need(n <= LLONG_MAX, "size exceeds JSON integer range");
  return jv_int(A, (long long)n);
}
static S encode(jv *v) {
  sbuf b = {};
  json_dump(&b, v, 0);
  S s(b.s, b.n);
  sb_free(&b);
  return s;
}
static jv *parse(const S &s) {
  const char *e = nullptr;
  jv *v = json_parse(A, s.data(), s.size(), &e);
  need(v, "invalid JSON");
  return v;
}
static jv *load(const fs::path &p) {
  std::ifstream f(p);
  need(bool(f), "cannot open " + p.string());
  std::ostringstream s;
  s << f.rdbuf();
  return parse(s.str());
}
static void save(const fs::path &p, jv *v) {
  std::ofstream f(p);
  need(bool(f), "cannot write " + p.string());
  f << encode(v) << '\n';
  need(bool(f), "write failed " + p.string());
}
static S sha(const fs::path &p) {
  FILE *f = fopen(p.c_str(), "rb");
  need(f, "cannot hash " + p.string());
  EVP_MD_CTX *c = EVP_MD_CTX_new();
  if (!c) {
    fclose(f);
    throw std::runtime_error("SHA allocation");
  }
  bool ok = EVP_DigestInit_ex(c, EVP_sha256(), nullptr) == 1;
  unsigned char b[65536], out[EVP_MAX_MD_SIZE];
  size_t n;
  while (ok && (n = fread(b, 1, sizeof b, f)))
    ok = EVP_DigestUpdate(c, b, n) == 1;
  ok = ok && !ferror(f);
  unsigned len = 0;
  if (ok)
    ok = EVP_DigestFinal_ex(c, out, &len) == 1;
  EVP_MD_CTX_free(c);
  fclose(f);
  need(ok && len == 32, "SHA failed");
  std::ostringstream s;
  for (unsigned i = 0; i < len; i++)
    s << std::hex << std::setw(2) << std::setfill('0') << unsigned(out[i]);
  return s.str();
}
static bool hex(const S &s, size_t n) {
  return s.size() == n &&
         s.find_first_not_of("0123456789abcdefABCDEF") == S::npos;
}
static bool revision(const S &s) { return hex(s, 40) || hex(s, 64); }
static S lower(S s) {
  for (char &c : s)
    if (c >= 'A' && c <= 'F')
      c += 32;
  return s;
}
static jv *missing(jv *m, bool ref) {
  jv *r = list();
  auto ck = [&](bool b, const char *s) {
    if (!b)
      jv_push(A, r, str(s));
  };
  ck(yes(get(m, "teacher_forced")), "teacher_forced=true");
  auto b = get(m, "base_model");
  ck(!text(get(b, "id")).empty(), "base_model.id");
  ck(revision(text(get(b, "revision"))),
     "base_model.revision (immutable hash)");
  ck(hex(text(get(get(m, "tokenizer"), "sha256")), 64), "tokenizer.sha256");
  auto files = get(get(m, "checkpoint"), "files");
  ck(files && files->k == J_LIST && files->l.n > 0, "checkpoint.files");
  if (files && files->k == J_LIST)
    for (int i = 0; i < files->l.n; i++) {
      auto f = files->l.a[i];
      ck(hex(text(get(f, "sha256")), 64), "checkpoint.files[].sha256");
      ck(!text(get(f, "path")).empty() || !text(get(f, "source_uri")).empty(),
         "checkpoint.files[].path/source_uri");
    }
  auto imp = get(m, "implementation");
  ck(!text(get(imp, "name")).empty(), "implementation.name");
  ck(revision(text(get(imp, "revision"))),
     "implementation.revision (immutable hash)");
  auto cmd = get(m, "command");
  ck(cmd && ((cmd->k == J_STR && cmd->s.n) || (cmd->k == J_LIST && cmd->l.n)),
     "command");
  if (ref)
    ck(yes(get(imp, "independent_reference")),
       "implementation.independent_reference=true");
  return r;
}
struct Opt {
  std::map<S, S> v;
  bool diagnostic = false;
  S req(const S &k) const {
    auto i = v.find(k);
    need(i != v.end(), "missing --" + k);
    return i->second;
  }
  S val(const S &k, const S &d = "") const {
    auto i = v.find(k);
    return i == v.end() ? d : i->second;
  }
};
static uint64_t uintarg(const S &s) {
  need(!s.empty() && s.find_first_not_of("0123456789") == S::npos,
       "invalid unsigned integer");
  size_t n;
  auto v = std::stoull(s, &n);
  need(n == s.size() && v > 0, "invalid positive integer");
  return v;
}
static double limitarg(const S &s) {
  size_t n;
  double x = std::stod(s, &n);
  need(n == s.size() && std::isfinite(x) && x >= 0,
       "limit must be finite nonnegative");
  return x;
}
static std::vector<U> ids(const fs::path &p) {
  auto n = fs::file_size(p);
  need(n && n % 4 == 0 && n / 4 <= SIZE_MAX, "invalid u32 bytes");
  std::vector<U> x(n / 4);
  FILE *f = fopen(p.c_str(), "rb");
  need(f, "cannot read IDs");
  bool ok = fread(x.data(), 4, x.size(), f) == x.size();
  fclose(f);
  need(ok, "short IDs read");
  return x;
}
static void checkshape(uint64_t T, uint64_t V, const fs::path &p) {
  need(T && V && V <= UINT32_MAX && T <= uint64_t(LLONG_MAX) / 4 / V,
       "capture dimensions overflow");
  need(fs::file_size(p) == T * V * 4, "logits shape/byte mismatch");
}
struct Capture {
  fs::path dir;
  jv *m;
  uint64_t T, V;
  std::vector<U> id, vocab;
  std::vector<size_t> order;
};
static Capture capture(const S &dir, const S &kind) {
  Capture c;
  c.dir = dir;
  c.m = load(c.dir / "manifest.json");
  need(c.m->k == J_DICT, "manifest must be object");
  auto metadata = get(c.m, "metadata");
  need(!metadata || metadata->k == J_DICT, "metadata must be object");
  need(text(get(c.m, "schema")) == SCHEMA && text(get(c.m, "kind")) == kind,
       "wrong schema/kind");
  need(text(get(c.m, "dtype")) == "<f4" &&
           text(get(c.m, "byte_order")) == "little",
       "wrong dtype/byte order");
  c.T = integer(get(c.m, "positions"));
  c.V = integer(get(c.m, "vocab_size"));
  auto ar = get(c.m, "artifacts");
  need(ar && ar->k == J_DICT, "missing artifacts");
  for (int i = 0; i < ar->o.n; i++) {
    S name(ar->o.key[i], ar->o.klen[i]);
    need(name != "." && name != ".." && fs::path(name).filename() == name,
         "artifact must be plain filename");
    auto w = ar->o.val[i];
    auto p = c.dir / name;
    auto bytes = get(w, "bytes");
    need(bytes && bytes->k == J_INT && bytes->i >= 0, "invalid artifact bytes");
    need(fs::file_size(p) == uint64_t(bytes->i) &&
             sha(p) == text(get(w, "sha256")),
         "frozen artifact changed: " + p.string());
  }
  for (auto n : {"ids.u32", "logits.f32", "vocab_ids.u32"})
    need(get(ar, n), S("missing artifact ") + n);
  checkshape(c.T, c.V, c.dir / "logits.f32");
  c.id = ids(c.dir / "ids.u32");
  c.vocab = ids(c.dir / "vocab_ids.u32");
  need(c.id.size() == c.T && c.vocab.size() == c.V, "capture shape mismatch");
  c.order.resize(c.V);
  std::iota(c.order.begin(), c.order.end(), 0);
  std::sort(c.order.begin(), c.order.end(),
            [&](size_t a, size_t b) { return c.vocab[a] < c.vocab[b]; });
  for (size_t i = 1; i < c.V; i++)
    need(c.vocab[c.order[i - 1]] != c.vocab[c.order[i]],
         "duplicate vocabulary IDs");
  return c;
}
static void emit(jv *r, const Opt &o) {
  if (!o.val("out").empty())
    save(o.val("out"), r);
  std::cout << encode(r) << '\n';
}
static int freeze(const Opt &o) {
  auto id = ids(o.req("ids"));
  uint64_t V = uintarg(o.req("vocab-size"));
  checkshape(id.size(), V, o.req("logits"));
  S kind = o.req("kind");
  need(kind == "candidate" || kind == "reference", "invalid kind");
  std::vector<U> v;
  if (o.val("vocab-ids").empty()) {
    v.resize(V);
    std::iota(v.begin(), v.end(), 0);
  } else
    v = ids(o.val("vocab-ids"));
  need(v.size() == V, "vocabulary shape mismatch");
  auto sorted = v;
  std::sort(sorted.begin(), sorted.end());
  need(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
       "duplicate vocabulary IDs");
  jv *m = o.val("metadata").empty() ? obj() : load(o.val("metadata"));
  need(m->k == J_DICT, "metadata must be object");
  auto mk = get(m, "kind");
  need(!mk || mk->k == J_NULL || text(mk) == kind,
       "metadata kind contradicts --kind");
  auto files = get(get(m, "checkpoint"), "files");
  if (files) {
    need(files->k == J_LIST, "checkpoint.files must be list");
    for (int i = 0; i < files->l.n; i++) {
      auto item = files->l.a[i];
      need(item->k == J_DICT, "checkpoint file must be object");
      S p = text(get(item, "path"));
      bool local = !p.empty() && fs::is_regular_file(p);
      bs(item, "verified_local", local);
      if (local) {
        S h = sha(p), decl = text(get(item, "sha256"));
        need(decl.empty() || lower(decl) == h,
             "checkpoint hash mismatch: " + p);
        ss(item, "sha256", h);
        set(item, "bytes", size_value(fs::file_size(p)));
      }
    }
  }
  jv *raw = nullptr;
  if (!o.val("raw-info").empty()) {
    raw = load(o.val("raw-info"));
    need(yes(get(raw, "teacher_forced")) &&
             integer(get(raw, "positions")) == id.size() &&
             integer(get(raw, "vocab_size")) == V,
         "raw capture metadata mismatch");
  }
  fs::path out = o.req("out");
  need(fs::create_directory(out), "output already exists");
  fs::copy_file(o.req("ids"), out / "ids.u32");
  fs::copy_file(o.req("logits"), out / "logits.f32");
  {
    std::ofstream f(out / "vocab_ids.u32", std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
    need(bool(f), "vocabulary write failed");
  }
  if (raw)
    save(out / "raw_capture_info.json", raw);
  auto man = obj();
  ss(man, "schema", SCHEMA);
  ss(man, "kind", kind);
  set(man, "positions", size_value(id.size()));
  set(man, "vocab_size", size_value(V));
  ss(man, "dtype", "<f4");
  ss(man, "byte_order", "little");
  set(man, "metadata", m);
  std::time_t now = std::time(nullptr);
  std::tm utc;
  gmtime_r(&now, &utc);
  char stamp[32];
  strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &utc);
  ss(man, "created_utc", stamp);
  auto miss = missing(m, false);
  set(man, "missing_provenance", miss);
  ss(man, "provenance_note",
     "Producer declarations and local identities; not execution attestation.");
  auto artifacts = obj();
  for (auto leaf :
       {"ids.u32", "logits.f32", "vocab_ids.u32", "raw_capture_info.json"}) {
    auto p = out / leaf;
    if (!fs::exists(p))
      continue;
    auto a = obj();
    set(a, "bytes", size_value(fs::file_size(p)));
    ss(a, "sha256", sha(p));
    set(artifacts, leaf, a);
  }
  set(man, "artifacts", artifacts);
  save(out / "manifest.json", man);
  auto report = obj();
  ss(report, "capture", out.string());
  ss(report, "manifest_sha256", sha(out / "manifest.json"));
  set(report, "missing_provenance", miss);
  bs(report, "quality_validation", false);
  std::cout << encode(report) << '\n';
  return 0;
}
static D logsoft(const D &x) {
  double m = *std::max_element(x.begin(), x.end()), sum = 0;
  for (double v : x)
    sum += std::exp(v - m);
  double z = std::log(sum);
  D y(x.size());
  for (size_t i = 0; i < x.size(); i++)
    y[i] = (x[i] - m) - z;
  return y;
}
static std::vector<unsigned char> top(const D &x, size_t k) {
  D copy = x;
  std::nth_element(copy.begin(), copy.begin() + (x.size() - k), copy.end());
  double cutoff = copy[x.size() - k];
  std::vector<unsigned char> selected(x.size());
  size_t count = 0;
  for (size_t i = 0; i < x.size(); i++)
    if (x[i] > cutoff) {
      selected[i] = 1;
      count++;
    }
  for (size_t i = 0; i < x.size() && count < k; i++)
    if (x[i] == cutoff) {
      selected[i] = 1;
      count++;
    }
  return selected;
}
struct Rows {
  FILE *f;
  explicit Rows(const fs::path &p) : f(fopen(p.c_str(), "rb")) {
    need(f, "cannot open logits");
  }
  ~Rows() { fclose(f); }
  D read(const Capture &c) {
    std::vector<float> raw(c.V);
    need(fread(raw.data(), 4, c.V, f) == c.V, "short logits read");
    D out(c.V);
    for (size_t i = 0; i < c.V; i++)
      out[i] = raw[c.order[i]];
    return out;
  }
};
static void unattested(jv *r) {
  bs(r, "independent_checkpoint_evidence", false);
  ss(r, "provenance_status", "producer_declared_unattested");
  ss(r, "scope",
     "Numerical comparison of these supplied teacher-forced rows only; no "
     "claim about other prompts/tasks.");
  ss(r, "provenance_note",
     "Producer declarations and hashed files are not execution attestation.");
}
static int compare(const Opt &o) {
  Capture c = capture(o.req("candidate"), "candidate"),
          r = capture(o.req("reference"), "reference");
  need(c.id == r.id, "captures do not have identical teacher-forced token IDs");
  need(c.T == r.T && c.V == r.V, "reference/candidate dimensions differ");
  std::vector<U> v(c.V);
  for (size_t i = 0; i < c.V; i++) {
    v[i] = c.vocab[c.order[i]];
    need(v[i] == r.vocab[r.order[i]], "vocabulary ID sets differ");
  }
  for (U id : c.id)
    need(std::binary_search(v.begin(), v.end(), id),
         "input token IDs absent from vocabulary");
  jv *cm = get(c.m, "metadata"), *rm = get(r.m, "metadata");
  auto miss = obj(), mc = missing(cm, false), mr = missing(rm, true);
  set(miss, "candidate", mc);
  set(miss, "reference", mr);
  bool incomplete = mc->l.n || mr->l.n;
  for (auto spec : {std::pair<const char *, const char *>{"base_model", "id"},
                    {"base_model", "revision"},
                    {"tokenizer", "sha256"}}) {
    S a = text(get(get(cm, spec.first), spec.second)),
      b = text(get(get(rm, spec.first), spec.second));
    need(a.empty() || b.empty() || a == b,
         S("incompatible ") + spec.first + "." + spec.second);
  }
  size_t K = std::min<uint64_t>(uintarg(o.val("top-k", "10")), c.V);
  Rows cr(c.dir / "logits.f32"), rr(r.dir / "logits.f32");
  double sq = 0, abs_sum = 0, ref_sq = 0, center_sq = 0, max_abs = 0,
         max_prob = 0, kl_sum = 0, top1 = 0, overlap = 0, cos_sum = 0,
         ref_nll = 0, can_nll = 0, nll_delta = 0, ref_rank = 0, can_rank = 0;
  size_t cos_count = 0;
  auto detail = list();
  for (size_t t = 0; t < c.T; t++) {
    D x = cr.read(c), y = rr.read(r);
    size_t cn = 0, rn = 0;
    for (size_t i = 0; i < c.V; i++) {
      cn += !std::isfinite(x[i]);
      rn += !std::isfinite(y[i]);
    }
    if (cn || rn) {
      auto report = obj();
      ss(report, "status", "failed_numerical_contract");
      ns(report, "position", t);
      ns(report, "candidate_nonfinite", cn);
      ns(report, "reference_nonfinite", rn);
      bs(report, "quality_pass", false);
      set(report, "missing_provenance", miss);
      unattested(report);
      emit(report, o);
      return 3;
    }
    double xx = 0, yy = 0, xy = 0, center_mean = 0;
    for (size_t i = 0; i < c.V; i++) {
      double d = x[i] - y[i];
      sq += d * d;
      abs_sum += std::abs(d);
      ref_sq += y[i] * y[i];
      max_abs = std::max(max_abs, std::abs(d));
      xx += x[i] * x[i];
      yy += y[i] * y[i];
      xy += x[i] * y[i];
      center_mean += (x[i] - x[0]) - (y[i] - y[0]);
    }
    center_mean /= c.V;
    for (size_t i = 0; i < c.V; i++) {
      double d = ((x[i] - x[0]) - (y[i] - y[0])) - center_mean;
      center_sq += d * d;
    }
    if (xx && yy) {
      cos_sum += xy / std::sqrt(xx * yy);
      cos_count++;
    }
    D lx = logsoft(x), ly = logsoft(y);
    double kl = 0;
    for (size_t i = 0; i < c.V; i++) {
      double px = std::exp(lx[i]), py = std::exp(ly[i]);
      kl += py * (ly[i] - lx[i]);
      max_prob = std::max(max_prob, std::abs(px - py));
    }
    kl = std::max(0., kl);
    kl_sum += kl;
    size_t xp = std::max_element(x.begin(), x.end()) - x.begin(),
           yp = std::max_element(y.begin(), y.end()) - y.begin();
    top1 += xp == yp;
    auto xt = top(x, K), yt = top(y, K);
    size_t common = 0;
    for (size_t i = 0; i < c.V; i++)
      common += xt[i] && yt[i];
    double ov = double(common) / K;
    overlap += ov;
    auto row = obj();
    ns(row, "position", t);
    ns(row, "input_id", c.id[t]);
    ns(row, "candidate_top1", v[xp]);
    ns(row, "reference_top1", v[yp]);
    bs(row, "top1_agrees", xp == yp);
    ns(row, "topk_overlap", ov);
    ns(row, "reference_to_candidate_kl", kl);
    if (t + 1 < c.T) {
      U target = c.id[t + 1];
      size_t j = std::lower_bound(v.begin(), v.end(), target) - v.begin();
      double rnll = -ly[j], cnll = -lx[j];
      ref_nll += rnll;
      can_nll += cnll;
      nll_delta += cnll - rnll;
      size_t rk = 1, ck = 1;
      for (size_t i = 0; i < c.V; i++) {
        rk += y[i] > y[j] || (y[i] == y[j] && v[i] < target);
        ck += x[i] > x[j] || (x[i] == x[j] && v[i] < target);
      }
      ref_rank += rk;
      can_rank += ck;
      ns(row, "next_id", target);
      ns(row, "reference_nll", rnll);
      ns(row, "candidate_nll", cnll);
      ns(row, "nll_increase", cnll - rnll);
      ns(row, "reference_target_rank", rk);
      ns(row, "candidate_target_rank", ck);
    }
    jv_push(A, detail, row);
  }
  auto metrics = obj();
  double N = double(c.T) * c.V;
  ns(metrics, "positions", c.T);
  ns(metrics, "vocabulary", c.V);
  ns(metrics, "logit_rmse", std::sqrt(sq / N));
  ns(metrics, "mean_absolute_logit_error", abs_sum / N);
  ns(metrics, "maximum_absolute_logit_error", max_abs);
  ns(metrics, "relative_l2_logit_error", ref_sq ? std::sqrt(sq / ref_sq) : NAN);
  ns(metrics, "centered_logit_rmse", std::sqrt(center_sq / N));
  ns(metrics, "mean_cosine_similarity", cos_count ? cos_sum / cos_count : NAN);
  ns(metrics, "mean_reference_to_candidate_kl", kl_sum / c.T);
  ns(metrics, "maximum_absolute_probability_error", max_prob);
  ns(metrics, "top1_agreement", top1 / c.T);
  ns(metrics, "topk", K);
  ns(metrics, "mean_topk_overlap", overlap / c.T);
  ns(metrics, "teacher_forced_labels", c.T - 1);
  if (c.T > 1) {
    double rn = ref_nll / (c.T - 1), cn = can_nll / (c.T - 1),
           rp = std::exp(rn), cp = std::exp(cn);
    ns(metrics, "reference_mean_nll", rn);
    ns(metrics, "candidate_mean_nll", cn);
    ns(metrics, "mean_nll_increase", nll_delta / (c.T - 1));
    ns(metrics, "reference_perplexity", rp);
    ns(metrics, "candidate_perplexity", cp);
    bs(metrics, "perplexity_overflow",
       !std::isfinite(rp) || !std::isfinite(cp));
    ns(metrics, "reference_mean_target_rank", ref_rank / (c.T - 1));
    ns(metrics, "candidate_mean_target_rank", can_rank / (c.T - 1));
  }
  struct Criterion {
    const char *flag, *metric;
    bool minimum;
  };
  const Criterion specs[] = {
      {"max-mean-kl", "mean_reference_to_candidate_kl", false},
      {"max-rmse", "logit_rmse", false},
      {"max-abs-logit", "maximum_absolute_logit_error", false},
      {"min-top1-agreement", "top1_agreement", true},
      {"max-mean-nll-increase", "mean_nll_increase", false}};
  auto criteria = list();
  bool failed = false;
  for (auto s : specs) {
    if (!o.v.count(s.flag))
      continue;
    double bound = limitarg(o.req(s.flag));
    need(!s.minimum || bound <= 1, "min-top1-agreement must be <=1");
    auto val = get(metrics, s.metric);
    bool valid = val && val->k == J_NUM;
    double actual = valid ? val->d : NAN;
    bool pass = valid && (s.minimum ? actual >= bound : actual <= bound);
    failed |= !pass;
    auto criterion = obj();
    ss(criterion, "metric", s.metric);
    ss(criterion, "sense", s.minimum ? "min" : "max");
    ns(criterion, "limit", bound);
    ns(criterion, "actual", actual);
    bs(criterion, "passes", pass);
    jv_push(A, criteria, criterion);
  }
  auto report = obj();
  ss(report, "schema", "flash-sass.logit-comparison.v1");
  ss(report, "status",
     incomplete      ? "diagnostic_missing_provenance"
     : failed        ? "failed_declared_limits"
     : criteria->l.n ? "passed_declared_limits_unattested"
                     : "metrics_only");
  set(report, "quality_pass", failed ? jv_bool(A, false) : jv_new(A, J_NULL));
  set(report, "declared_limits_pass",
      criteria->l.n ? jv_bool(A, !failed) : jv_new(A, J_NULL));
  unattested(report);
  set(report, "missing_provenance", miss);
  ss(report, "candidate_manifest_sha256", sha(c.dir / "manifest.json"));
  ss(report, "reference_manifest_sha256", sha(r.dir / "manifest.json"));
  set(report, "metrics", metrics);
  set(report, "declared_criteria", criteria);
  set(report, "positions", detail);
  emit(report, o);
  if (incomplete && !o.diagnostic)
    return 2;
  return failed ? 3 : 0;
}
int main(int argc, char **argv) {
  A = arena_new();
  int rc = 2;
  try {
    uint32_t endian = 1;
    need(*reinterpret_cast<unsigned char *>(&endian) == 1 &&
             sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
         "requires IEEE754 float32 little-endian host");
    need(argc >= 2, "usage: model_quality freeze|compare [--key value ...]");
    S cmd = argv[1];
    need(cmd == "freeze" || cmd == "compare", "unknown command");
    Opt o;
    std::set<S> allowed =
        cmd == "freeze"
            ? std::set<S>{"kind",      "ids",      "logits",   "vocab-size",
                          "vocab-ids", "metadata", "raw-info", "out"}
            : std::set<S>{"candidate",
                          "reference",
                          "out",
                          "top-k",
                          "max-mean-kl",
                          "max-rmse",
                          "max-abs-logit",
                          "min-top1-agreement",
                          "max-mean-nll-increase"};
    for (int i = 2; i < argc; i++) {
      S key = argv[i];
      if (key == "--diagnostic" && cmd == "compare") {
        need(!o.diagnostic, "duplicate diagnostic");
        o.diagnostic = true;
        continue;
      }
      need(key.rfind("--", 0) == 0, "expected --option");
      key = key.substr(2);
      need(allowed.count(key) && !o.v.count(key) && i + 1 < argc,
           "unknown/duplicate/missing option: " + key);
      o.v[key] = argv[++i];
    }
    for (auto flag : {"max-mean-kl", "max-rmse", "max-abs-logit",
                      "min-top1-agreement", "max-mean-nll-increase"})
      if (o.v.count(flag)) {
        double x = limitarg(o.req(flag));
        need(S(flag) != "min-top1-agreement" || x <= 1,
             "min-top1-agreement must be <=1");
      }
    if (o.v.count("top-k"))
      uintarg(o.req("top-k"));
    rc = cmd == "freeze" ? freeze(o) : compare(o);
  } catch (const std::exception &e) {
    auto r = obj();
    ss(r, "status", "invalid_capture");
    ss(r, "error", e.what());
    bs(r, "quality_pass", false);
    std::cerr << encode(r) << '\n';
    rc = 2;
  }
  arena_free(A);
  return rc;
}
