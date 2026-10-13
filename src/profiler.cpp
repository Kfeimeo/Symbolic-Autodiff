#include "sad/profiler.hpp"

#include <ATen/record_function.h>
#include <malloc.h>
#include <sys/resource.h>

#include <atomic>

namespace sad {

namespace {
std::atomic<uint64_t> g_ops{0};
std::atomic<uint64_t> g_peak{0};
std::atomic<bool> g_sample{false};

void sample_peak() {
  const uint64_t cur = malloc_in_use_bytes();
  uint64_t prev = g_peak.load(std::memory_order_relaxed);
  while (cur > prev && !g_peak.compare_exchange_weak(prev, cur, std::memory_order_relaxed)) {
  }
}

std::unique_ptr<at::ObserverContext> on_start(const at::RecordFunction&) {
  g_ops.fetch_add(1, std::memory_order_relaxed);
  if (g_sample.load(std::memory_order_relaxed)) sample_peak();
  return nullptr;
}

void on_end(const at::RecordFunction&, at::ObserverContext*) {
  if (g_sample.load(std::memory_order_relaxed)) sample_peak();
}
}  // namespace

uint64_t malloc_in_use_bytes() {
  const struct mallinfo2 mi = mallinfo2();
  return static_cast<uint64_t>(mi.uordblks) + static_cast<uint64_t>(mi.hblkhd);
}

uint64_t peak_rss_bytes() {
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return static_cast<uint64_t>(ru.ru_maxrss) * 1024;
}

OpProfileScope::OpProfileScope(bool sample_memory) {
  g_ops.store(0);
  const uint64_t base = malloc_in_use_bytes();
  g_peak.store(base);
  g_sample.store(sample_memory);
  base_ = base;
  handle_ = at::addGlobalCallback(
      at::RecordFunctionCallback(&on_start, &on_end).scopes({at::RecordScope::FUNCTION}));
}

OpProfileScope::~OpProfileScope() {
  at::removeCallback(handle_);
  g_sample.store(false);
}

OpProfile OpProfileScope::result() const {
  OpProfile r;
  r.aten_ops = g_ops.load();
  r.peak_malloc_bytes = g_peak.load();
  r.base_malloc_bytes = base_;
  return r;
}

}  // namespace sad
