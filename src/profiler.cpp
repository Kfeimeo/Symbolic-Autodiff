#include "sad/profiler.hpp"

#include <ATen/record_function.h>

#include <atomic>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__GLIBC__)
#include <malloc.h>
#include <sys/resource.h>
#else
#include <sys/resource.h>
#endif

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

// Allocator in-use bytes. glibc: arena + mmap'd chunks (mallinfo2). Windows:
// private committed bytes of the process (PROCESS_MEMORY_COUNTERS_EX). Other
// platforms: 0 (unsupported, reported as such).
uint64_t malloc_in_use_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    return static_cast<uint64_t>(pmc.PrivateUsage);
  }
  return 0;
#elif defined(__GLIBC__)
  const struct mallinfo2 mi = mallinfo2();
  return static_cast<uint64_t>(mi.uordblks) + static_cast<uint64_t>(mi.hblkhd);
#else
  return 0;
#endif
}

bool malloc_accounting_supported() {
#if defined(_WIN32) || defined(__GLIBC__)
  return true;
#else
  return false;
#endif
}

uint64_t peak_rss_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<uint64_t>(pmc.PeakWorkingSetSize);
  }
  return 0;
#else
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return static_cast<uint64_t>(ru.ru_maxrss);  // bytes on macOS
#else
  return static_cast<uint64_t>(ru.ru_maxrss) * 1024;  // KiB on Linux
#endif
#endif
}

bool make_malloc_stable() {
#if defined(__GLIBC__)
  // Keep freed memory inside the process so that neither backend pays
  // mmap/munmap and page-fault costs that depend on run order.
  mallopt(M_MMAP_THRESHOLD, 1 << 30);
  mallopt(M_TRIM_THRESHOLD, -1);
  mallopt(M_TOP_PAD, 64 << 20);
  return true;
#else
  return false;  // no equivalent tuning on this allocator
#endif
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
