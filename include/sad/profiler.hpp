// Backend-neutral measurement of ATen operator invocations and allocator
// memory. Works for the SLP executor and for LibTorch Autograd alike, so the
// two can be compared with the same instrument.
#pragma once

#include <cstdint>

namespace sad {

struct OpProfile {
  uint64_t aten_ops = 0;            // RecordFunction FUNCTION-scope events
  uint64_t peak_malloc_bytes = 0;   // peak glibc in-use bytes sampled at op boundaries
  uint64_t base_malloc_bytes = 0;   // in-use bytes when the profile started
  uint64_t peak_malloc_delta() const {
    return peak_malloc_bytes > base_malloc_bytes ? peak_malloc_bytes - base_malloc_bytes : 0;
  }
};

// RAII scope: counts every ATen operator dispatched while alive (all threads)
// and samples malloc usage before/after each operator.
class OpProfileScope {
 public:
  explicit OpProfileScope(bool sample_memory = true);
  ~OpProfileScope();
  OpProfile result() const;

 private:
  uint64_t handle_;
  uint64_t base_ = 0;
};

// Current glibc malloc in-use bytes (arena + mmap chunks).
uint64_t malloc_in_use_bytes();
// Peak resident set size of the process in bytes.
uint64_t peak_rss_bytes();

}  // namespace sad
