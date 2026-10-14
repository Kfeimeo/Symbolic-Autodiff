// SLP executor: dispatches every instruction to ATen, keeps a
// NodeId -> at::Tensor map and releases values after their last use.
#pragma once

#include <ATen/ATen.h>

#include <unordered_map>
#include <vector>

#include "sad/slp.hpp"

namespace sad {

at::ScalarType to_aten(DType dt);
DType from_aten(at::ScalarType st);

// Pure ATen dispatch of one operator (shared with the tree evaluator).
// `device` is only used by operand-less ops (Constant).
at::Tensor dispatch(OpCode op, const Attrs& attrs, const Shape& shape, DType dtype,
                    const at::Tensor* operands, size_t n_operands, at::Device device);

struct ExecStats {
  uint64_t instructions = 0;      // SLP instructions dispatched
  uint64_t releases = 0;          // values released after last use
  uint64_t peak_live_bytes = 0;   // logical bytes of live storages (dedup by storage)
  uint64_t end_live_bytes = 0;    // live bytes when the run finished
  uint64_t peak_live_values = 0;  // number of simultaneously live values
};

class Executor {
 public:
  explicit Executor(const SLP& program);

  // All inputs must live on one device; the executor adopts it.
  void bind(NodeId input, at::Tensor t);
  at::Device device() const { return device_; }
  void bind_inputs(const std::vector<at::Tensor>& in_order);  // program.inputs order

  // Executes the whole program. With track_memory the (slower) storage
  // accounting is enabled and ExecStats is filled.
  void run(bool track_memory = false);
  // Executes instructions [begin, end). Used to time forward and backward
  // portions of a unified program separately.
  void run_range(size_t begin, size_t end, bool track_memory = false);
  size_t instr_index_of(NodeId id) const;  // first instruction defining id

  const at::Tensor& value(NodeId id) const { return vals_[id]; }
  bool has_value(NodeId id) const { return vals_[id].defined(); }
  // Drops all non-input values (outputs included) so the next run starts clean.
  void reset();
  const ExecStats& stats() const { return stats_; }
  const SLP& program() const { return p_; }

 private:
  void on_define(const at::Tensor& t);
  void on_release(const at::Tensor& t);

  const SLP& p_;
  at::Device device_ = at::kCPU;
  bool device_set_ = false;
  std::vector<at::Tensor> vals_;
  std::vector<uint32_t> instr_of_;
  ExecStats stats_;
  // storage data pointer -> (bytes, live value count)
  std::unordered_map<const void*, std::pair<uint64_t, uint32_t>> live_storages_;
  uint64_t live_bytes_ = 0;
  uint64_t live_values_ = 0;
  bool tracking_ = false;
};

}  // namespace sad
