// Baseline C: LibTorch eager execution with native Autograd.
#pragma once

#include <torch/torch.h>

#include <vector>

#include "sad/tasks.hpp"

namespace sad {

struct TorchBackend {
  using Value = torch::Tensor;
  const std::vector<torch::Tensor>& inputs;  // leaves, in declaration order
  size_t next = 0;
  Value input(Shape, const char*) { return inputs.at(next++); }
  Value add(Value& x, Value& y) { return at::add(x, y); }
  Value mul(Value& x, Value& y) { return at::mul(x, y); }
  Value matmul(Value& x, Value& y) { return at::matmul(x, y); }
  Value relu(Value& x) { return at::relu(x); }
  Value sum(Value& x, std::vector<int64_t> dims, bool keepdim) {
    // Same normalisation as the DAG executor: an empty dim list means all.
    if (dims.empty()) {
      for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    return at::sum(x, at::IntArrayRef(dims), keepdim);
  }
};

// Deterministic input tensors for a task (fixed seed), generated on the CPU
// and moved to `device`.
std::vector<torch::Tensor> make_task_inputs(const TaskParams& p, uint64_t seed,
                                            at::Device device = at::kCPU);
// Parses "cpu" / "cuda" / "cuda:N"; throws if CUDA is requested but absent.
at::Device parse_device(const std::string& name);
// Waits for all queued work on `device` (no-op on CPU).
void synchronize(at::Device device);
// Peak bytes allocated by the device caching allocator since the last reset
// (CUDA only; returns false when not available).
bool device_allocator_peak(at::Device device, uint64_t& peak_bytes);
void device_allocator_reset_peak(at::Device device);

// Fresh autograd leaves sharing the data of `inputs`.
std::vector<torch::Tensor> make_leaves(const std::vector<torch::Tensor>& inputs);

struct TorchResult {
  torch::Tensor loss;
  std::vector<torch::Tensor> grads;
};

// Eager forward with autograd recording, then torch::autograd::grad.
TorchResult torch_reference(const TaskParams& p, const std::vector<torch::Tensor>& inputs);

// Evaluates a naive tree expression with ATen (used to validate the tree
// differentiation against the DAG on small problems).
at::Tensor eval_tree(const TreeNode& n, const std::vector<torch::Tensor>& inputs);

struct Closeness {
  double max_abs_err = 0;
  double max_rel_err = 0;  // max_abs_err / max(|reference|)
  bool allclose = true;
};
Closeness compare(const torch::Tensor& got, const torch::Tensor& ref, double rtol, double atol);

}  // namespace sad
