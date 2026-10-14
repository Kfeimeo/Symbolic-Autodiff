#include "sad/torch_baseline.hpp"

#include "sad/executor.hpp"

namespace sad {

std::vector<torch::Tensor> make_task_inputs(const TaskParams& p, uint64_t seed) {
  torch::manual_seed(seed);
  std::vector<torch::Tensor> out;
  const auto opts = torch::TensorOptions().dtype(to_aten(p.dtype));
  for (const auto& spec : task_input_specs(p)) {
    const Shape& s = spec.shape;
    torch::Tensor t = torch::randn(at::IntArrayRef(s), opts);
    // Keep activations O(1) at any depth (no overflow, no subnormals):
    //  - matrices get a Kaiming/He scale,
    //  - the sharing task sums `sharing` identical copies, so weights and
    //    biases are divided by that factor (the maths then equals the chain),
    //  - the diamond task sums `width` branches, so weights get 1/sqrt(width).
    if (s.size() == 2 && spec.role == "w") {
      t = t * std::sqrt(2.0 / static_cast<double>(s[0]));
    } else if (p.task == "sharing" && spec.role != "x") {
      t = t / static_cast<double>(p.sharing);
    } else if (p.task == "diamond" && spec.role == "w") {
      t = t / std::sqrt(static_cast<double>(p.width));
    }
    out.push_back(t.contiguous());
  }
  return out;
}

std::vector<torch::Tensor> make_leaves(const std::vector<torch::Tensor>& inputs) {
  std::vector<torch::Tensor> leaves;
  leaves.reserve(inputs.size());
  for (const auto& t : inputs) leaves.push_back(t.detach().requires_grad_(true));
  return leaves;
}

TorchResult torch_reference(const TaskParams& p, const std::vector<torch::Tensor>& inputs) {
  auto leaves = make_leaves(inputs);
  TorchBackend be{leaves};
  TorchResult r;
  r.loss = build_task(be, p);
  auto grads = torch::autograd::grad({r.loss}, leaves, {}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  for (size_t i = 0; i < grads.size(); ++i) {
    r.grads.push_back(grads[i].defined() ? grads[i] : torch::zeros_like(leaves[i]));
  }
  return r;
}

at::Tensor eval_tree(const TreeNode& n, const std::vector<torch::Tensor>& inputs) {
  if (n.op == OpCode::Input) return inputs.at(n.attrs.input_index);
  at::Tensor operands[4];
  for (size_t i = 0; i < n.kids.size(); ++i) operands[i] = eval_tree(*n.kids[i], inputs);
  return dispatch(n.op, n.attrs, n.shape, n.dtype, operands, n.kids.size());
}

Closeness compare(const torch::Tensor& got, const torch::Tensor& ref, double rtol, double atol) {
  Closeness c;
  torch::NoGradGuard ng;
  auto g = got.detach().to(torch::kDouble);
  auto r = ref.detach().to(torch::kDouble);
  if (g.sizes() != r.sizes()) {
    c.allclose = false;
    c.max_abs_err = c.max_rel_err = std::numeric_limits<double>::infinity();
    return c;
  }
  const auto diff = (g - r).abs();
  c.max_abs_err = diff.numel() ? diff.max().item<double>() : 0.0;
  const double scale = r.numel() ? r.abs().max().item<double>() : 0.0;
  c.max_rel_err = c.max_abs_err / (scale > 0 ? scale : 1.0);
  c.allclose = torch::allclose(g, r, rtol, atol + rtol * scale);
  return c;
}

}  // namespace sad
