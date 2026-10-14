#include "sad/executor.hpp"

#include <ATen/ExpandUtils.h>

#include <stdexcept>

namespace sad {

at::ScalarType to_aten(DType dt) {
  switch (dt) {
    case DType::Bool: return at::kBool;
    case DType::Float32: return at::kFloat;
    case DType::Float64: return at::kDouble;
  }
  throw std::logic_error("to_aten: bad dtype");
}

DType from_aten(at::ScalarType st) {
  switch (st) {
    case at::kBool: return DType::Bool;
    case at::kFloat: return DType::Float32;
    case at::kDouble: return DType::Float64;
    default: throw std::logic_error("from_aten: unsupported scalar type");
  }
}

at::Tensor dispatch(OpCode op, const Attrs& attrs, const Shape& shape, DType dtype,
                    const at::Tensor* v, size_t n) {
  switch (op) {
    case OpCode::Constant:
      return at::full(at::IntArrayRef(shape), attrs.value, at::TensorOptions().dtype(to_aten(dtype)));
    case OpCode::Add: return at::add(v[0], v[1]);
    case OpCode::Mul: return at::mul(v[0], v[1]);
    case OpCode::MatMul: return at::matmul(v[0], v[1]);
    case OpCode::ReLU: return at::relu(v[0]);
    case OpCode::Sum: return at::sum(v[0], at::IntArrayRef(attrs.dims), attrs.keepdim);
    case OpCode::Transpose: return v[0].t();
    case OpCode::BroadcastTo: return v[0].expand(at::IntArrayRef(shape));
    case OpCode::SumToShape: return at::sum_to(v[0], at::IntArrayRef(shape));
    case OpCode::Reshape: return v[0].reshape(at::IntArrayRef(shape));
    case OpCode::GreaterThan: return at::gt(v[0], v[1]);
    case OpCode::OnesLike: return at::ones_like(v[0]);
    case OpCode::Input:
    default:
      (void)n;
      throw std::logic_error(std::string("dispatch: cannot execute ") + op_name(op));
  }
}

Executor::Executor(const SLP& program) : p_(program) {
  vals_.resize(p_.graph->size());
  instr_of_.assign(p_.graph->size(), static_cast<uint32_t>(-1));
  for (size_t k = 0; k < p_.instrs.size(); ++k) instr_of_[p_.instrs[k].dst] = static_cast<uint32_t>(k);
}

void Executor::bind(NodeId input, at::Tensor t) {
  const Node& n = p_.graph->node(input);
  if (n.op != OpCode::Input) throw std::logic_error("bind: node is not an Input");
  if (t.sizes() != at::IntArrayRef(n.shape) || t.scalar_type() != to_aten(n.dtype)) {
    throw std::logic_error("bind: tensor shape/dtype does not match Input " + shape_str(n.shape));
  }
  vals_[input] = std::move(t);
}

void Executor::bind_inputs(const std::vector<at::Tensor>& in_order) {
  if (in_order.size() != p_.inputs.size()) throw std::logic_error("bind_inputs: wrong count");
  for (size_t i = 0; i < in_order.size(); ++i) bind(p_.inputs[i], in_order[i]);
}

size_t Executor::instr_index_of(NodeId id) const {
  const uint32_t k = instr_of_[id];
  if (k == static_cast<uint32_t>(-1)) throw std::logic_error("instr_index_of: node not executed");
  return k;
}

void Executor::on_define(const at::Tensor& t) {
  ++live_values_;
  const void* key = t.storage().data();
  const uint64_t bytes = t.storage().nbytes();
  auto [it, inserted] = live_storages_.try_emplace(key, std::make_pair(bytes, 0u));
  if (inserted) live_bytes_ += bytes;
  ++it->second.second;
  stats_.peak_live_bytes = std::max(stats_.peak_live_bytes, live_bytes_);
  stats_.peak_live_values = std::max(stats_.peak_live_values, live_values_);
}

void Executor::on_release(const at::Tensor& t) {
  --live_values_;
  auto it = live_storages_.find(t.storage().data());
  if (it == live_storages_.end()) return;
  if (--it->second.second == 0) {
    live_bytes_ -= it->second.first;
    live_storages_.erase(it);
  }
}

void Executor::run(bool track_memory) { run_range(0, p_.instrs.size(), track_memory); }

void Executor::run_range(size_t begin, size_t end, bool track_memory) {
  if (begin == 0) {
    stats_ = ExecStats{};
    live_storages_.clear();
    live_bytes_ = 0;
    live_values_ = 0;
    tracking_ = track_memory;
    for (auto in : p_.inputs) {
      if (!vals_[in].defined()) throw std::logic_error("run: unbound input");
    }
  }
  const Graph& g = *p_.graph;
  at::Tensor operands[4];
  for (size_t k = begin; k < end; ++k) {
    const Instr& ins = p_.instrs[k];
    const Node& n = g.node(ins.dst);
    for (size_t j = 0; j < n.inputs.size(); ++j) operands[j] = vals_[n.inputs[j]];
    at::Tensor out = dispatch(n.op, n.attrs, n.shape, n.dtype, operands, n.inputs.size());
    for (size_t j = 0; j < n.inputs.size(); ++j) operands[j].reset();
    ++stats_.instructions;
    if (tracking_) on_define(out);
    vals_[ins.dst] = std::move(out);
    for (auto r : ins.release) {
      if (tracking_) on_release(vals_[r]);
      vals_[r].reset();
      ++stats_.releases;
    }
  }
  if (end == p_.instrs.size()) stats_.end_live_bytes = live_bytes_;
}

void Executor::reset() {
  for (const auto& ins : p_.instrs) vals_[ins.dst].reset();
  live_storages_.clear();
  live_bytes_ = 0;
  live_values_ = 0;
}

}  // namespace sad
