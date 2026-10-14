#include "sad/type_inference.hpp"

#include <algorithm>
#include <sstream>

namespace sad {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw TypeError(msg); }

const Node& in(const Graph& g, const Node& n, size_t i) {
  if (i >= n.inputs.size()) fail(std::string(op_name(n.op)) + ": missing input");
  return g.node(n.inputs[i]);
}

void expect_arity(const Node& n, size_t k) {
  if (n.inputs.size() != k) {
    std::ostringstream os;
    os << op_name(n.op) << ": expected " << k << " inputs, got " << n.inputs.size();
    fail(os.str());
  }
}
}  // namespace

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  const size_t r = std::max(a.size(), b.size());
  Shape out(r);
  for (size_t i = 0; i < r; ++i) {
    const int64_t da = i < r - a.size() ? 1 : a[i - (r - a.size())];
    const int64_t db = i < r - b.size() ? 1 : b[i - (r - b.size())];
    if (da != db && da != 1 && db != 1) {
      fail("broadcast mismatch: " + shape_str(a) + " vs " + shape_str(b));
    }
    out[i] = std::max(da, db);
  }
  return out;
}

bool broadcastable_to(const Shape& from, const Shape& to) {
  if (from.size() > to.size()) return false;
  const size_t off = to.size() - from.size();
  for (size_t i = 0; i < from.size(); ++i) {
    if (from[i] != 1 && from[i] != to[i + off]) return false;
  }
  return true;
}

DType promote(DType a, DType b) {
  return static_cast<uint8_t>(a) >= static_cast<uint8_t>(b) ? a : b;
}

Shape sum_shape(const Shape& x, std::vector<int64_t>& dims, bool keepdim) {
  const int64_t rank = static_cast<int64_t>(x.size());
  if (dims.empty()) {
    for (int64_t i = 0; i < rank; ++i) dims.push_back(i);
  }
  for (auto& d : dims) {
    if (d < -rank || d >= rank) fail("Sum: dim out of range");
    if (d < 0) d += rank;
  }
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
  Shape out;
  for (int64_t i = 0; i < rank; ++i) {
    const bool reduced = std::binary_search(dims.begin(), dims.end(), i);
    if (!reduced) out.push_back(x[i]);
    else if (keepdim) out.push_back(1);
  }
  return out;
}

Shape keepdim_shape(const Shape& x, const std::vector<int64_t>& dims) {
  Shape out = x;
  for (auto d : dims) out[d] = 1;
  return out;
}

void infer_type(const Graph& g, Node& n) {
  switch (n.op) {
    case OpCode::Input:
    case OpCode::Constant:
      expect_arity(n, 0);
      for (auto d : n.shape) if (d < 0) fail("negative dimension");
      return;
    case OpCode::Add:
    case OpCode::Mul: {
      expect_arity(n, 2);
      const Node& a = in(g, n, 0);
      const Node& b = in(g, n, 1);
      n.shape = broadcast_shapes(a.shape, b.shape);
      n.dtype = promote(a.dtype, b.dtype);
      return;
    }
    case OpCode::GreaterThan: {
      expect_arity(n, 2);
      const Node& a = in(g, n, 0);
      const Node& b = in(g, n, 1);
      n.shape = broadcast_shapes(a.shape, b.shape);
      n.dtype = DType::Bool;
      return;
    }
    case OpCode::MatMul: {
      expect_arity(n, 2);
      const Node& a = in(g, n, 0);
      const Node& b = in(g, n, 1);
      if (a.shape.size() != 2 || b.shape.size() != 2) fail("MatMul: operands must be 2-D");
      if (a.shape[1] != b.shape[0]) {
        fail("MatMul: inner dimension mismatch " + shape_str(a.shape) + " x " + shape_str(b.shape));
      }
      if (a.dtype != b.dtype) fail("MatMul: dtype mismatch");
      if (a.dtype == DType::Bool) fail("MatMul: bool operands");
      n.shape = {a.shape[0], b.shape[1]};
      n.dtype = a.dtype;
      return;
    }
    case OpCode::ReLU: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      if (a.dtype == DType::Bool) fail("ReLU: bool operand");
      n.shape = a.shape;
      n.dtype = a.dtype;
      return;
    }
    case OpCode::Sum: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      n.shape = sum_shape(a.shape, n.attrs.dims, n.attrs.keepdim);
      n.dtype = a.dtype == DType::Bool ? DType::Float32 : a.dtype;
      return;
    }
    case OpCode::Transpose: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      if (a.shape.size() != 2) fail("Transpose: operand must be 2-D");
      n.shape = {a.shape[1], a.shape[0]};
      n.dtype = a.dtype;
      return;
    }
    case OpCode::BroadcastTo: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      if (!broadcastable_to(a.shape, n.shape)) {
        fail("BroadcastTo: cannot broadcast " + shape_str(a.shape) + " to " + shape_str(n.shape));
      }
      n.dtype = a.dtype;
      return;
    }
    case OpCode::SumToShape: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      if (!broadcastable_to(n.shape, a.shape)) {
        fail("SumToShape: " + shape_str(n.shape) + " is not a broadcast source of " + shape_str(a.shape));
      }
      n.dtype = a.dtype;
      return;
    }
    case OpCode::Reshape: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      if (numel(a.shape) != numel(n.shape)) {
        fail("Reshape: numel mismatch " + shape_str(a.shape) + " -> " + shape_str(n.shape));
      }
      n.dtype = a.dtype;
      return;
    }
    case OpCode::OnesLike: {
      expect_arity(n, 1);
      const Node& a = in(g, n, 0);
      n.shape = a.shape;
      n.dtype = a.dtype;
      return;
    }
    default:
      fail("unknown op");
  }
}

}  // namespace sad
