#include "sad/ir.hpp"

#include <sstream>

namespace sad {

const char* op_name(OpCode op) {
  switch (op) {
    case OpCode::Input: return "Input";
    case OpCode::Constant: return "Constant";
    case OpCode::Add: return "Add";
    case OpCode::Mul: return "Mul";
    case OpCode::MatMul: return "MatMul";
    case OpCode::ReLU: return "ReLU";
    case OpCode::Sum: return "Sum";
    case OpCode::Transpose: return "Transpose";
    case OpCode::BroadcastTo: return "BroadcastTo";
    case OpCode::SumToShape: return "SumToShape";
    case OpCode::Reshape: return "Reshape";
    case OpCode::GreaterThan: return "GreaterThan";
    case OpCode::OnesLike: return "OnesLike";
    default: return "?";
  }
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::Bool: return "bool";
    case DType::Float32: return "f32";
    case DType::Float64: return "f64";
  }
  return "?";
}

size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Bool: return 1;
    case DType::Float32: return 4;
    case DType::Float64: return 8;
  }
  return 0;
}

int64_t numel(const Shape& s) {
  int64_t n = 1;
  for (auto d : s) n *= d;
  return n;
}

std::string shape_str(const Shape& s) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) os << ',';
    os << s[i];
  }
  os << ']';
  return os.str();
}

bool Attrs::operator==(const Attrs& o) const {
  // Compare the double bit-wise so that -0.0 and 0.0 are distinct constants
  // and NaN compares consistently.
  uint64_t a, b;
  static_assert(sizeof(a) == sizeof(value));
  __builtin_memcpy(&a, &value, sizeof(a));
  __builtin_memcpy(&b, &o.value, sizeof(b));
  return dims == o.dims && keepdim == o.keepdim && a == b &&
         input_index == o.input_index;
}

size_t Graph::node_bytes() const {
  size_t bytes = nodes_.capacity() * sizeof(Node);
  for (const auto& n : nodes_) {
    bytes += n.inputs.capacity() * sizeof(NodeId);
    bytes += n.shape.capacity() * sizeof(int64_t);
    bytes += n.attrs.dims.capacity() * sizeof(int64_t);
  }
  return bytes;
}

}  // namespace sad
