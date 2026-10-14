// Static shape / dtype inference and checking for every OpCode.
#pragma once

#include <stdexcept>

#include "sad/ir.hpp"

namespace sad {

struct TypeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// NumPy/PyTorch broadcasting of two shapes. Throws TypeError on mismatch.
Shape broadcast_shapes(const Shape& a, const Shape& b);
// True if `from` can be broadcast to `to`.
bool broadcastable_to(const Shape& from, const Shape& to);
// Result dtype of a binary arithmetic op (Bool < Float32 < Float64).
DType promote(DType a, DType b);
// Shape of Sum(x, dims, keepdim); dims are normalised in place.
Shape sum_shape(const Shape& x, std::vector<int64_t>& dims, bool keepdim);
// Shape after reducing with keepdim=true (used by the Sum VJP).
Shape keepdim_shape(const Shape& x, const std::vector<int64_t>& dims);

// Fills n.shape / n.dtype from n.op, n.inputs and n.attrs; the shape of
// BroadcastTo / SumToShape / Reshape is an attribute and must already be set
// in n.shape. Throws TypeError when the operands are incompatible.
void infer_type(const Graph& g, Node& n);

}  // namespace sad
