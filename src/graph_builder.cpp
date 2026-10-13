#include "sad/graph_builder.hpp"

namespace sad {

NodeId GraphBuilder::make(OpCode op, std::vector<NodeId> inputs, Attrs attrs, Shape shape) {
  Node n;
  n.op = op;
  n.inputs = std::move(inputs);
  n.attrs = std::move(attrs);
  n.shape = std::move(shape);
  infer_type(g_, n);
  return g_.emplace(std::move(n));
}

NodeId GraphBuilder::input(Shape shape, DType dtype) {
  Node n;
  n.op = OpCode::Input;
  n.shape = std::move(shape);
  n.dtype = dtype;
  n.attrs.input_index = g_.next_input_index();
  infer_type(g_, n);
  const NodeId id = g_.emplace(std::move(n));
  g_.register_input(id);
  return id;
}

NodeId GraphBuilder::constant(double value, Shape shape, DType dtype) {
  Node n;
  n.op = OpCode::Constant;
  n.shape = std::move(shape);
  n.dtype = dtype;
  n.attrs.value = value;
  infer_type(g_, n);
  return g_.emplace(std::move(n));
}

NodeId GraphBuilder::add(NodeId a, NodeId b) { return make(OpCode::Add, {a, b}); }
NodeId GraphBuilder::mul(NodeId a, NodeId b) { return make(OpCode::Mul, {a, b}); }
NodeId GraphBuilder::matmul(NodeId a, NodeId b) { return make(OpCode::MatMul, {a, b}); }
NodeId GraphBuilder::relu(NodeId a) { return make(OpCode::ReLU, {a}); }
NodeId GraphBuilder::transpose(NodeId a) { return make(OpCode::Transpose, {a}); }
NodeId GraphBuilder::greater_than(NodeId a, NodeId b) { return make(OpCode::GreaterThan, {a, b}); }
NodeId GraphBuilder::ones_like(NodeId a) { return make(OpCode::OnesLike, {a}); }

NodeId GraphBuilder::sum(NodeId a, std::vector<int64_t> dims, bool keepdim) {
  Attrs attrs;
  attrs.dims = std::move(dims);
  attrs.keepdim = keepdim;
  return make(OpCode::Sum, {a}, std::move(attrs));
}

NodeId GraphBuilder::broadcast_to(NodeId a, Shape shape) {
  if (g_.node(a).shape == shape) return a;
  return make(OpCode::BroadcastTo, {a}, {}, std::move(shape));
}

NodeId GraphBuilder::sum_to_shape(NodeId a, Shape shape) {
  if (g_.node(a).shape == shape) return a;
  return make(OpCode::SumToShape, {a}, {}, std::move(shape));
}

NodeId GraphBuilder::reshape(NodeId a, Shape shape) {
  if (g_.node(a).shape == shape) return a;
  return make(OpCode::Reshape, {a}, {}, std::move(shape));
}

}  // namespace sad
