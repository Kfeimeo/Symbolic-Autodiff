// Deterministic node construction API on top of Graph: performs type
// inference, shape checking and hash-consing for every created node.
#pragma once

#include <string>

#include "sad/ir.hpp"
#include "sad/type_inference.hpp"

namespace sad {

class GraphBuilder {
 public:
  explicit GraphBuilder(Graph& g) : g_(g) {}
  Graph& graph() { return g_; }
  const Graph& graph() const { return g_; }

  NodeId input(Shape shape, DType dtype = DType::Float32);
  NodeId constant(double value, Shape shape = {}, DType dtype = DType::Float32);
  NodeId add(NodeId a, NodeId b);
  NodeId mul(NodeId a, NodeId b);
  NodeId matmul(NodeId a, NodeId b);
  NodeId relu(NodeId a);
  NodeId sum(NodeId a, std::vector<int64_t> dims = {}, bool keepdim = false);
  NodeId transpose(NodeId a);
  // Identity (no node created) when the shape already matches.
  NodeId broadcast_to(NodeId a, Shape shape);
  // Identity (no node created) when the shape already matches.
  NodeId sum_to_shape(NodeId a, Shape shape);
  // Identity (no node created) when the shape already matches.
  NodeId reshape(NodeId a, Shape shape);
  NodeId greater_than(NodeId a, NodeId b);
  NodeId ones_like(NodeId a);

  const Node& node(NodeId id) const { return g_.node(id); }
  const Shape& shape(NodeId id) const { return g_.node(id).shape; }
  DType dtype(NodeId id) const { return g_.node(id).dtype; }

 private:
  NodeId make(OpCode op, std::vector<NodeId> inputs, Attrs attrs = {}, Shape shape = {});
  Graph& g_;
};

}  // namespace sad
