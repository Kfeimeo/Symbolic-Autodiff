// Symbolic reverse-mode automatic differentiation on the Tensor DAG.
//
// Walks the forward DAG in reverse topological order, applies a local
// symbolic VJP rule per operator and accumulates gradient contributions by
// creating Add nodes. Backward nodes live in the same Graph (same NodeId
// space, same hash-consing) and refer to forward intermediates directly.
// No tensor computation happens here.
#pragma once

#include <span>
#include <vector>

#include "sad/graph_builder.hpp"

namespace sad {

struct AutodiffStats {
  uint64_t nodes_before = 0;         // graph size before differentiation
  uint64_t nodes_after = 0;          // graph size after differentiation
  uint64_t visited = 0;              // forward nodes with a non-zero adjoint
  uint64_t grad_contributions = 0;   // local VJP contributions produced
  uint64_t accumulations = 0;        // Add nodes requested for accumulation
  uint64_t create_requests = 0;      // node creation requests during AD
  uint64_t hashcons_hits = 0;        // of which answered by hash-consing
  uint64_t backward_nodes() const { return nodes_after - nodes_before; }
};

struct GradResult {
  std::vector<NodeId> grads;  // one per `wrt`, same order
  AutodiffStats stats;
};

// Gradient of scalar `loss` with respect to every node in `wrt`. A `wrt`
// node that does not influence `loss` receives a zero Constant of its shape.
GradResult backward(GraphBuilder& b, NodeId loss, std::span<const NodeId> wrt);

}  // namespace sad
