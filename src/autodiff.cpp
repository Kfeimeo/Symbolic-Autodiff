#include "sad/autodiff.hpp"

#include <stdexcept>

#include "sad/analysis.hpp"

namespace sad {

namespace {

struct Backward {
  GraphBuilder& b;
  const Graph& g;
  std::vector<NodeId> adj;  // adjoint per forward node (kInvalidNode = zero)
  AutodiffStats stats;

  void accumulate(NodeId target, NodeId contrib) {
    ++stats.grad_contributions;
    if (g.node(contrib).shape != g.node(target).shape) {
      throw std::logic_error(std::string("autodiff: gradient shape ") +
                             shape_str(g.node(contrib).shape) + " does not match node shape " +
                             shape_str(g.node(target).shape) + " for " +
                             op_name(g.node(target).op));
    }
    if (adj[target] == kInvalidNode) {
      adj[target] = contrib;
    } else {
      ++stats.accumulations;
      adj[target] = b.add(adj[target], contrib);
    }
  }

  // Local symbolic VJP rules. `gy` is the adjoint of node `id`.
  void vjp(NodeId id, NodeId gy) {
    // Copy: creating backward nodes may reallocate the node pool, which would
    // invalidate a reference into it.
    const Node n = g.node(id);
    const std::vector<NodeId>& in = n.inputs;
    switch (n.op) {
      case OpCode::Input:
      case OpCode::Constant:
      case OpCode::OnesLike:
      case OpCode::GreaterThan:
        return;  // leaves or piecewise-constant: no gradient flows
      case OpCode::Add: {
        accumulate(in[0], b.sum_to_shape(gy, g.node(in[0]).shape));
        accumulate(in[1], b.sum_to_shape(gy, g.node(in[1]).shape));
        return;
      }
      case OpCode::Mul: {
        const NodeId ga = b.mul(gy, in[1]);
        const NodeId gb = b.mul(gy, in[0]);
        accumulate(in[0], b.sum_to_shape(ga, g.node(in[0]).shape));
        accumulate(in[1], b.sum_to_shape(gb, g.node(in[1]).shape));
        return;
      }
      case OpCode::MatMul: {
        accumulate(in[0], b.matmul(gy, b.transpose(in[1])));
        accumulate(in[1], b.matmul(b.transpose(in[0]), gy));
        return;
      }
      case OpCode::ReLU: {
        // d relu(x) = 1[x > 0]; the subgradient at exactly 0 is taken as 0.
        const NodeId mask = b.greater_than(in[0], b.constant(0.0, {}, g.node(in[0]).dtype));
        accumulate(in[0], b.mul(gy, mask));
        return;
      }
      case OpCode::Sum: {
        const Shape& xs = g.node(in[0]).shape;
        NodeId gk = gy;
        if (!n.attrs.keepdim) gk = b.reshape(gy, keepdim_shape(xs, n.attrs.dims));
        accumulate(in[0], b.broadcast_to(gk, xs));
        return;
      }
      case OpCode::Transpose:
        accumulate(in[0], b.transpose(gy));
        return;
      case OpCode::BroadcastTo:
        accumulate(in[0], b.sum_to_shape(gy, g.node(in[0]).shape));
        return;
      case OpCode::SumToShape:
        accumulate(in[0], b.broadcast_to(gy, g.node(in[0]).shape));
        return;
      case OpCode::Reshape:
        accumulate(in[0], b.reshape(gy, g.node(in[0]).shape));
        return;
      default:
        throw std::logic_error("autodiff: no VJP rule for op");
    }
  }
};

}  // namespace

GradResult backward(GraphBuilder& b, NodeId loss, std::span<const NodeId> wrt) {
  Graph& g = b.graph();
  if (numel(g.node(loss).shape) != 1) {
    throw std::logic_error("autodiff: loss must be a scalar (numel == 1)");
  }
  Backward bw{b, g, {}, {}};
  bw.stats.nodes_before = g.size();
  const auto hits0 = g.stats().hashcons_hits;
  const auto req0 = g.stats().create_requests;

  // Only forward nodes reachable from the loss can have a non-zero adjoint.
  const NodeId roots[1] = {loss};
  const std::vector<bool> fwd = reachable_from(g, roots);
  bw.adj.assign(bw.stats.nodes_before, kInvalidNode);
  bw.adj[loss] = b.ones_like(loss);

  // Node ids are a topological order, so a descending sweep visits every node
  // after all of its consumers: its adjoint is complete when we reach it.
  for (NodeId id = loss + 1; id-- > 0;) {
    if (!fwd[id] || bw.adj[id] == kInvalidNode) continue;
    ++bw.stats.visited;
    bw.vjp(id, bw.adj[id]);
  }

  GradResult res;
  res.grads.reserve(wrt.size());
  for (auto w : wrt) {
    NodeId gw = w < bw.adj.size() ? bw.adj[w] : kInvalidNode;
    if (gw == kInvalidNode) gw = b.constant(0.0, g.node(w).shape, g.node(w).dtype);
    res.grads.push_back(gw);
  }
  bw.stats.nodes_after = g.size();
  bw.stats.hashcons_hits = g.stats().hashcons_hits - hits0;
  bw.stats.create_requests = g.stats().create_requests - req0;
  res.stats = bw.stats;
  return res;
}

}  // namespace sad
