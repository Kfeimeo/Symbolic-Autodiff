// Structural tests: hash-consing, type inference, analysis, autodiff graph
// construction, SLP liveness and the tree baseline. No tensor execution.
#include "sad/analysis.hpp"
#include "sad/autodiff.hpp"
#include "sad/graph_builder.hpp"
#include "sad/slp.hpp"
#include "sad/tasks.hpp"
#include "sad/tree_ast.hpp"
#include "testing.hpp"

using namespace sad;

TEST(hashcons_merges_identical_nodes) {
  Graph g;
  GraphBuilder b(g);
  NodeId a = b.input({4});
  NodeId c = b.input({4});
  REQUIRE(a != c);  // inputs are distinct identities
  NodeId s1 = b.add(a, c);
  NodeId s2 = b.add(a, c);
  REQUIRE_EQ(s1, s2);
  NodeId s3 = b.add(c, a);  // no commutativity rewrite
  REQUIRE(s3 != s1);
  REQUIRE_EQ(g.stats().hashcons_hits, uint64_t(1));
  NodeId k1 = b.constant(1.0, {4});
  NodeId k2 = b.constant(1.0, {4});
  NodeId k3 = b.constant(1.0, {2});
  NodeId k4 = b.constant(1.0, {4}, DType::Float64);
  NodeId k5 = b.constant(2.0, {4});
  REQUIRE_EQ(k1, k2);
  REQUIRE(k1 != k3);
  REQUIRE(k1 != k4);
  REQUIRE(k1 != k5);
  NodeId r1 = b.sum(a, {0}, true);
  NodeId r2 = b.sum(a, {0}, false);
  NodeId r3 = b.sum(a, {0}, true);
  REQUIRE(r1 != r2);
  REQUIRE_EQ(r1, r3);
  NodeId m = b.input({2, 3});
  NodeId bc1 = b.broadcast_to(a, {3, 4});
  NodeId bc2 = b.broadcast_to(a, {2, 4});
  NodeId bc3 = b.broadcast_to(a, {3, 4});
  REQUIRE(bc1 != bc2);
  REQUIRE_EQ(bc1, bc3);
  REQUIRE_EQ(b.broadcast_to(a, {4}), a);  // identity elided
  REQUIRE_EQ(b.transpose(b.transpose(m)) != m, true);  // no algebraic simplification
}

TEST(hashcons_disabled_creates_duplicates) {
  Graph g(/*hash_consing=*/false);
  GraphBuilder b(g);
  NodeId a = b.input({4});
  NodeId s1 = b.add(a, a);
  NodeId s2 = b.add(a, a);
  REQUIRE(s1 != s2);
  REQUIRE_EQ(g.stats().hashcons_hits, uint64_t(0));
  REQUIRE_EQ(g.size(), size_t(3));
}

TEST(type_inference_rules) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({8, 1, 5});
  NodeId y = b.input({4, 5});
  NodeId z = b.add(x, y);
  REQUIRE(b.shape(z) == Shape({8, 4, 5}));
  REQUIRE_THROWS(b.add(b.input({3}), b.input({4})), TypeError);
  NodeId m1 = b.input({3, 4});
  NodeId m2 = b.input({4, 2});
  REQUIRE(b.shape(b.matmul(m1, m2)) == Shape({3, 2}));
  REQUIRE_THROWS(b.matmul(m2, m1), TypeError);
  REQUIRE_THROWS(b.matmul(m1, b.input({4})), TypeError);
  REQUIRE(b.shape(b.transpose(m1)) == Shape({4, 3}));
  REQUIRE_THROWS(b.transpose(x), TypeError);
  REQUIRE(b.shape(b.sum(x, {}, false)) == Shape({}));
  REQUIRE(b.shape(b.sum(x, {1}, false)) == Shape({8, 5}));
  REQUIRE(b.shape(b.sum(x, {-1}, true)) == Shape({8, 1, 1}));
  REQUIRE(b.shape(b.sum(x, {0, 2}, false)) == Shape({1}));
  REQUIRE_THROWS(b.sum(x, {3}), TypeError);
  REQUIRE(b.shape(b.sum_to_shape(z, {4, 5})) == Shape({4, 5}));
  REQUIRE(b.shape(b.sum_to_shape(z, {8, 1, 5})) == Shape({8, 1, 5}));
  REQUIRE_THROWS(b.sum_to_shape(z, {8, 5}), TypeError);
  REQUIRE_THROWS(b.broadcast_to(y, {4, 6}), TypeError);
  REQUIRE(b.shape(b.reshape(y, {20})) == Shape({20}));
  REQUIRE_THROWS(b.reshape(y, {21}), TypeError);
  REQUIRE(b.dtype(b.greater_than(x, b.constant(0.0))) == DType::Bool);
  REQUIRE(b.dtype(b.mul(b.greater_than(x, b.constant(0.0)), x)) == DType::Float32);
  REQUIRE(b.dtype(b.add(x, b.input({5}, DType::Float64))) == DType::Float64);
  REQUIRE_THROWS(b.matmul(m1, b.input({4, 2}, DType::Float64)), TypeError);
}

TEST(def_use_and_reachability) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({4});
  NodeId y = b.mul(x, x);
  NodeId z = b.add(y, x);
  NodeId dead = b.relu(x);
  (void)dead;
  const NodeId roots[1] = {z};
  auto mark = reachable_from(g, roots);
  REQUIRE(mark[x] && mark[y] && mark[z] && !mark[dead]);
  auto du = def_use(g, mark);
  REQUIRE_EQ(du.use_count[x], 3u);
  REQUIRE_EQ(du.use_count[y], 1u);
  REQUIRE_EQ(du.use_count[z], 0u);
  REQUIRE_EQ(du.edges, uint64_t(4));
  auto order = topo_order(g, roots);
  REQUIRE(order == std::vector<NodeId>({x, y, z}));
}

TEST(autodiff_accumulates_and_shares) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({4});
  NodeId w = b.input({4});
  NodeId t = b.mul(x, w);
  NodeId u = b.add(t, x);  // x reaches the loss through two paths
  NodeId loss = b.sum(u);
  const NodeId wrt[2] = {x, w};
  auto gr = backward(b, loss, wrt);
  REQUIRE_EQ(gr.grads.size(), size_t(2));
  REQUIRE(b.shape(gr.grads[0]) == Shape({4}));
  REQUIRE(b.shape(gr.grads[1]) == Shape({4}));
  REQUIRE_EQ(gr.stats.accumulations, uint64_t(1));  // the two x paths are added once
  REQUIRE(g.node(gr.grads[0]).op == OpCode::Add);
  // The forward intermediates are referenced, not copied: w is an operand of a
  // backward Mul node.
  auto du = def_use(g);
  REQUIRE(du.use_count[w] >= 2);
}

TEST(autodiff_zero_gradient_for_unused_input) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({3});
  NodeId y = b.input({3});
  NodeId loss = b.sum(b.relu(x));
  const NodeId wrt[2] = {x, y};
  auto gr = backward(b, loss, wrt);
  REQUIRE(g.node(gr.grads[1]).op == OpCode::Constant);
  REQUIRE(g.node(gr.grads[1]).attrs.value == 0.0);
  REQUIRE(b.shape(gr.grads[1]) == Shape({3}));
}

TEST(autodiff_requires_scalar_loss) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({3});
  const NodeId wrt[1] = {x};
  REQUIRE_THROWS(backward(b, x, wrt), std::logic_error);
}

TEST(autodiff_is_symbolic_and_linear) {
  // Backward node count grows linearly with depth for the chain task.
  std::vector<uint64_t> bwd;
  for (int depth : {8, 16, 32, 64}) {
    Graph g;
    GraphBuilder b(g);
    TaskParams p;
    p.task = "chain";
    p.n = 16;
    p.depth = depth;
    DagBackend be{b, DType::Float32};
    NodeId loss = build_task(be, p);
    auto gr = backward(b, loss, g.inputs());
    bwd.push_back(gr.stats.backward_nodes());
  }
  for (size_t i = 1; i < bwd.size(); ++i) {
    const double ratio = double(bwd[i]) / double(bwd[i - 1]);
    REQUIRE(ratio > 1.8 && ratio < 2.2);
  }
}

TEST(autodiff_survives_node_pool_reallocation) {
  // Regression: the backward sweep must not hold references into the node
  // pool while it appends nodes (the pool reallocates past its reserve).
  Graph g;
  GraphBuilder b(g);
  TaskParams p;
  p.task = "chain";
  p.n = 4;
  p.depth = 2000;
  DagBackend be{b, DType::Float32};
  NodeId loss = build_task(be, p);
  auto gr = backward(b, loss, g.inputs());
  REQUIRE(g.size() > 8000);
  for (size_t i = 0; i < gr.grads.size(); ++i) REQUIRE(b.shape(gr.grads[i]) == Shape({4}));
  const NodeId outs[1] = {loss};
  SLP prog = compile(g, outs);
  REQUIRE(prog.num_ops() + g.inputs().size() == static_cast<size_t>(p.depth) * 5 + 2);
}

TEST(slp_liveness_releases_after_last_use) {
  Graph g;
  GraphBuilder b(g);
  NodeId x = b.input({4});
  NodeId a = b.relu(x);
  NodeId c = b.mul(a, a);
  NodeId d = b.add(c, x);
  NodeId loss = b.sum(d);
  const NodeId outs[1] = {loss};
  SLP p = compile(g, outs);
  REQUIRE_EQ(p.instrs.size(), size_t(4));
  // a's last use is the Mul instruction; it is released there, not earlier.
  REQUIRE(p.instrs[1].dst == c);
  REQUIRE(p.instrs[1].release == std::vector<NodeId>({a}));
  REQUIRE(p.instrs[2].release == std::vector<NodeId>({c}));
  REQUIRE(p.instrs[3].release == std::vector<NodeId>({d}));
  REQUIRE(p.keep[x] && p.keep[loss]);
  REQUIRE(p.static_peak_live_bytes() < p.total_defined_bytes());
  SLP q = compile(g, outs, CompileOptions{false});
  for (const auto& i : q.instrs) REQUIRE(i.release.empty());
}

TEST(tree_copies_on_reuse_and_caps) {
  TreeBuilder tb(/*cap=*/0);
  TreeValue x = tb.input({4});
  TreeValue y = tb.mul(x, x);  // second use copies the leaf
  REQUIRE_EQ(tree_size(y.node()), uint64_t(3));
  TreeValue z = tb.mul(y, y);  // copies the whole subtree
  REQUIRE_EQ(tree_size(z.node()), uint64_t(7));
  REQUIRE(tb.stats().copies >= 4);

  TreeBuilder capped(/*cap=*/50);
  TreeValue v = capped.input({4});
  bool exceeded = false;
  try {
    for (int i = 0; i < 10; ++i) v = capped.mul(v, v);
  } catch (const TreeCapExceeded&) {
    exceeded = true;
  }
  REQUIRE(exceeded);
}

TEST(tree_backward_grows_without_sharing) {
  // For the chain, the DAG gradient is linear in depth, the tree is quadratic.
  uint64_t prev_tree = 0, prev_dag = 0;
  for (int depth : {4, 8, 16}) {
    TaskParams p;
    p.task = "chain";
    p.n = 4;
    p.depth = depth;
    TreeBuilder tb;
    TreeBackend tbe{tb, DType::Float32};
    TreeValue loss = build_task(tbe, p);
    auto tg = tree_backward(tb, loss.node(), tb.num_inputs());
    Graph g;
    GraphBuilder b(g);
    DagBackend dbe{b, DType::Float32};
    NodeId dl = build_task(dbe, p);
    auto dg = backward(b, dl, g.inputs());
    if (prev_tree) {
      REQUIRE(double(tg.grad_nodes) / double(prev_tree) > 3.0);        // ~4x per doubling
      REQUIRE(double(dg.stats.backward_nodes()) / double(prev_dag) < 2.3);  // ~2x per doubling
    }
    prev_tree = tg.grad_nodes;
    prev_dag = dg.stats.backward_nodes();
  }
}
