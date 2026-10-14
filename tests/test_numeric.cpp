// Numerical tests: the DAG executor + symbolic VJP against LibTorch Autograd.
#include "sad/analysis.hpp"
#include "sad/autodiff.hpp"
#include "sad/executor.hpp"
#include "sad/profiler.hpp"
#include "sad/slp.hpp"
#include "sad/tasks.hpp"
#include "sad/torch_baseline.hpp"
#include "testing.hpp"

using namespace sad;

namespace {

constexpr double kRtol = 1e-9;
constexpr double kAtol = 1e-10;

struct Fixture {
  Graph g{true};
  GraphBuilder b{g};
  std::vector<torch::Tensor> inputs;  // same order as g.inputs()

  NodeId input(torch::Tensor t) {
    inputs.push_back(t.to(torch::kDouble).contiguous());
    return b.input(Shape(t.sizes().begin(), t.sizes().end()), DType::Float64);
  }

  // Runs forward+backward through the SLP executor and compares against
  // torch autograd computed by `ref` on fresh leaves.
  void check(NodeId loss, const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& ref) {
    auto gr = backward(b, loss, g.inputs());
    std::vector<NodeId> outs{loss};
    outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
    SLP p = compile(g, outs);
    Executor ex(p);
    ex.bind_inputs(inputs);
    ex.run(/*track_memory=*/true);

    auto leaves = make_leaves(inputs);
    torch::Tensor tl = ref(leaves);
    auto tg = torch::autograd::grad({tl}, leaves, {}, false, false, /*allow_unused=*/true);

    auto cl = compare(ex.value(loss), tl, kRtol, kAtol);
    REQUIRE(cl.allclose);
    for (size_t i = 0; i < leaves.size(); ++i) {
      torch::Tensor want = tg[i].defined() ? tg[i] : torch::zeros_like(leaves[i]);
      auto c = compare(ex.value(gr.grads[i]), want, kRtol, kAtol);
      if (!c.allclose) {
        std::ostringstream os;
        os << "gradient " << i << " mismatch: max_abs_err=" << c.max_abs_err;
        throw testing::Failure(os.str());
      }
      REQUIRE(ex.value(gr.grads[i]).sizes() == leaves[i].sizes());
    }
    // Everything that is not an input/output has been released.
    for (NodeId id = 0; id < g.size(); ++id) {
      if (ex.has_value(id)) REQUIRE(p.keep[id]);
    }
  }
};

torch::Tensor rnd(at::IntArrayRef shape, uint64_t seed) {
  torch::manual_seed(seed);
  return torch::randn(shape, torch::kDouble);
}

}  // namespace

TEST(numeric_broadcast_add_mul) {
  Fixture f;
  NodeId x = f.input(rnd({3, 1, 5}, 1));
  NodeId y = f.input(rnd({4, 5}, 2));
  NodeId z = f.input(rnd({5}, 3));
  NodeId loss = f.b.sum(f.b.mul(f.b.add(x, y), z));
  f.check(loss, [](const auto& v) { return ((v[0] + v[1]) * v[2]).sum(); });
}

TEST(numeric_repeated_operand) {
  Fixture f;
  NodeId x = f.input(rnd({6}, 4));
  NodeId sq = f.b.mul(x, x);
  NodeId loss = f.b.sum(f.b.add(f.b.mul(sq, x), f.b.add(x, x)));
  f.check(loss, [](const auto& v) { return (v[0] * v[0] * v[0] + v[0] + v[0]).sum(); });
}

TEST(numeric_relu_at_zero) {
  Fixture f;
  // Exact zeros, negatives and positives.
  torch::Tensor t = torch::tensor({-2.0, -1.0, 0.0, 0.0, 1.0, 2.0}, torch::kDouble);
  NodeId x = f.input(t);
  NodeId w = f.input(rnd({6}, 5));
  NodeId loss = f.b.sum(f.b.mul(f.b.relu(x), w));
  f.check(loss, [](const auto& v) { return (torch::relu(v[0]) * v[1]).sum(); });
  // The gradient at exactly zero must be zero (not w).
  Graph g2;
  GraphBuilder b2(g2);
  NodeId x2 = b2.input({6}, DType::Float64);
  NodeId l2 = b2.sum(b2.relu(x2));
  auto gr = backward(b2, l2, g2.inputs());
  SLP p = compile(g2, gr.grads);
  Executor ex(p);
  ex.bind(x2, t);
  ex.run();
  auto gx = ex.value(gr.grads[0]);
  REQUIRE(gx[2].item<double>() == 0.0);
  REQUIRE(gx[3].item<double>() == 0.0);
  REQUIRE(gx[4].item<double>() == 1.0);
  REQUIRE(gx[0].item<double>() == 0.0);
}

TEST(numeric_matmul_sizes) {
  for (auto [m, k, n] : std::vector<std::tuple<int64_t, int64_t, int64_t>>{{1, 1, 1}, {2, 3, 4}, {7, 5, 3}, {16, 32, 8}}) {
    Fixture f;
    NodeId a = f.input(rnd({m, k}, 10 + m));
    NodeId b = f.input(rnd({k, n}, 20 + n));
    NodeId c = f.input(rnd({n}, 30));
    NodeId loss = f.b.sum(f.b.relu(f.b.add(f.b.matmul(a, b), c)));
    f.check(loss, [](const auto& v) { return torch::relu(torch::matmul(v[0], v[1]) + v[2]).sum(); });
  }
}

TEST(numeric_transpose_and_sum_dims) {
  Fixture f;
  NodeId a = f.input(rnd({3, 4}, 41));
  NodeId b = f.input(rnd({3, 4}, 42));
  NodeId t = f.b.transpose(a);                       // [4,3]
  NodeId m = f.b.matmul(t, b);                       // [4,4]
  NodeId s0 = f.b.sum(m, {0}, false);                // [4]
  NodeId s1 = f.b.sum(m, {1}, true);                 // [4,1]
  NodeId mixed = f.b.add(f.b.mul(s1, s0), s1);       // [4,4]
  NodeId loss = f.b.sum(f.b.sum(mixed, {-1}, false));
  f.check(loss, [](const auto& v) {
    auto m = torch::matmul(v[0].t(), v[1]);
    auto s0 = m.sum(0);
    auto s1 = m.sum(1, true);
    return (s1 * s0 + s1).sum(-1).sum();
  });
}

TEST(numeric_diamond_accumulation) {
  Fixture f;
  NodeId x = f.input(rnd({5}, 50));
  NodeId w1 = f.input(rnd({5}, 51));
  NodeId w2 = f.input(rnd({5}, 52));
  NodeId a = f.b.relu(f.b.mul(x, w1));
  NodeId c = f.b.relu(f.b.mul(x, w2));
  NodeId d = f.b.add(f.b.mul(a, c), f.b.add(a, c));
  NodeId loss = f.b.sum(f.b.mul(d, x));
  f.check(loss, [](const auto& v) {
    auto a = torch::relu(v[0] * v[1]);
    auto c = torch::relu(v[0] * v[2]);
    return ((a * c + a + c) * v[0]).sum();
  });
}

TEST(numeric_all_tasks_match_autograd) {
  for (const char* task : {"chain", "sharing", "diamond", "nn", "mlp"}) {
    TaskParams p;
    p.task = task;
    p.n = 32;
    p.depth = 4;
    p.width = 3;
    p.sharing = 3;
    p.batch = 8;
    p.in = 12;
    p.hidden = 10;
    p.out = 6;
    p.dtype = DType::Float64;
    auto inputs = make_task_inputs(p, 123);
    Graph g;
    GraphBuilder b(g);
    DagBackend be{b, p.dtype};
    NodeId loss = build_task(be, p);
    auto gr = backward(b, loss, g.inputs());
    std::vector<NodeId> outs{loss};
    outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
    SLP prog = compile(g, outs);
    Executor ex(prog);
    ex.bind_inputs(inputs);
    ex.run();
    auto ref = torch_reference(p, inputs);
    REQUIRE(compare(ex.value(loss), ref.loss, kRtol, kAtol).allclose);
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto c = compare(ex.value(gr.grads[i]), ref.grads[i], kRtol, kAtol);
      if (!c.allclose) {
        std::ostringstream os;
        os << task << ": grad " << i << " max_abs_err=" << c.max_abs_err;
        throw testing::Failure(os.str());
      }
    }
    // Hash-consing off must give identical numbers.
    Graph g2(false);
    GraphBuilder b2(g2);
    DagBackend be2{b2, p.dtype};
    NodeId loss2 = build_task(be2, p);
    auto gr2 = backward(b2, loss2, g2.inputs());
    std::vector<NodeId> outs2{loss2};
    outs2.insert(outs2.end(), gr2.grads.begin(), gr2.grads.end());
    SLP prog2 = compile(g2, outs2);
    Executor ex2(prog2);
    ex2.bind_inputs(inputs);
    ex2.run();
    REQUIRE(g2.size() >= g.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      REQUIRE(compare(ex2.value(gr2.grads[i]), ref.grads[i], kRtol, kAtol).allclose);
    }
  }
}

TEST(executor_liveness_and_rerun) {
  TaskParams p;
  p.task = "mlp";
  p.batch = 16;
  p.in = 32;
  p.hidden = 32;
  p.out = 8;
  p.depth = 3;
  auto inputs = make_task_inputs(p, 7);
  Graph g;
  GraphBuilder b(g);
  DagBackend be{b, p.dtype};
  NodeId loss = build_task(be, p);
  auto gr = backward(b, loss, g.inputs());
  std::vector<NodeId> outs{loss};
  outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
  SLP keep_all = compile(g, outs, CompileOptions{false});
  SLP release = compile(g, outs);
  Executor e1(keep_all), e2(release);
  e1.bind_inputs(inputs);
  e2.bind_inputs(inputs);
  e1.run(true);
  e2.run(true);
  REQUIRE(e2.stats().peak_live_bytes < e1.stats().peak_live_bytes);
  REQUIRE(e2.stats().releases > 0);
  REQUIRE_EQ(e1.stats().releases, uint64_t(0));
  REQUIRE_EQ(e2.stats().instructions, uint64_t(release.instrs.size()));
  // Outputs survive, intermediates do not.
  for (NodeId id = 0; id < g.size(); ++id) {
    if (release.keep[id]) REQUIRE(e2.has_value(id));
    else REQUIRE(!e2.has_value(id));
  }
  // A second run reproduces the same values.
  torch::Tensor first = e2.value(gr.grads[0]).clone();
  e2.reset();
  REQUIRE(!e2.has_value(loss));
  e2.run();
  REQUIRE(torch::equal(first, e2.value(gr.grads[0])));
  REQUIRE(torch::allclose(e1.value(loss), e2.value(loss)));
  // Forward and backward halves can be run separately on the same program.
  e2.reset();
  const size_t split = e2.instr_index_of(loss) + 1;
  e2.run_range(0, split);
  REQUIRE(e2.has_value(loss));
  REQUIRE(!e2.has_value(gr.grads[0]));
  e2.run_range(split, release.instrs.size());
  REQUIRE(torch::equal(first, e2.value(gr.grads[0])));
}

TEST(profiler_counts_aten_ops) {
  torch::Tensor a = torch::ones({4, 4});
  OpProfile prof;
  {
    OpProfileScope scope(true);
    auto c = at::add(a, a);
    auto d = at::mul(c, a);
    (void)d;
    prof = scope.result();
  }
  REQUIRE(prof.aten_ops >= 2);
}

TEST(tree_backward_matches_dag_numerically) {
  TaskParams p;
  p.task = "diamond";
  p.n = 6;
  p.depth = 2;
  p.width = 2;
  p.dtype = DType::Float64;
  auto inputs = make_task_inputs(p, 99);
  TreeBuilder tb;
  TreeBackend tbe{tb, p.dtype};
  TreeValue loss = build_task(tbe, p);
  auto tg = tree_backward(tb, loss.node(), tb.num_inputs());
  auto ref = torch_reference(p, inputs);
  REQUIRE(compare(eval_tree(loss.node(), inputs), ref.loss, kRtol, kAtol).allclose);
  for (size_t i = 0; i < inputs.size(); ++i) {
    REQUIRE(tg.grads[i] != nullptr);
    REQUIRE(compare(eval_tree(*tg.grads[i], inputs), ref.grads[i], kRtol, kAtol).allclose);
  }
}

TEST(executor_runs_on_cuda_when_available) {
  // Device plumbing: inputs on CUDA => constants and all values on CUDA, and
  // the result matches the CPU run. Skipped (passes) without a GPU.
  if (!torch::cuda::is_available()) return;
  TaskParams p;
  p.task = "mlp";
  p.batch = 8;
  p.in = 16;
  p.hidden = 16;
  p.out = 4;
  p.depth = 2;
  auto cpu_inputs = make_task_inputs(p, 5, at::kCPU);
  auto gpu_inputs = make_task_inputs(p, 5, at::Device("cuda"));
  Graph g;
  GraphBuilder b(g);
  DagBackend be{b, p.dtype};
  NodeId loss = build_task(be, p);
  auto gr = backward(b, loss, g.inputs());
  std::vector<NodeId> outs{loss};
  outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
  SLP prog = compile(g, outs);
  Executor cpu(prog), gpu(prog);
  cpu.bind_inputs(cpu_inputs);
  gpu.bind_inputs(gpu_inputs);
  cpu.run();
  gpu.run();
  REQUIRE(gpu.device().is_cuda());
  REQUIRE(gpu.value(loss).device().is_cuda());
  for (size_t i = 0; i < gr.grads.size(); ++i) {
    REQUIRE(gpu.value(gr.grads[i]).device().is_cuda());
    REQUIRE(compare(gpu.value(gr.grads[i]).cpu(), cpu.value(gr.grads[i]), 1e-4, 1e-5).allclose);
  }
  auto ref = torch_reference(p, gpu_inputs);
  REQUIRE(compare(gpu.value(loss).cpu(), ref.loss.cpu(), 1e-4, 1e-5).allclose);
}
