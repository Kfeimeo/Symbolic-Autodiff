// Benchmark driver: Tensor DAG (hash-consing on/off), naive tree AST and
// LibTorch eager Autograd on identical, parametrised tasks.
#include <torch/torch.h>

#include <malloc.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "sad/analysis.hpp"
#include "sad/autodiff.hpp"
#include "sad/bench.hpp"
#include "sad/executor.hpp"
#include "sad/profiler.hpp"
#include "sad/slp.hpp"
#include "sad/tasks.hpp"
#include "sad/torch_baseline.hpp"
#include "sad/tree_ast.hpp"

using namespace sad;

namespace {

struct Options {
  std::string suite = "all";
  std::string csv = "results/results.csv";
  int threads = 4;
  int repeats = 20;
  int warmup = 3;
  uint64_t tree_cap = 2000000;
  uint64_t seed = 42;
  int graph_repeats = 5;  // build/diff/compile are repeated; medians reported
  bool skip_tree = false;
  bool skip_nohc = false;
  bool torch_first = false;   // run the torch baseline before the DAG backends
  bool malloc_stable = false; // disable glibc mmap/trim so freed memory is reused
  bool verbose = false;
  std::string dtype = "f32";
  // Single configuration overrides (--task selects this mode).
  std::string task;
  int64_t n = -1, batch = -1, in = -1, hidden = -1, out = -1;
  int depth = -1, width = -1, sharing = -1;
};

double rtol_for(DType dt) { return dt == DType::Float64 ? 1e-9 : 1e-4; }
double atol_for(DType dt) { return dt == DType::Float64 ? 1e-10 : 1e-6; }

Row base_row(const TaskParams& p, const Options& o, const char* backend) {
  Row r;
  r.set("task", p.task);
  r.set("label", p.label());
  r.set("backend", std::string(backend));
  r.set("n", p.n);
  r.set("depth", p.depth);
  r.set("width", p.width);
  r.set("sharing", p.sharing);
  r.set("batch", p.batch);
  r.set("in", p.in);
  r.set("hidden", p.hidden);
  r.set("out", p.out);
  r.set("dtype", std::string(dtype_name(p.dtype)));
  r.set("threads", o.threads);
  r.set("repeats", o.repeats);
  r.set("warmup", o.warmup);
  r.set("torch_first", o.torch_first);
  r.set("malloc_stable", o.malloc_stable);
  return r;
}

void set_summary(Row& r, const std::string& prefix, const Summary& s) {
  r.set(prefix + "_median_us", s.median);
  r.set(prefix + "_p90_us", s.p90);
  r.set(prefix + "_min_us", s.min);
  r.set(prefix + "_mean_us", s.mean);
}

// ---------------------------------------------------------------------------
// B. Tensor DAG + symbolic VJP + SLP executor
// ---------------------------------------------------------------------------
Row run_dag(const TaskParams& p, const Options& o, bool hashcons,
            const std::vector<torch::Tensor>& inputs, const TorchResult& ref) {
  Row r = base_row(p, o, hashcons ? "dag" : "dag_nohc");
  r.set("hashcons", hashcons);

  // Graph construction, symbolic differentiation and SLP compilation are
  // repeated on fresh graphs; the first sample is reported as the cold cost
  // and the median as the steady cost.
  std::vector<double> build_s, diff_s, compile_s;
  for (int rep = 0; rep + 1 < o.graph_repeats; ++rep) {
    Timer t;
    Graph g2(hashcons);
    GraphBuilder b2(g2);
    DagBackend be2{b2, p.dtype};
    const NodeId loss2 = build_task(be2, p);
    build_s.push_back(t.us());
    t.reset();
    GradResult gr2 = backward(b2, loss2, g2.inputs());
    diff_s.push_back(t.us());
    t.reset();
    std::vector<NodeId> outs2{loss2};
    outs2.insert(outs2.end(), gr2.grads.begin(), gr2.grads.end());
    SLP full2 = compile(g2, outs2);
    const NodeId fwd_out2[1] = {loss2};
    SLP fwd2 = compile(g2, fwd_out2);
    compile_s.push_back(t.us());
  }

  // Build the forward DAG (the instance that is executed below).
  const uint64_t malloc0 = malloc_in_use_bytes();
  Timer t;
  Graph g(hashcons);
  GraphBuilder b(g);
  DagBackend be{b, p.dtype};
  const NodeId loss = build_task(be, p);
  const double build_us = t.us();
  const uint64_t fwd_nodes = g.size();
  const uint64_t fwd_hits = g.stats().hashcons_hits;

  // Symbolic reverse-mode differentiation (no tensor computation).
  t.reset();
  GradResult gr = backward(b, loss, g.inputs());
  const double diff_us = t.us();

  // SLP compilation with liveness, for the unified program and forward only.
  t.reset();
  std::vector<NodeId> outs{loss};
  outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
  SLP full = compile(g, outs);
  const NodeId fwd_out[1] = {loss};
  SLP fwd = compile(g, fwd_out);
  const double compile_us = t.us();
  const uint64_t graph_malloc = malloc_in_use_bytes() - malloc0;
  build_s.push_back(build_us);
  diff_s.push_back(diff_us);
  compile_s.push_back(compile_us);

  r.set("fwd_nodes", fwd_nodes);
  r.set("bwd_nodes", gr.stats.backward_nodes());
  r.set("total_nodes", g.size());
  r.set("fwd_edges", fwd.edges);
  r.set("total_edges", full.edges);
  r.set("hashcons_hits_fwd", fwd_hits);
  r.set("hashcons_hits_bwd", gr.stats.hashcons_hits);
  r.set("ad_create_requests", gr.stats.create_requests);
  r.set("ad_visited", gr.stats.visited);
  r.set("ad_grad_contributions", gr.stats.grad_contributions);
  r.set("ad_accumulations", gr.stats.accumulations);
  r.set("slp_instrs_fwd", fwd.num_ops());
  r.set("slp_instrs_full", full.num_ops());
  r.set("graph_node_bytes", g.node_bytes());
  r.set("graph_malloc_bytes", graph_malloc);
  r.set("static_peak_live_bytes", full.static_peak_live_bytes());
  r.set("total_defined_bytes", full.total_defined_bytes());
  r.set("build_us", summarize(build_s).median);
  r.set("diff_us", summarize(diff_s).median);
  r.set("compile_us", summarize(compile_s).median);
  r.set("build_first_us", build_s.front());
  r.set("diff_first_us", diff_s.front());
  r.set("compile_first_us", compile_s.front());
  r.set("graph_repeats", o.graph_repeats);
  r.set("op_histogram", op_histogram_str(op_histogram(g)));

  // First (cold) execution of the unified program.
  Executor ex(full);
  ex.bind_inputs(inputs);
  t.reset();
  ex.run();
  r.set("first_exec_us", t.us());

  // Correctness against the autograd reference.
  {
    const double rtol = rtol_for(p.dtype), atol = atol_for(p.dtype);
    Closeness cl = compare(ex.value(loss), ref.loss, rtol, atol);
    double max_abs = 0, max_rel = 0;
    bool ok = true;
    for (size_t i = 0; i < gr.grads.size(); ++i) {
      Closeness c = compare(ex.value(gr.grads[i]), ref.grads[i], rtol, atol);
      max_abs = std::max(max_abs, c.max_abs_err);
      max_rel = std::max(max_rel, c.max_rel_err);
      ok = ok && c.allclose;
    }
    r.set("loss_allclose", cl.allclose);
    r.set("loss_abs_err", cl.max_abs_err);
    r.set("grad_allclose", ok);
    r.set("grad_max_abs_err", max_abs);
    r.set("grad_max_rel_err", max_rel);
  }

  // Steady-state timing of the unified program, split at the loss.
  const size_t split = ex.instr_index_of(loss) + 1;
  std::vector<double> fwd_s, bwd_s, tot_s;
  for (int i = 0; i < o.warmup + o.repeats; ++i) {
    ex.reset();
    Timer tt;
    ex.run_range(0, split);
    const double f = tt.us();
    ex.run_range(split, full.num_ops());
    const double tot = tt.us();
    if (i >= o.warmup) {
      fwd_s.push_back(f);
      bwd_s.push_back(tot - f);
      tot_s.push_back(tot);
    }
  }
  set_summary(r, "fwd", summarize(fwd_s));
  set_summary(r, "bwd", summarize(bwd_s));
  set_summary(r, "total", summarize(tot_s));

  // Forward-only program (nothing kept for backward).
  Executor exf(fwd);
  exf.bind_inputs(inputs);
  std::vector<double> fo_s;
  for (int i = 0; i < o.warmup + o.repeats; ++i) {
    exf.reset();
    Timer tt;
    exf.run();
    if (i >= o.warmup) fo_s.push_back(tt.us());
  }
  set_summary(r, "fwd_only", summarize(fo_s));

  // Instrumented pass: ATen operator invocations, live bytes, malloc peak.
  ex.reset();
  {
    OpProfileScope scope(true);
    ex.run(true);
    OpProfile prof = scope.result();
    r.set("aten_ops_full", prof.aten_ops);
    r.set("peak_malloc_delta_bytes", prof.peak_malloc_delta());
  }
  r.set("peak_live_bytes", ex.stats().peak_live_bytes);
  r.set("peak_live_values", ex.stats().peak_live_values);
  r.set("end_live_bytes", ex.stats().end_live_bytes);
  r.set("releases", ex.stats().releases);
  exf.reset();
  {
    OpProfileScope scope(true);
    exf.run(true);
    OpProfile prof = scope.result();
    r.set("aten_ops_fwd", prof.aten_ops);
    r.set("peak_malloc_delta_fwd_bytes", prof.peak_malloc_delta());
  }
  r.set("peak_live_bytes_fwd", exf.stats().peak_live_bytes);
  r.set("peak_rss_bytes", peak_rss_bytes());
  return r;
}

// ---------------------------------------------------------------------------
// C. LibTorch eager Autograd
// ---------------------------------------------------------------------------
Row run_torch(const TaskParams& p, const Options& o, const std::vector<torch::Tensor>& inputs) {
  Row r = base_row(p, o, "torch");
  auto leaves = make_leaves(inputs);

  // First (cold) forward + backward.
  Timer t;
  {
    TorchBackend be{leaves};
    torch::Tensor loss = build_task(be, p);
    auto grads = torch::autograd::grad({loss}, leaves, {}, false, false, true);
  }
  r.set("first_exec_us", t.us());

  std::vector<double> fwd_s, bwd_s, tot_s, nograd_s;
  torch::Tensor loss;
  std::vector<torch::Tensor> grads;
  for (int i = 0; i < o.warmup + o.repeats; ++i) {
    loss.reset();
    grads.clear();
    Timer tt;
    TorchBackend be{leaves};
    loss = build_task(be, p);
    const double f = tt.us();
    grads = torch::autograd::grad({loss}, leaves, {}, false, false, true);
    const double tot = tt.us();
    if (i >= o.warmup) {
      fwd_s.push_back(f);
      bwd_s.push_back(tot - f);
      tot_s.push_back(tot);
    }
  }
  set_summary(r, "fwd", summarize(fwd_s));
  set_summary(r, "bwd", summarize(bwd_s));
  set_summary(r, "total", summarize(tot_s));

  // Forward without autograd recording: the pure eager execution cost.
  for (int i = 0; i < o.warmup + o.repeats; ++i) {
    loss.reset();
    torch::NoGradGuard ng;
    Timer tt;
    TorchBackend be{inputs};
    loss = build_task(be, p);
    if (i >= o.warmup) nograd_s.push_back(tt.us());
  }
  set_summary(r, "fwd_only", summarize(nograd_s));

  // Instrumented passes.
  loss.reset();
  grads.clear();
  {
    OpProfileScope scope(true);
    TorchBackend be{leaves};
    loss = build_task(be, p);
    grads = torch::autograd::grad({loss}, leaves, {}, false, false, true);
    OpProfile prof = scope.result();
    r.set("aten_ops_full", prof.aten_ops);
    r.set("peak_malloc_delta_bytes", prof.peak_malloc_delta());
  }
  loss.reset();
  grads.clear();
  {
    OpProfileScope scope(true);
    torch::NoGradGuard ng;
    TorchBackend be{inputs};
    loss = build_task(be, p);
    OpProfile prof = scope.result();
    r.set("aten_ops_fwd", prof.aten_ops);
    r.set("peak_malloc_delta_fwd_bytes", prof.peak_malloc_delta());
  }
  loss.reset();
  {
    OpProfileScope scope(false);
    TorchBackend be{leaves};
    loss = build_task(be, p);
    r.set("aten_ops_fwd_grad_mode", scope.result().aten_ops);
  }
  r.set("peak_rss_bytes", peak_rss_bytes());
  return r;
}

// ---------------------------------------------------------------------------
// A. Naive tree AST symbolic differentiation
// ---------------------------------------------------------------------------
Row run_tree(const TaskParams& p, const Options& o) {
  Row r = base_row(p, o, "tree");
  r.set("tree_cap", o.tree_cap);
  const uint64_t malloc0 = malloc_in_use_bytes();
  TreeBuilder tb(o.tree_cap);
  TreeBackend be{tb, p.dtype};
  bool capped = false;
  double build_us = 0, diff_us = 0;
  uint64_t fwd_tree_nodes = 0, grad_nodes = 0, bytes = 0, contributions = 0;
  Timer t;
  try {
    TreeValue loss = build_task(be, p);
    build_us = t.us();
    fwd_tree_nodes = tree_size(loss.node());
    bytes = tree_bytes(loss.node());
    t.reset();
    TreeGradResult tg = tree_backward(tb, loss.node(), tb.num_inputs());
    diff_us = t.us();
    grad_nodes = tg.grad_nodes;
    contributions = tg.contributions;
    for (const auto& gt : tg.grads) {
      if (gt) bytes += tree_bytes(*gt);
    }
  } catch (const TreeCapExceeded&) {
    capped = true;
    if (build_us == 0) build_us = t.us();
    else diff_us = t.us();
  }
  r.set("tree_capped", capped);
  r.set("tree_nodes_created", tb.stats().nodes_created);
  r.set("tree_copies", tb.stats().copies);
  r.set("fwd_nodes", fwd_tree_nodes);
  r.set("bwd_nodes", grad_nodes);
  r.set("total_nodes", fwd_tree_nodes + grad_nodes);
  r.set("ad_grad_contributions", contributions);
  r.set("build_us", build_us);
  r.set("diff_us", diff_us);
  r.set("graph_node_bytes", bytes);
  r.set("graph_malloc_bytes", malloc_in_use_bytes() - malloc0);
  return r;
}

void print_row(const Row& r) {
  auto get = [&](const char* k) -> std::string {
    const std::string* v = r.get(k);
    return v ? *v : "-";
  };
  auto num = [&](const char* k) -> double {
    const std::string* v = r.get(k);
    return v ? std::atof(v->c_str()) : 0.0;
  };
  const std::string be = get("backend");
  std::printf("  %-9s", be.c_str());
  if (be == "tree") {
    std::printf(" fwd_nodes=%s grad_nodes=%s created=%s build=%s diff=%s bytes=%s%s\n",
                get("fwd_nodes").c_str(), get("bwd_nodes").c_str(), get("tree_nodes_created").c_str(),
                format_us(num("build_us")).c_str(), format_us(num("diff_us")).c_str(),
                format_bytes(num("graph_node_bytes")).c_str(), num("tree_capped") ? " CAPPED" : "");
  } else if (be == "torch") {
    std::printf(" fwd=%s bwd=%s total=%s (p90 %s) fwd_nograd=%s ops=%s peak_malloc=%s\n",
                format_us(num("fwd_median_us")).c_str(), format_us(num("bwd_median_us")).c_str(),
                format_us(num("total_median_us")).c_str(), format_us(num("total_p90_us")).c_str(),
                format_us(num("fwd_only_median_us")).c_str(), get("aten_ops_full").c_str(),
                format_bytes(num("peak_malloc_delta_bytes")).c_str());
  } else {
    std::printf(" nodes=%s+%s build=%s diff=%s compile=%s fwd=%s bwd=%s total=%s (p90 %s) ops=%s live=%s malloc=%s grad_ok=%s err=%s\n",
                get("fwd_nodes").c_str(), get("bwd_nodes").c_str(), format_us(num("build_us")).c_str(),
                format_us(num("diff_us")).c_str(), format_us(num("compile_us")).c_str(),
                format_us(num("fwd_median_us")).c_str(), format_us(num("bwd_median_us")).c_str(),
                format_us(num("total_median_us")).c_str(), format_us(num("total_p90_us")).c_str(),
                get("aten_ops_full").c_str(), format_bytes(num("peak_live_bytes")).c_str(),
                format_bytes(num("peak_malloc_delta_bytes")).c_str(), get("grad_allclose").c_str(),
                get("grad_max_rel_err").c_str());
  }
}

void run_config(const TaskParams& p, const Options& o, CsvTable& csv, const std::string& suite,
                bool quiet = false) {
  if (!quiet) std::printf("%s\n", p.label().c_str());
  std::fflush(stdout);
  auto inputs = make_task_inputs(p, o.seed);
  TorchResult ref = torch_reference(p, inputs);

  std::vector<Row> rows;
  if (o.torch_first) rows.push_back(run_torch(p, o, inputs));
  rows.push_back(run_dag(p, o, true, inputs, ref));
  if (!o.skip_nohc) rows.push_back(run_dag(p, o, false, inputs, ref));
  if (!o.torch_first) rows.push_back(run_torch(p, o, inputs));
  if (!o.skip_tree) rows.push_back(run_tree(p, o));
  for (auto& r : rows) {
    r.set("suite", suite);
    csv.add(r);
    if (!quiet) print_row(r);
  }
  std::fflush(stdout);
}

TaskParams make(const std::string& task, DType dt) {
  TaskParams p;
  p.task = task;
  p.dtype = dt;
  return p;
}

std::vector<std::pair<std::string, TaskParams>> configs_for(const std::string& suite, DType dt) {
  std::vector<std::pair<std::string, TaskParams>> out;
  auto add = [&](const std::string& s, TaskParams p) { out.emplace_back(s, std::move(p)); };
  if (suite == "chain" || suite == "all") {
    for (int64_t n : {64, 1024, 65536}) {
      for (int depth : {4, 16, 64, 256}) {
        TaskParams p = make("chain", dt);
        p.n = n;
        p.depth = depth;
        add("chain", p);
      }
    }
  }
  if (suite == "sharing" || suite == "all") {
    for (int64_t n : {1024, 65536}) {
      for (int k : {1, 2, 4, 8}) {
        TaskParams p = make("sharing", dt);
        p.n = n;
        p.depth = 8;
        p.sharing = k;
        add("sharing", p);
      }
    }
  }
  if (suite == "diamond" || suite == "all") {
    for (int depth : {4, 16}) {
      for (int width : {2, 4, 8}) {
        TaskParams p = make("diamond", dt);
        p.n = 4096;
        p.depth = depth;
        p.width = width;
        add("diamond", p);
      }
    }
  }
  if (suite == "nn" || suite == "all") {
    for (auto [b, i, o] : std::vector<std::tuple<int64_t, int64_t, int64_t>>{{32, 64, 32}, {256, 512, 256}, {1024, 1024, 1024}}) {
      TaskParams p = make("nn", dt);
      p.batch = b;
      p.in = i;
      p.out = o;
      add("nn", p);
    }
  }
  if (suite == "mlp" || suite == "all") {
    for (int64_t hidden : {64, 256, 1024}) {
      for (int layers : {2, 4, 8}) {
        TaskParams p = make("mlp", dt);
        p.batch = 128;
        p.in = hidden;
        p.hidden = hidden;
        p.out = 64;
        p.depth = layers;
        add("mlp", p);
      }
    }
  }
  if (suite == "scaling" || suite == "all") {
    // Tiny tensors: isolates graph-construction / differentiation cost and
    // framework per-op overhead from kernel time.
    for (int depth : {64, 128, 256, 512, 1024, 2048, 4096}) {
      TaskParams p = make("chain", dt);
      p.n = 8;
      p.depth = depth;
      add("scaling", p);
    }
    for (int depth : {16, 64, 256, 1024}) {
      TaskParams p = make("diamond", dt);
      p.n = 8;
      p.depth = depth;
      p.width = 4;
      add("scaling", p);
    }
    for (int layers : {8, 32, 128}) {
      TaskParams p = make("mlp", dt);
      p.batch = 4;
      p.in = 8;
      p.hidden = 8;
      p.out = 8;
      p.depth = layers;
      add("scaling", p);
    }
  }
  if (suite == "custom") return out;  // filled by main() from the overrides
  if (suite == "quick") {
    TaskParams c = make("chain", dt);
    c.n = 1024;
    c.depth = 16;
    add("quick", c);
    TaskParams s = make("sharing", dt);
    s.n = 1024;
    s.depth = 4;
    s.sharing = 4;
    add("quick", s);
    TaskParams d = make("diamond", dt);
    d.n = 1024;
    d.depth = 4;
    d.width = 4;
    add("quick", d);
    TaskParams nn = make("nn", dt);
    add("quick", nn);
    TaskParams m = make("mlp", dt);
    m.depth = 3;
    add("quick", m);
  }
  return out;
}

void usage() {
  std::printf(
      "usage: sad_bench [--suite all|quick|chain|sharing|diamond|nn|mlp|scaling] [--csv path]\n"
      "                 [--threads N] [--repeats N] [--warmup N] [--tree-cap N] [--seed N]\n"
      "                 [--graph-repeats N] [--dtype f32|f64] [--skip-tree] [--skip-nohc]\n"
      "                 [--torch-first] [--malloc-stable]\n"
      "       single configuration: --task chain|sharing|diamond|nn|mlp [--n N] [--depth N]\n"
      "                 [--width N] [--sharing N] [--batch N] [--in N] [--hidden N] [--out N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--suite") o.suite = next();
    else if (a == "--csv") o.csv = next();
    else if (a == "--threads") o.threads = std::stoi(next());
    else if (a == "--repeats") o.repeats = std::stoi(next());
    else if (a == "--warmup") o.warmup = std::stoi(next());
    else if (a == "--tree-cap") o.tree_cap = std::stoull(next());
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--dtype") o.dtype = next();
    else if (a == "--graph-repeats") o.graph_repeats = std::stoi(next());
    else if (a == "--torch-first") o.torch_first = true;
    else if (a == "--malloc-stable") o.malloc_stable = true;
    else if (a == "--task") o.task = next();
    else if (a == "--n") o.n = std::stoll(next());
    else if (a == "--depth") o.depth = std::stoi(next());
    else if (a == "--width") o.width = std::stoi(next());
    else if (a == "--sharing") o.sharing = std::stoi(next());
    else if (a == "--batch") o.batch = std::stoll(next());
    else if (a == "--in") o.in = std::stoll(next());
    else if (a == "--hidden") o.hidden = std::stoll(next());
    else if (a == "--out") o.out = std::stoll(next());
    else if (a == "--skip-tree") o.skip_tree = true;
    else if (a == "--skip-nohc") o.skip_nohc = true;
    else if (a == "--verbose") o.verbose = true;
    else {
      usage();
      return 2;
    }
  }
  const DType dt = o.dtype == "f64" ? DType::Float64 : DType::Float32;
  if (o.malloc_stable) {
    // Keep freed memory inside the process so that neither backend pays
    // mmap/munmap and page-fault costs that depend on run order.
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_TOP_PAD, 64 << 20);
  }
  at::set_num_threads(o.threads);
  at::set_num_interop_threads(1);
  torch::manual_seed(o.seed);
  std::printf("libtorch %s, threads=%d, repeats=%d, warmup=%d, dtype=%s, tree_cap=%llu\n",
              TORCH_VERSION, at::get_num_threads(), o.repeats, o.warmup, o.dtype.c_str(),
              static_cast<unsigned long long>(o.tree_cap));

  // Silent warm-up so that the first measured configuration does not pay for
  // lazy initialisation (thread pool, allocator, page faults).
  {
    Options w = o;
    w.repeats = 2;
    w.warmup = 1;
    w.tree_cap = 10000;
    CsvTable scratch;
    TaskParams p = make("mlp", dt);
    p.depth = 2;
    run_config(p, w, scratch, "warmup", /*quiet=*/true);
  }

  CsvTable csv;
  auto configs = configs_for(o.task.empty() ? o.suite : "custom", dt);
  if (!o.task.empty()) {
    TaskParams p = make(o.task, dt);
    if (o.n >= 0) p.n = o.n;
    if (o.depth >= 0) p.depth = o.depth;
    if (o.width >= 0) p.width = o.width;
    if (o.sharing >= 0) p.sharing = o.sharing;
    if (o.batch >= 0) p.batch = o.batch;
    if (o.in >= 0) p.in = o.in;
    if (o.hidden >= 0) p.hidden = o.hidden;
    if (o.out >= 0) p.out = o.out;
    configs.emplace_back("custom", p);
  }
  for (const auto& [suite, p] : configs) run_config(p, o, csv, suite);
  if (csv.size() == 0) {
    usage();
    return 2;
  }
  csv.write(o.csv);
  std::printf("wrote %zu rows to %s\n", csv.size(), o.csv.c_str());
  return 0;
}
