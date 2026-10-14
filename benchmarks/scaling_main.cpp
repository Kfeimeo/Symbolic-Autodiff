// Torch-free micro-benchmark: graph construction, symbolic differentiation
// and SLP compilation cost per node for growing chain / diamond graphs.
// Each size is repeated on fresh graphs; medians are reported.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sad/autodiff.hpp"
#include "sad/bench.hpp"
#include "sad/slp.hpp"
#include "sad/tasks.hpp"

using namespace sad;

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
  const bool hashcons = !(argc > 2 && std::string(argv[2]) == "nohc");
  std::printf("hash-consing %s\n", hashcons ? "on" : "off");
  std::printf("%-8s %-7s %-8s %-8s %-8s %-10s %-10s %-10s %-10s %-10s\n", "task", "depth", "fwd_n", "bwd_n",
              "edges", "build_us", "diff_us", "compile_us", "diff_ns/n", "edge_ns");
  for (const char* task : {"chain", "diamond"}) {
    for (int depth : {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
      std::vector<double> bt, dt, ct;
      size_t fwd = 0, bwd = 0, edges = 0;
      for (int rep = 0; rep < reps; ++rep) {
        TaskParams p;
        p.task = task;
        p.n = 8;
        p.depth = depth;
        p.width = 4;
        Timer t;
        Graph g(hashcons);
        GraphBuilder b(g);
        DagBackend be{b, p.dtype};
        const NodeId loss = build_task(be, p);
        bt.push_back(t.us());
        fwd = g.size();
        t.reset();
        GradResult gr = backward(b, loss, g.inputs());
        dt.push_back(t.us());
        bwd = gr.stats.backward_nodes();
        t.reset();
        std::vector<NodeId> outs{loss};
        outs.insert(outs.end(), gr.grads.begin(), gr.grads.end());
        SLP s = compile(g, outs);
        ct.push_back(t.us());
        edges = s.edges;
      }
      const Summary B = summarize(bt), D = summarize(dt), C = summarize(ct);
      std::printf("%-8s %-7d %-8zu %-8zu %-8zu %-10.1f %-10.1f %-10.1f %-10.0f %-10.0f\n", task, depth, fwd, bwd,
                  edges, B.median, D.median, C.median, D.median * 1000 / static_cast<double>(fwd),
                  D.median * 1000 / static_cast<double>(edges));
      std::fflush(stdout);
    }
  }
  return 0;
}
