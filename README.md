# Symbolic-Autodiff: Tensor DAG + Symbolic VJP + SLP Executor

A C++20 experiment that measures what a hash-consed **Tensor DAG** with
**symbolic reverse-mode differentiation** and a **straight-line-program (SLP)
executor** buys you, compared with

* **A.** naive tree-AST symbolic differentiation (no sub-expression sharing), and
* **C.** LibTorch eager execution with native Autograd.

All three run the identical computations on the same ATen kernels, device,
dtype, shapes and thread count. The custom differentiation engine does not use
LibTorch Autograd; Autograd is only the control group and the correctness
reference. `REPORT.md` contains the measurements and the analysis.

## Layout

| Path | Contents |
|------|----------|
| `include/sad/ir.hpp`, `src/ir.cpp` | Immutable typed Tensor DAG: `NodeId`, `OpCode`, `Attrs`, `Node`, `Graph` |
| `include/sad/hashcons.hpp`, `src/hashcons.cpp` | Structural hash-consing (op, inputs, attrs, shape, dtype; no algebraic rewriting) |
| `include/sad/type_inference.hpp` | Shape/dtype inference and checking for every op (broadcasting, matmul, sum, …) |
| `include/sad/graph_builder.hpp` | Deterministic node construction API |
| `include/sad/analysis.hpp` | Reachability, topological order, def-use analysis, op histograms, dumps |
| `include/sad/autodiff.hpp`, `src/autodiff.cpp` | Symbolic reverse-mode VJP on the DAG (same graph, same NodeId space) |
| `include/sad/slp.hpp`, `src/slp.cpp` | SLP compilation and liveness (last-use release lists) |
| `include/sad/executor.hpp`, `src/executor.cpp` | ATen dispatch, `NodeId -> at::Tensor` map, memory accounting |
| `include/sad/profiler.hpp` | ATen operator counting (RecordFunction) and malloc sampling, backend neutral |
| `include/sad/tree_ast.hpp`, `src/tree_ast.cpp` | Baseline A: tree AST + copying derivative rules + node cap |
| `include/sad/torch_baseline.hpp` | Baseline C: eager LibTorch + `torch::autograd::grad` |
| `include/sad/tasks.hpp` | Parametrised tasks written once, instantiated for all three backends |
| `tests/` | Unit tests (structural + numerical against Autograd) |
| `benchmarks/bench_main.cpp` | Benchmark driver, CSV output |
| `scripts/` | LibTorch download, one-shot benchmark run, plotting |
| `results/` | CSV, logs and figures of the last run |

## Building

Requirements: CMake ≥ 3.18, a C++20 compiler (tested with GCC 13), LibTorch
(CPU build, tested with 2.14.1). Optional: Python 3 with pandas + matplotlib
for the plots.

```bash
scripts/setup_libtorch.sh            # downloads LibTorch CPU into third_party/libtorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # or -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j$(nproc)
./build/sad_tests                    # 21 unit tests
./build/sad_bench --suite quick      # a fast sanity benchmark
```

`scripts/run_benchmarks.sh [suite]` builds, tests, runs the suite, writes
`results/<suite>.csv` + `results/<suite>.log` and renders figures.

```
sad_bench [--suite all|quick|chain|sharing|diamond|nn|mlp|scaling] [--csv path]
          [--threads N] [--repeats N] [--warmup N] [--tree-cap N] [--seed N]
          [--dtype f32|f64] [--skip-tree] [--skip-nohc]
```

## Design summary

**DAG.** Nodes are stored in a `std::vector<Node>`; a node's inputs always have
smaller ids, so the vector is a topological order. Every node is a symbolic
tensor value (OpCode, input ids, attributes, shape, dtype). Operators are pure:
no in-place mutation, no randomness, no control flow. Supported ops: `Input`,
`Constant` (scalar fill with explicit shape/dtype), `Add`, `Mul` (NumPy
broadcasting), `MatMul` (2-D), `ReLU`, `Sum` (dims, keepdim), `Transpose`
(2-D), `BroadcastTo`, `SumToShape`, `Reshape`, `GreaterThan`, `OnesLike`.

**Hash-consing.** `Graph::emplace` looks the structural hash up in a
multimap and compares op, input ids, attributes, shape and dtype. Only
identical structure is merged (`Add(a,b)` ≠ `Add(b,a)`). Inputs carry a unique
index and never merge; constants merge by value bits, shape and dtype. The
builder elides `BroadcastTo/SumToShape/Reshape` when the target shape already
equals the operand shape (these are identities by definition, not algebraic
rewrites).

**Symbolic VJP.** `backward(builder, loss, wrt)` sweeps the forward ids in
descending order (reverse topological order), applies a local rule per op and
accumulates contributions with `Add` nodes. Backward nodes are created in the
same graph through the same hash-consing table, so e.g. `Transpose(W)` or a
ReLU mask requested twice exists once, and backward nodes reference forward
intermediates directly. No Jacobian is built, no subgraph is copied, no tensor
is computed. Work is O(forward nodes + edges) with bounded-size local rules.

**SLP + executor.** `compile(graph, outputs)` emits the reachable non-input
nodes in topological order and computes the last use of every value; a value
is released right after its last use unless it is an input or output.
`Executor` maps `NodeId -> at::Tensor`, dispatches one ATen call per
instruction and can run a prefix (forward) and the rest (backward) separately
for timing. Memory is reported at three levels: graph node bytes, logical live
tensor bytes (deduplicated by storage; views cost nothing) and the process
malloc footprint sampled at operator boundaries.

**Tree baseline.** Expressions are trees: the first operator consuming a value
takes it, every further use deep-copies it. The derivative rules are the same
VJP rules but every adjoint is materialised as a fresh tree, copying the
sibling subtrees it refers to. A node cap aborts runaway expansion.

**Fairness.** The torch backend calls the same `at::add/mul/matmul/relu/sum`
functions on the same tensors, with the same thread count. Operator counts for
both backends come from the same `at::RecordFunction` global callback.

## Tests

`sad_tests` covers hash-consing semantics (including the disabled mode),
shape/dtype inference and errors, def-use analysis, gradient accumulation and
sharing, zero gradients, linear growth of the backward graph, SLP liveness,
tree copy/cap behaviour, and numerical agreement with LibTorch Autograd in
float64 (rtol 1e-9) for broadcasting, repeated operands, ReLU at exact zeros,
several matmul sizes, transposes and partial sums, diamonds, and all benchmark
tasks with hash-consing on and off.
