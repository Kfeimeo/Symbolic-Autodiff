// Parametrised benchmark tasks, written once against a generic "backend"
// interface and instantiated for the Tensor DAG, the naive tree AST and
// LibTorch eager Autograd, so all three run the identical computation.
#pragma once

#include <string>
#include <vector>

#include "sad/graph_builder.hpp"
#include "sad/tree_ast.hpp"

namespace sad {

struct TaskParams {
  std::string task = "chain";  // chain | sharing | diamond | nn | mlp
  int64_t n = 256;             // elementwise vector size
  int depth = 8;               // chain / sharing / diamond layers, mlp hidden layers
  int width = 4;               // diamond branches per layer
  int sharing = 1;             // sharing task: duplicate derivations per layer
  int64_t batch = 64, in = 128, hidden = 128, out = 64;  // nn / mlp
  DType dtype = DType::Float32;
  std::string label() const;
};

// Backend concept:
//   using Value = ...;
//   Value input(Shape, const char* role /* "x" | "w" | "b" */);  Value add(Value&, Value&);  Value mul(Value&, Value&);
//   Value matmul(Value&, Value&);  Value relu(Value&);
//   Value sum(Value&, std::vector<int64_t> dims, bool keepdim);
template <class B>
typename B::Value build_task(B& b, const TaskParams& p) {
  using V = typename B::Value;
  if (p.task == "chain") {
    // y_{i+1} = relu(y_i * w_i + b_i); loss = sum(y_D)
    V y = b.input({p.n}, "x");
    for (int i = 0; i < p.depth; ++i) {
      V w = b.input({p.n}, "w");
      V bias = b.input({p.n}, "b");
      V t = b.mul(y, w);
      V s = b.add(t, bias);
      y = b.relu(s);
    }
    return b.sum(y, {}, false);
  }
  if (p.task == "sharing") {
    // Each layer derives the expression (y_i * w_i + b_i) `sharing` times
    // (syntactic duplicates) and sums the copies: y_{i+1} = relu(sum_k copies).
    V y = b.input({p.n}, "x");
    for (int i = 0; i < p.depth; ++i) {
      V w = b.input({p.n}, "w");
      V bias = b.input({p.n}, "b");
      V acc{};
      for (int k = 0; k < p.sharing; ++k) {
        V t = b.mul(y, w);
        V s = b.add(t, bias);
        if (k == 0) acc = std::move(s);
        else acc = b.add(acc, s);
      }
      y = b.relu(acc);
    }
    return b.sum(y, {}, false);
  }
  if (p.task == "diamond") {
    // Each layer fans the state out into `width` branches and joins them:
    // s_{i+1} = sum_j relu(s_i * w_ij). Backward accumulates `width` paths.
    V s = b.input({p.n}, "x");
    for (int i = 0; i < p.depth; ++i) {
      V acc{};
      for (int j = 0; j < p.width; ++j) {
        V w = b.input({p.n}, "w");
        V t = b.mul(s, w);
        V r = b.relu(t);
        if (j == 0) acc = std::move(r);
        else acc = b.add(acc, r);
      }
      s = std::move(acc);
    }
    return b.sum(s, {}, false);
  }
  if (p.task == "nn") {
    // loss = sum(relu(X @ W + bias)), bias broadcast over the batch.
    V x = b.input({p.batch, p.in}, "x");
    V w = b.input({p.in, p.out}, "w");
    V bias = b.input({p.out}, "b");
    V h = b.matmul(x, w);
    V hb = b.add(h, bias);
    V r = b.relu(hb);
    return b.sum(r, {}, false);
  }
  if (p.task == "mlp") {
    // depth hidden layers: h = relu(h @ W_l + b_l); loss = sum(h * h)
    V h = b.input({p.batch, p.in}, "x");
    int64_t d_in = p.in;
    for (int l = 0; l <= p.depth; ++l) {
      const int64_t d_out = l == p.depth ? p.out : p.hidden;
      V w = b.input({d_in, d_out}, "w");
      V bias = b.input({d_out}, "b");
      V m = b.matmul(h, w);
      V mb = b.add(m, bias);
      h = b.relu(mb);
      d_in = d_out;
    }
    V sq = b.mul(h, h);
    return b.sum(sq, {}, false);
  }
  throw std::runtime_error("unknown task: " + p.task);
}

struct DagBackend {
  using Value = NodeId;
  GraphBuilder& b;
  DType dtype;
  Value input(Shape s, const char*) { return b.input(std::move(s), dtype); }
  Value add(Value& x, Value& y) { return b.add(x, y); }
  Value mul(Value& x, Value& y) { return b.mul(x, y); }
  Value matmul(Value& x, Value& y) { return b.matmul(x, y); }
  Value relu(Value& x) { return b.relu(x); }
  Value sum(Value& x, std::vector<int64_t> dims, bool keepdim) { return b.sum(x, std::move(dims), keepdim); }
};

struct TreeBackend {
  using Value = TreeValue;
  TreeBuilder& b;
  DType dtype;
  Value input(Shape s, const char*) { return b.input(std::move(s), dtype); }
  Value add(Value& x, Value& y) { return b.add(x, y); }
  Value mul(Value& x, Value& y) { return b.mul(x, y); }
  Value matmul(Value& x, Value& y) { return b.matmul(x, y); }
  Value relu(Value& x) { return b.relu(x); }
  Value sum(Value& x, std::vector<int64_t> dims, bool keepdim) { return b.sum(x, std::move(dims), keepdim); }
};

struct InputSpec {
  Shape shape;
  std::string role;  // "x" (activation), "w" (weight), "b" (bias)
};
// Inputs a task declares, in declaration order.
std::vector<InputSpec> task_input_specs(const TaskParams& p);

}  // namespace sad
