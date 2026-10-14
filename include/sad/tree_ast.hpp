// Baseline A: naive tree (AST) symbolic differentiation without sharing.
//
// Expressions are trees: every node has exactly one parent. Reusing a value
// as an operand a second time deep-copies it. Differentiation applies the
// classic reverse-mode rules but materialises every adjoint as a fresh tree,
// copying the sibling subtrees it refers to. A global node cap aborts the
// construction when the expression swells beyond the limit.
#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "sad/ir.hpp"

namespace sad {

struct TreeNode;
using TreePtr = std::shared_ptr<TreeNode>;

struct TreeNode {
  OpCode op = OpCode::Input;
  std::vector<TreePtr> kids;
  Attrs attrs;
  Shape shape;
  DType dtype = DType::Float32;
};

struct TreeCapExceeded : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// A value handle. The first operator that consumes the handle takes the tree
// as its child; every later use deep-copies it, so each tree node has exactly
// one parent and no subexpression is ever shared.
class TreeValue {
 public:
  TreeValue() = default;
  explicit TreeValue(TreePtr n) : node_(std::move(n)) {}
  const TreeNode& node() const { return *node_; }
  const Shape& shape() const { return node_->shape; }
  DType dtype() const { return node_->dtype; }
  TreePtr node_;
  bool consumed = false;
};

struct TreeStats {
  uint64_t nodes_created = 0;  // every node ever allocated (copies included)
  uint64_t copies = 0;         // nodes allocated by deep copies
  uint64_t cap = 0;
};

class TreeBuilder {
 public:
  explicit TreeBuilder(uint64_t node_cap = 0) { stats_.cap = node_cap; }

  TreeValue input(Shape shape, DType dtype = DType::Float32);
  TreeValue constant(double value, Shape shape = {}, DType dtype = DType::Float32);
  TreeValue add(TreeValue& a, TreeValue& b);
  TreeValue mul(TreeValue& a, TreeValue& b);
  TreeValue matmul(TreeValue& a, TreeValue& b);
  TreeValue relu(TreeValue& a);
  TreeValue sum(TreeValue& a, std::vector<int64_t> dims = {}, bool keepdim = false);
  TreeValue transpose(TreeValue& a);

  // Owned-tree primitives used by the differentiator.
  TreePtr clone(const TreeNode& n);
  TreePtr make(OpCode op, std::vector<TreePtr> kids, Attrs attrs = {}, Shape shape = {});
  TreePtr take(TreeValue& v);  // first use: the tree itself; later uses: a deep copy

  const TreeStats& stats() const { return stats_; }
  uint32_t num_inputs() const { return next_input_; }

 private:
  void count(uint64_t k);
  TreeStats stats_;
  uint32_t next_input_ = 0;
};

uint64_t tree_size(const TreeNode& n);
uint64_t tree_depth(const TreeNode& n);
uint64_t tree_bytes(const TreeNode& n);  // approximate heap bytes of the tree

struct TreeGradResult {
  std::vector<TreePtr> grads;  // per input index; null => zero gradient
  uint64_t grad_nodes = 0;     // total nodes in all gradient trees
  uint64_t contributions = 0;  // leaf adjoints produced
};

// Reverse-mode symbolic differentiation of a scalar loss tree w.r.t. every
// input. Every adjoint is a fresh tree; subtrees are copied, never shared.
TreeGradResult tree_backward(TreeBuilder& b, const TreeNode& loss, uint32_t num_inputs);

}  // namespace sad
