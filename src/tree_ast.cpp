#include "sad/tree_ast.hpp"

#include "sad/type_inference.hpp"

namespace sad {

void TreeBuilder::count(uint64_t k) {
  stats_.nodes_created += k;
  if (stats_.cap && stats_.nodes_created > stats_.cap) {
    throw TreeCapExceeded("tree node cap exceeded (" + std::to_string(stats_.cap) + ")");
  }
}

TreePtr TreeBuilder::clone(const TreeNode& n) {
  count(1);
  ++stats_.copies;
  auto c = std::make_shared<TreeNode>();
  c->op = n.op;
  c->attrs = n.attrs;
  c->shape = n.shape;
  c->dtype = n.dtype;
  c->kids.reserve(n.kids.size());
  for (const auto& k : n.kids) c->kids.push_back(clone(*k));
  return c;
}

TreePtr TreeBuilder::take(TreeValue& v) {
  if (!v.node_) throw std::logic_error("TreeBuilder: empty value");
  if (v.consumed) return clone(*v.node_);
  v.consumed = true;
  return v.node_;
}

TreePtr TreeBuilder::make(OpCode op, std::vector<TreePtr> kids, Attrs attrs, Shape shape) {
  count(1);
  auto n = std::make_shared<TreeNode>();
  n->op = op;
  n->kids = std::move(kids);
  n->attrs = std::move(attrs);
  n->shape = std::move(shape);
  // Type inference: reuse the DAG rules through a tiny adapter graph-less path.
  switch (op) {
    case OpCode::Add:
    case OpCode::Mul:
      n->shape = broadcast_shapes(n->kids[0]->shape, n->kids[1]->shape);
      n->dtype = promote(n->kids[0]->dtype, n->kids[1]->dtype);
      break;
    case OpCode::GreaterThan:
      n->shape = broadcast_shapes(n->kids[0]->shape, n->kids[1]->shape);
      n->dtype = DType::Bool;
      break;
    case OpCode::MatMul:
      if (n->kids[0]->shape.size() != 2 || n->kids[1]->shape.size() != 2 ||
          n->kids[0]->shape[1] != n->kids[1]->shape[0]) {
        throw TypeError("tree MatMul: shape mismatch");
      }
      n->shape = {n->kids[0]->shape[0], n->kids[1]->shape[1]};
      n->dtype = n->kids[0]->dtype;
      break;
    case OpCode::ReLU:
    case OpCode::OnesLike:
      n->shape = n->kids[0]->shape;
      n->dtype = n->kids[0]->dtype;
      break;
    case OpCode::Sum:
      n->shape = sum_shape(n->kids[0]->shape, n->attrs.dims, n->attrs.keepdim);
      n->dtype = n->kids[0]->dtype;
      break;
    case OpCode::Transpose:
      n->shape = {n->kids[0]->shape[1], n->kids[0]->shape[0]};
      n->dtype = n->kids[0]->dtype;
      break;
    case OpCode::BroadcastTo:
    case OpCode::SumToShape:
    case OpCode::Reshape:
      n->dtype = n->kids[0]->dtype;  // shape is the attribute
      break;
    case OpCode::Input:
    case OpCode::Constant:
      break;
    default:
      throw std::logic_error("tree make: unsupported op");
  }
  return n;
}

TreeValue TreeBuilder::input(Shape shape, DType dtype) {
  count(1);
  auto n = std::make_shared<TreeNode>();
  n->op = OpCode::Input;
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->attrs.input_index = next_input_++;
  return TreeValue(std::move(n));
}

TreeValue TreeBuilder::constant(double value, Shape shape, DType dtype) {
  count(1);
  auto n = std::make_shared<TreeNode>();
  n->op = OpCode::Constant;
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->attrs.value = value;
  return TreeValue(std::move(n));
}

namespace {
TreeValue wrap(TreePtr p) { return TreeValue(std::move(p)); }
}  // namespace

TreeValue TreeBuilder::add(TreeValue& a, TreeValue& b) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  kids.push_back(take(b));
  return wrap(make(OpCode::Add, std::move(kids)));
}
TreeValue TreeBuilder::mul(TreeValue& a, TreeValue& b) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  kids.push_back(take(b));
  return wrap(make(OpCode::Mul, std::move(kids)));
}
TreeValue TreeBuilder::matmul(TreeValue& a, TreeValue& b) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  kids.push_back(take(b));
  return wrap(make(OpCode::MatMul, std::move(kids)));
}
TreeValue TreeBuilder::relu(TreeValue& a) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  return wrap(make(OpCode::ReLU, std::move(kids)));
}
TreeValue TreeBuilder::sum(TreeValue& a, std::vector<int64_t> dims, bool keepdim) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  Attrs at;
  at.dims = std::move(dims);
  at.keepdim = keepdim;
  return wrap(make(OpCode::Sum, std::move(kids), std::move(at)));
}
TreeValue TreeBuilder::transpose(TreeValue& a) {
  std::vector<TreePtr> kids;
  kids.push_back(take(a));
  return wrap(make(OpCode::Transpose, std::move(kids)));
}

uint64_t tree_size(const TreeNode& n) {
  uint64_t s = 1;
  for (const auto& k : n.kids) s += tree_size(*k);
  return s;
}

uint64_t tree_depth(const TreeNode& n) {
  uint64_t d = 0;
  for (const auto& k : n.kids) d = std::max(d, tree_depth(*k));
  return d + 1;
}

uint64_t tree_bytes(const TreeNode& n) {
  uint64_t b = sizeof(TreeNode) + n.kids.capacity() * sizeof(TreePtr) +
               n.shape.capacity() * sizeof(int64_t) + n.attrs.dims.capacity() * sizeof(int64_t);
  for (const auto& k : n.kids) b += tree_bytes(*k);
  return b;
}

namespace {

struct TreeBackward {
  TreeBuilder& b;
  TreeGradResult res;

  TreePtr sum_to(TreePtr g, const Shape& s) {
    if (g->shape == s) return g;
    std::vector<TreePtr> kids;
    kids.push_back(std::move(g));
    return b.make(OpCode::SumToShape, std::move(kids), {}, s);
  }
  TreePtr broadcast(TreePtr g, const Shape& s) {
    if (g->shape == s) return g;
    std::vector<TreePtr> kids;
    kids.push_back(std::move(g));
    return b.make(OpCode::BroadcastTo, std::move(kids), {}, s);
  }
  TreePtr reshape(TreePtr g, const Shape& s) {
    if (g->shape == s) return g;
    std::vector<TreePtr> kids;
    kids.push_back(std::move(g));
    return b.make(OpCode::Reshape, std::move(kids), {}, s);
  }
  TreePtr binary(OpCode op, TreePtr x, TreePtr y) {
    std::vector<TreePtr> kids;
    kids.push_back(std::move(x));
    kids.push_back(std::move(y));
    return b.make(op, std::move(kids));
  }
  TreePtr unary(OpCode op, TreePtr x) {
    std::vector<TreePtr> kids;
    kids.push_back(std::move(x));
    return b.make(op, std::move(kids));
  }

  void backprop(const TreeNode& n, TreePtr gy) {
    switch (n.op) {
      case OpCode::Input: {
        ++res.contributions;
        auto& slot = res.grads[n.attrs.input_index];
        slot = slot ? binary(OpCode::Add, std::move(slot), std::move(gy)) : std::move(gy);
        return;
      }
      case OpCode::Constant:
      case OpCode::OnesLike:
      case OpCode::GreaterThan:
        return;
      case OpCode::Add: {
        backprop(*n.kids[0], sum_to(b.clone(*gy), n.kids[0]->shape));
        backprop(*n.kids[1], sum_to(std::move(gy), n.kids[1]->shape));
        return;
      }
      case OpCode::Mul: {
        TreePtr ga = binary(OpCode::Mul, b.clone(*gy), b.clone(*n.kids[1]));
        TreePtr gb = binary(OpCode::Mul, std::move(gy), b.clone(*n.kids[0]));
        backprop(*n.kids[0], sum_to(std::move(ga), n.kids[0]->shape));
        backprop(*n.kids[1], sum_to(std::move(gb), n.kids[1]->shape));
        return;
      }
      case OpCode::MatMul: {
        TreePtr ga = binary(OpCode::MatMul, b.clone(*gy), unary(OpCode::Transpose, b.clone(*n.kids[1])));
        TreePtr gb = binary(OpCode::MatMul, unary(OpCode::Transpose, b.clone(*n.kids[0])), std::move(gy));
        backprop(*n.kids[0], std::move(ga));
        backprop(*n.kids[1], std::move(gb));
        return;
      }
      case OpCode::ReLU: {
        TreePtr zero = b.make(OpCode::Constant, {}, Attrs{{}, false, 0.0, 0}, {});
        zero->dtype = n.kids[0]->dtype;
        TreePtr mask = binary(OpCode::GreaterThan, b.clone(*n.kids[0]), std::move(zero));
        backprop(*n.kids[0], binary(OpCode::Mul, std::move(gy), std::move(mask)));
        return;
      }
      case OpCode::Sum: {
        const Shape& xs = n.kids[0]->shape;
        TreePtr gk = std::move(gy);
        if (!n.attrs.keepdim) gk = reshape(std::move(gk), keepdim_shape(xs, n.attrs.dims));
        backprop(*n.kids[0], broadcast(std::move(gk), xs));
        return;
      }
      case OpCode::Transpose:
        backprop(*n.kids[0], unary(OpCode::Transpose, std::move(gy)));
        return;
      case OpCode::BroadcastTo:
        backprop(*n.kids[0], sum_to(std::move(gy), n.kids[0]->shape));
        return;
      case OpCode::SumToShape:
        backprop(*n.kids[0], broadcast(std::move(gy), n.kids[0]->shape));
        return;
      case OpCode::Reshape:
        backprop(*n.kids[0], reshape(std::move(gy), n.kids[0]->shape));
        return;
      default:
        throw std::logic_error("tree_backward: unsupported op");
    }
  }
};

}  // namespace

TreeGradResult tree_backward(TreeBuilder& b, const TreeNode& loss, uint32_t num_inputs) {
  if (numel(loss.shape) != 1) throw std::logic_error("tree_backward: loss must be scalar");
  TreeBackward bw{b, {}};
  bw.res.grads.resize(num_inputs);
  std::vector<TreePtr> seed_kids;
  seed_kids.push_back(b.clone(loss));  // OnesLike(loss), as in the DAG version
  TreePtr seed = b.make(OpCode::OnesLike, std::move(seed_kids));
  bw.backprop(loss, std::move(seed));
  for (const auto& g : bw.res.grads) {
    if (g) bw.res.grad_nodes += tree_size(*g);
  }
  return bw.res;
}

}  // namespace sad
