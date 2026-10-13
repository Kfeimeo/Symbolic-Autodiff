#include "sad/analysis.hpp"

#include <sstream>

namespace sad {

std::vector<bool> reachable_from(const Graph& g, std::span<const NodeId> roots) {
  std::vector<bool> mark(g.size(), false);
  // Inputs always have smaller ids than their consumers, so a single
  // descending sweep from the largest root propagates reachability.
  NodeId hi = 0;
  for (auto r : roots) {
    mark[r] = true;
    hi = std::max(hi, r);
  }
  if (roots.empty()) return mark;
  for (NodeId id = hi + 1; id-- > 0;) {
    if (!mark[id]) continue;
    for (auto in : g.node(id).inputs) mark[in] = true;
  }
  return mark;
}

std::vector<NodeId> topo_order(const Graph& g, std::span<const NodeId> roots) {
  auto mark = reachable_from(g, roots);
  std::vector<NodeId> order;
  order.reserve(g.size());
  for (NodeId id = 0; id < g.size(); ++id) {
    if (mark[id]) order.push_back(id);
  }
  return order;
}

DefUse def_use(const Graph& g, const std::vector<bool>& live) {
  DefUse du;
  du.use_count.assign(g.size(), 0);
  du.users.resize(g.size());
  for (NodeId id = 0; id < g.size(); ++id) {
    if (!live.empty() && !live[id]) continue;
    for (auto in : g.node(id).inputs) {
      ++du.use_count[in];
      du.users[in].push_back(id);
      ++du.edges;
    }
  }
  return du;
}

OpHistogram op_histogram(const Graph& g, const std::vector<bool>& live) {
  OpHistogram h;
  h.count.assign(static_cast<size_t>(OpCode::kCount), 0);
  for (NodeId id = 0; id < g.size(); ++id) {
    if (!live.empty() && !live[id]) continue;
    ++h.count[static_cast<size_t>(g.node(id).op)];
    ++h.total;
  }
  return h;
}

std::string op_histogram_str(const OpHistogram& h) {
  std::ostringstream os;
  bool first = true;
  for (size_t i = 0; i < h.count.size(); ++i) {
    if (!h.count[i]) continue;
    if (!first) os << ' ';
    first = false;
    os << op_name(static_cast<OpCode>(i)) << '=' << h.count[i];
  }
  return os.str();
}

std::string dump(const Graph& g, size_t max_nodes) {
  std::ostringstream os;
  for (NodeId id = 0; id < g.size() && id < max_nodes; ++id) {
    const Node& n = g.node(id);
    os << '%' << id << " = " << op_name(n.op) << '(';
    for (size_t i = 0; i < n.inputs.size(); ++i) {
      if (i) os << ", ";
      os << '%' << n.inputs[i];
    }
    os << ") : " << dtype_name(n.dtype) << shape_str(n.shape);
    if (n.op == OpCode::Constant) os << " value=" << n.attrs.value;
    if (n.op == OpCode::Input) os << " input#" << n.attrs.input_index;
    if (n.op == OpCode::Sum) {
      os << " dims=" << shape_str(n.attrs.dims) << (n.attrs.keepdim ? " keepdim" : "");
    }
    os << '\n';
  }
  if (g.size() > max_nodes) os << "... (" << g.size() - max_nodes << " more)\n";
  return os.str();
}

}  // namespace sad
