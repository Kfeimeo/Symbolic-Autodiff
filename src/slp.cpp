#include "sad/slp.hpp"

#include <sstream>

#include "sad/analysis.hpp"

namespace sad {

namespace {
bool is_view_op(OpCode op) {
  return op == OpCode::Transpose || op == OpCode::BroadcastTo || op == OpCode::Reshape;
}
}  // namespace

uint64_t SLP::total_defined_bytes() const {
  uint64_t bytes = 0;
  for (const auto& i : instrs) {
    const Node& n = graph->node(i.dst);
    if (is_view_op(n.op)) continue;
    bytes += static_cast<uint64_t>(numel(n.shape)) * dtype_size(n.dtype);
  }
  return bytes;
}

uint64_t SLP::static_peak_live_bytes() const {
  uint64_t live = 0, peak = 0;
  for (const auto& i : instrs) {
    const Node& n = graph->node(i.dst);
    if (!is_view_op(n.op)) live += static_cast<uint64_t>(numel(n.shape)) * dtype_size(n.dtype);
    peak = std::max(peak, live);
    for (auto r : i.release) {
      const Node& rn = graph->node(r);
      if (!is_view_op(rn.op)) live -= static_cast<uint64_t>(numel(rn.shape)) * dtype_size(rn.dtype);
    }
  }
  return peak;
}

SLP compile(const Graph& g, std::span<const NodeId> outputs, CompileOptions opts) {
  SLP p;
  p.graph = &g;
  p.outputs.assign(outputs.begin(), outputs.end());
  const std::vector<bool> live = reachable_from(g, outputs);
  p.keep.assign(g.size(), false);
  p.last_use.assign(g.size(), 0);
  for (auto o : outputs) p.keep[o] = true;
  for (auto in : g.inputs()) {
    if (live[in]) {
      p.keep[in] = true;
      p.inputs.push_back(in);
    }
  }

  // Pass 1: emit instructions in topological (ascending id) order and record
  // the last instruction index that uses every value.
  std::vector<uint32_t> instr_of(g.size(), 0);
  for (NodeId id = 0; id < g.size(); ++id) {
    if (!live[id] || g.node(id).op == OpCode::Input) continue;
    const uint32_t idx = static_cast<uint32_t>(p.instrs.size());
    instr_of[id] = idx;
    for (auto in : g.node(id).inputs) {
      p.last_use[in] = idx;
      ++p.edges;
    }
    p.instrs.push_back(Instr{id, {}});
  }

  // Pass 2: attach release lists. A value defined but never used (and not an
  // output) is released right after its definition.
  if (opts.release_dead_values) {
    std::vector<bool> used(g.size(), false);
    for (const auto& i : p.instrs) {
      for (auto in : g.node(i.dst).inputs) used[in] = true;
    }
    for (NodeId id = 0; id < g.size(); ++id) {
      if (!live[id] || p.keep[id] || g.node(id).op == OpCode::Input) continue;
      const uint32_t at = used[id] ? p.last_use[id] : instr_of[id];
      p.instrs[at].release.push_back(id);
    }
  }
  return p;
}

std::string dump(const SLP& p, size_t max_instrs) {
  std::ostringstream os;
  for (size_t k = 0; k < p.instrs.size() && k < max_instrs; ++k) {
    const Instr& i = p.instrs[k];
    const Node& n = p.graph->node(i.dst);
    os << k << ": %" << i.dst << " = " << op_name(n.op) << '(';
    for (size_t j = 0; j < n.inputs.size(); ++j) {
      if (j) os << ", ";
      os << '%' << n.inputs[j];
    }
    os << ") " << shape_str(n.shape);
    if (!i.release.empty()) {
      os << "  release:";
      for (auto r : i.release) os << " %" << r;
    }
    os << '\n';
  }
  return os.str();
}

}  // namespace sad
