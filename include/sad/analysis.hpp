// DAG traversal and def-use analysis.
#pragma once

#include <span>
#include <vector>

#include "sad/ir.hpp"

namespace sad {

// Nodes reachable from `roots` by following inputs (roots included).
std::vector<bool> reachable_from(const Graph& g, std::span<const NodeId> roots);

// Topological order (ascending NodeId) restricted to the reachable set.
std::vector<NodeId> topo_order(const Graph& g, std::span<const NodeId> roots);

struct DefUse {
  std::vector<uint32_t> use_count;            // number of operand slots using node i
  std::vector<std::vector<NodeId>> users;     // consumers of node i (with repetition)
  uint64_t edges = 0;                         // total operand references
};

// Def-use information restricted to the nodes marked in `live` (all nodes when
// `live` is empty).
DefUse def_use(const Graph& g, const std::vector<bool>& live = {});

struct OpHistogram {
  std::vector<uint64_t> count;  // indexed by OpCode
  uint64_t total = 0;
};
OpHistogram op_histogram(const Graph& g, const std::vector<bool>& live = {});
std::string op_histogram_str(const OpHistogram& h);

// Human readable dump of a graph (debug aid).
std::string dump(const Graph& g, size_t max_nodes = 200);

}  // namespace sad
