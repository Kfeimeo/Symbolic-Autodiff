// Straight-Line Program: a topologically ordered SSA-style instruction list
// with liveness information (last use) for every value.
#pragma once

#include <span>
#include <vector>

#include "sad/ir.hpp"

namespace sad {

struct Instr {
  NodeId dst;                   // value defined by this instruction
  std::vector<NodeId> release;  // values whose last use is this instruction
};

struct SLP {
  const Graph* graph = nullptr;
  std::vector<Instr> instrs;    // executable (non-Input) nodes in topo order
  std::vector<NodeId> inputs;   // graph inputs the program reads
  std::vector<NodeId> outputs;  // values that must stay alive at the end
  std::vector<bool> keep;       // keep[id]: never released by the program
  std::vector<uint32_t> last_use;  // instruction index of last use (per node)
  uint64_t edges = 0;           // operand references among executed instrs

  size_t num_ops() const { return instrs.size(); }
  // Logical bytes of all values defined by the program (no liveness).
  uint64_t total_defined_bytes() const;
  // Peak logical live bytes predicted statically from shapes and liveness
  // (views such as Transpose/BroadcastTo/Reshape are counted as 0 bytes).
  uint64_t static_peak_live_bytes() const;
};

struct CompileOptions {
  bool release_dead_values = true;   // false => keep everything until the end
};

// Compiles the sub-DAG needed for `outputs` into an SLP. Inputs are always
// kept (they belong to the caller); `outputs` are kept; everything else is
// released right after its last use.
SLP compile(const Graph& g, std::span<const NodeId> outputs, CompileOptions opts = {});

std::string dump(const SLP& p, size_t max_instrs = 200);

}  // namespace sad
