// Structural hash-consing of DAG nodes.
//
// Two nodes share a NodeId iff OpCode, input NodeId sequence, attributes,
// shape and dtype are identical. No algebraic rewriting is done:
// Add(a,b) and Add(b,a) are different nodes.
#pragma once

#include <cstdint>

#include "sad/ir.hpp"

namespace sad {

uint64_t structural_hash(const Node& n);
bool structurally_equal(const Node& a, const Node& b);

}  // namespace sad
