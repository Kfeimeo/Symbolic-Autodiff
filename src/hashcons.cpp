#include "sad/hashcons.hpp"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace sad {

namespace {
inline uint64_t mix(uint64_t h, uint64_t v) {
  // 64-bit FNV-1a style mixing with a final avalanche step.
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return h;
}
}  // namespace

uint64_t structural_hash(const Node& n) {
  uint64_t h = 0xcbf29ce484222325ULL;
  h = mix(h, static_cast<uint64_t>(n.op));
  h = mix(h, static_cast<uint64_t>(n.dtype));
  h = mix(h, n.inputs.size());
  for (auto i : n.inputs) h = mix(h, i);
  h = mix(h, n.shape.size());
  for (auto d : n.shape) h = mix(h, static_cast<uint64_t>(d));
  h = mix(h, n.attrs.dims.size());
  for (auto d : n.attrs.dims) h = mix(h, static_cast<uint64_t>(d));
  h = mix(h, n.attrs.keepdim);
  uint64_t bits;
  std::memcpy(&bits, &n.attrs.value, sizeof(bits));
  h = mix(h, bits);
  h = mix(h, n.attrs.input_index);
  return h;
}

bool structurally_equal(const Node& a, const Node& b) {
  return a.op == b.op && a.dtype == b.dtype && a.inputs == b.inputs &&
         a.shape == b.shape && a.attrs == b.attrs;
}

// The table maps a structural hash to candidate node ids. Full structural
// equality is checked against the node pool, so the key is never stored twice.
struct Graph::HashConsTable {
  std::unordered_multimap<uint64_t, NodeId> buckets;
};

Graph::Graph(bool hash_consing)
    : table_(new HashConsTable()), hash_consing_(hash_consing) {
  nodes_.reserve(1024);
}

Graph::Graph(Graph&& o) noexcept
    : nodes_(std::move(o.nodes_)),
      inputs_(std::move(o.inputs_)),
      outputs_(std::move(o.outputs_)),
      table_(o.table_),
      hash_consing_(o.hash_consing_),
      input_counter_(o.input_counter_),
      stats_(o.stats_) {
  o.table_ = nullptr;
}

Graph& Graph::operator=(Graph&& o) noexcept {
  if (this != &o) {
    delete table_;
    nodes_ = std::move(o.nodes_);
    inputs_ = std::move(o.inputs_);
    outputs_ = std::move(o.outputs_);
    table_ = o.table_;
    hash_consing_ = o.hash_consing_;
    input_counter_ = o.input_counter_;
    stats_ = o.stats_;
    o.table_ = nullptr;
  }
  return *this;
}

Graph::~Graph() { delete table_; }

NodeId Graph::emplace(Node n) {
  ++stats_.create_requests;
  for (auto in : n.inputs) {
    if (in >= nodes_.size()) {
      throw std::logic_error("Graph::emplace: input refers to a later node");
    }
  }
  if (hash_consing_) {
    const uint64_t h = structural_hash(n);
    auto range = table_->buckets.equal_range(h);
    for (auto it = range.first; it != range.second; ++it) {
      if (structurally_equal(nodes_[it->second], n)) {
        ++stats_.hashcons_hits;
        return it->second;
      }
    }
    const NodeId id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(std::move(n));
    table_->buckets.emplace(h, id);
    return id;
  }
  const NodeId id = static_cast<NodeId>(nodes_.size());
  nodes_.push_back(std::move(n));
  return id;
}

}  // namespace sad
