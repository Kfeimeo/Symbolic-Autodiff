// Tensor DAG intermediate representation.
//
// The DAG is immutable, side-effect free and statically shaped. Every node is
// a *symbolic* tensor value (no data). Nodes live in a std::vector<Node> and
// are identified by their dense index (NodeId). Inputs of a node always refer
// to nodes created earlier, so the node vector is already a topological order.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sad {

using NodeId = uint32_t;
constexpr NodeId kInvalidNode = static_cast<NodeId>(-1);

using Shape = std::vector<int64_t>;

enum class DType : uint8_t { Bool = 0, Float32 = 1, Float64 = 2 };

enum class OpCode : uint8_t {
  Input,        // leaf, identified by attrs.input_index
  Constant,     // scalar fill value broadcast to a static shape (attrs.value)
  Add,          // broadcasting elementwise add
  Mul,          // broadcasting elementwise mul
  MatMul,       // 2-D matrix multiply [m,k]x[k,n]
  ReLU,         // max(x,0), gradient at 0 defined as 0
  Sum,          // reduce over attrs.dims (empty = all), attrs.keepdim
  Transpose,    // 2-D transpose
  BroadcastTo,  // expand to the node's shape (NumPy rules)
  SumToShape,   // inverse of broadcasting: reduce to the node's shape
  Reshape,      // reshape to the node's shape (same numel)
  GreaterThan,  // broadcasting a > b, Bool result
  OnesLike,     // tensor of ones with the input's shape/dtype
  kCount
};

const char* op_name(OpCode op);
const char* dtype_name(DType dt);
size_t dtype_size(DType dt);
int64_t numel(const Shape& s);
std::string shape_str(const Shape& s);

// Operator attributes. Kept as a plain struct so that nodes are cheap to copy
// and compare. Only the fields relevant to the op are meaningful; the rest are
// kept at their defaults so that structural comparison stays well defined.
struct Attrs {
  std::vector<int64_t> dims;  // Sum: reduction dims (empty => all)
  bool keepdim = false;       // Sum
  double value = 0.0;         // Constant: fill value
  uint32_t input_index = 0;   // Input: unique identity of the input
  bool operator==(const Attrs& o) const;
};

struct Node {
  OpCode op = OpCode::Input;
  std::vector<NodeId> inputs;
  Attrs attrs;
  Shape shape;
  DType dtype = DType::Float32;
};

// Statistics that the graph collects while nodes are created.
struct GraphStats {
  uint64_t create_requests = 0;  // emplace() calls
  uint64_t hashcons_hits = 0;    // requests answered by an existing node
};

class Graph {
 public:
  explicit Graph(bool hash_consing = true);
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) noexcept;
  Graph& operator=(Graph&&) noexcept;
  ~Graph();

  // Creates a node (or returns the id of a structurally identical one when
  // hash-consing is enabled). Shape/dtype of `n` must already be inferred.
  NodeId emplace(Node n);

  const Node& node(NodeId id) const { return nodes_[id]; }
  size_t size() const { return nodes_.size(); }
  const std::vector<Node>& nodes() const { return nodes_; }

  const std::vector<NodeId>& inputs() const { return inputs_; }
  const std::vector<NodeId>& outputs() const { return outputs_; }
  void add_output(NodeId id) { outputs_.push_back(id); }
  void set_outputs(std::vector<NodeId> o) { outputs_ = std::move(o); }
  void register_input(NodeId id) { inputs_.push_back(id); }
  uint32_t next_input_index() { return input_counter_++; }

  bool hash_consing() const { return hash_consing_; }
  const GraphStats& stats() const { return stats_; }

  // Approximate bytes used by node storage (vectors included).
  size_t node_bytes() const;

 private:
  struct HashConsTable;  // defined in hashcons.cpp
  std::vector<Node> nodes_;
  std::vector<NodeId> inputs_;
  std::vector<NodeId> outputs_;
  HashConsTable* table_ = nullptr;
  bool hash_consing_ = true;
  uint32_t input_counter_ = 0;
  GraphStats stats_;
};

}  // namespace sad
