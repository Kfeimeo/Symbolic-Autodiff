#include "sad/tasks.hpp"

#include <sstream>

namespace sad {

std::string TaskParams::label() const {
  std::ostringstream os;
  os << task;
  if (task == "chain") os << "(n=" << n << ",depth=" << depth << ")";
  else if (task == "sharing") os << "(n=" << n << ",depth=" << depth << ",k=" << sharing << ")";
  else if (task == "diamond") os << "(n=" << n << ",depth=" << depth << ",width=" << width << ")";
  else if (task == "nn") os << "(batch=" << batch << ",in=" << in << ",out=" << out << ")";
  else if (task == "mlp") os << "(batch=" << batch << ",in=" << in << ",hidden=" << hidden << ",out=" << out << ",layers=" << depth << ")";
  return os.str();
}

namespace {
struct ShapeRecorder {
  struct Value {
    Shape shape;
  };
  std::vector<InputSpec>& specs;
  Value input(Shape s, const char* role) {
    specs.push_back(InputSpec{s, role});
    return Value{std::move(s)};
  }
  Value add(Value& x, Value& y) { return Value{broadcast_shapes(x.shape, y.shape)}; }
  Value mul(Value& x, Value& y) { return Value{broadcast_shapes(x.shape, y.shape)}; }
  Value matmul(Value& x, Value& y) { return Value{{x.shape[0], y.shape[1]}}; }
  Value relu(Value& x) { return x; }
  Value sum(Value& x, std::vector<int64_t> dims, bool keepdim) {
    return Value{sum_shape(x.shape, dims, keepdim)};
  }
};
}  // namespace

std::vector<InputSpec> task_input_specs(const TaskParams& p) {
  std::vector<InputSpec> specs;
  ShapeRecorder rec{specs};
  build_task(rec, p);
  return specs;
}

}  // namespace sad
