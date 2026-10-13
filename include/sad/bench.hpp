// Timing, summary statistics and CSV output for the benchmarks.
#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace sad {

using Clock = std::chrono::steady_clock;

class Timer {
 public:
  Timer() : t0_(Clock::now()) {}
  void reset() { t0_ = Clock::now(); }
  double us() const {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0_).count();
  }

 private:
  Clock::time_point t0_;
};

struct Summary {
  double median = 0, p90 = 0, min = 0, max = 0, mean = 0;
  size_t n = 0;
};
Summary summarize(std::vector<double> samples);

// One CSV row: ordered key/value pairs. Columns are the union over all rows in
// insertion order.
class Row {
 public:
  void set(const std::string& key, const std::string& value);
  void set(const std::string& key, double value);
  void set(const std::string& key, int64_t value);
  void set(const std::string& key, uint64_t value) { set(key, static_cast<int64_t>(value)); }
  void set(const std::string& key, int value) { set(key, static_cast<int64_t>(value)); }
  void set(const std::string& key, bool value) { set(key, static_cast<int64_t>(value)); }
  const std::string* get(const std::string& key) const;
  const std::vector<std::pair<std::string, std::string>>& items() const { return items_; }

 private:
  std::vector<std::pair<std::string, std::string>> items_;
};

class CsvTable {
 public:
  void add(const Row& r);
  void write(const std::string& path) const;
  std::string to_string() const;
  size_t size() const { return rows_.size(); }

 private:
  std::vector<std::string> columns_;
  std::vector<Row> rows_;
};

std::string format_bytes(double bytes);
std::string format_us(double us);

}  // namespace sad
