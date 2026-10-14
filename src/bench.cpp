#include "sad/bench.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace sad {

Summary summarize(std::vector<double> s) {
  Summary r;
  if (s.empty()) return r;
  std::sort(s.begin(), s.end());
  r.n = s.size();
  r.min = s.front();
  r.max = s.back();
  r.mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
  const auto pick = [&](double q) {
    const double pos = q * static_cast<double>(s.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, s.size() - 1);
    return s[lo] + (s[hi] - s[lo]) * (pos - static_cast<double>(lo));
  };
  r.median = pick(0.5);
  r.p90 = pick(0.9);
  return r;
}

void Row::set(const std::string& key, const std::string& value) {
  for (auto& kv : items_) {
    if (kv.first == key) {
      kv.second = value;
      return;
    }
  }
  items_.emplace_back(key, value);
}

void Row::set(const std::string& key, double value) {
  std::ostringstream os;
  os.precision(6);
  os << value;
  set(key, os.str());
}

void Row::set(const std::string& key, int64_t value) { set(key, std::to_string(value)); }

const std::string* Row::get(const std::string& key) const {
  for (const auto& kv : items_) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

void CsvTable::add(const Row& r) {
  for (const auto& kv : r.items()) {
    if (std::find(columns_.begin(), columns_.end(), kv.first) == columns_.end()) {
      columns_.push_back(kv.first);
    }
  }
  rows_.push_back(r);
}

namespace {
std::string csv_escape(const std::string& v) {
  if (v.find_first_of(",\"\n") == std::string::npos) return v;
  std::string out = "\"";
  for (char c : v) {
    if (c == '"') out += '"';
    out += c;
  }
  out += '"';
  return out;
}
}  // namespace

std::string CsvTable::to_string() const {
  std::ostringstream os;
  for (size_t i = 0; i < columns_.size(); ++i) {
    if (i) os << ',';
    os << columns_[i];
  }
  os << '\n';
  for (const auto& r : rows_) {
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) os << ',';
      if (const auto* v = r.get(columns_[i])) os << csv_escape(*v);
    }
    os << '\n';
  }
  return os.str();
}

void CsvTable::write(const std::string& path) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << to_string();
}

std::string format_bytes(double b) {
  const char* units[] = {"B", "KiB", "MiB", "GiB"};
  int u = 0;
  while (b >= 1024 && u < 3) {
    b /= 1024;
    ++u;
  }
  std::ostringstream os;
  os.precision(u == 0 ? 0 : 1);
  os << std::fixed << b << units[u];
  return os.str();
}

std::string format_us(double us) {
  std::ostringstream os;
  os << std::fixed;
  if (us < 1000) {
    os.precision(1);
    os << us << "us";
  } else if (us < 1e6) {
    os.precision(2);
    os << us / 1000 << "ms";
  } else {
    os.precision(3);
    os << us / 1e6 << "s";
  }
  return os.str();
}

}  // namespace sad
