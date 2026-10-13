// Minimal test framework: TEST(name) { REQUIRE(...); } registered at static init.
#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct Failure : std::exception {
  std::string msg;
  explicit Failure(std::string m) : msg(std::move(m)) {}
  const char* what() const noexcept override { return msg.c_str(); }
};

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline int run_all(int argc, char** argv) {
  std::string filter = argc > 1 ? argv[1] : "";
  int failed = 0, ran = 0;
  for (const auto& c : registry()) {
    if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) continue;
    ++ran;
    try {
      c.fn();
      std::printf("[ OK ] %s\n", c.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[FAIL] %s: %s\n", c.name, e.what());
    }
  }
  std::printf("%d tests, %d failed\n", ran, failed);
  return failed ? 1 : 0;
}

}  // namespace testing

#define TEST(name)                                                   \
  static void test_##name();                                         \
  static testing::Registrar reg_##name(#name, &test_##name);         \
  static void test_##name()

#define REQUIRE(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::ostringstream os_;                                                    \
      os_ << __FILE__ << ':' << __LINE__ << " REQUIRE failed: " #cond;             \
      throw testing::Failure(os_.str());                                         \
    }                                                                            \
  } while (0)

#define REQUIRE_EQ(a, b)                                                                    \
  do {                                                                                    \
    auto va_ = (a);                                                                       \
    auto vb_ = (b);                                                                       \
    if (!(va_ == vb_)) {                                                                  \
      std::ostringstream os_;                                                             \
      os_ << __FILE__ << ':' << __LINE__ << " REQUIRE_EQ failed: " #a " == " #b " (" << va_ \
          << " vs " << vb_ << ")";                                                        \
      throw testing::Failure(os_.str());                                                  \
    }                                                                                     \
  } while (0)

#define REQUIRE_THROWS(expr, Ex)                                                           \
  do {                                                                                   \
    bool thrown_ = false;                                                                \
    try {                                                                                \
      (void)(expr);                                                                      \
    } catch (const Ex&) {                                                                \
      thrown_ = true;                                                                    \
    }                                                                                    \
    if (!thrown_) {                                                                      \
      std::ostringstream os_;                                                            \
      os_ << __FILE__ << ':' << __LINE__ << " expected " #Ex " from " #expr;             \
      throw testing::Failure(os_.str());                                                 \
    }                                                                                    \
  } while (0)
