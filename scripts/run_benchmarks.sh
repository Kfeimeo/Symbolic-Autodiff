#!/usr/bin/env bash
# Builds the project, runs the unit tests, runs the benchmark suites and
# renders the plots. Results land in results/.
# Usage: scripts/run_benchmarks.sh [suite] [extra sad_bench args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE="${1:-all}"
shift || true
cd "$ROOT"
if [ ! -d third_party/libtorch ] && [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
  scripts/setup_libtorch.sh
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${CMAKE_PREFIX_PATH:+-DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH} > /dev/null
cmake --build build -j"$(nproc)"
./build/sad_tests
mkdir -p results
./build/sad_bench --suite "$SUITE" --csv "results/${SUITE}.csv" "$@" | tee "results/${SUITE}.log"
if python3 -c "import matplotlib, pandas" 2>/dev/null; then
  python3 scripts/plot_results.py "results/${SUITE}.csv" results
else
  echo "matplotlib/pandas not available: skipping plots"
fi
