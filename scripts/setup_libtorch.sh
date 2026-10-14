#!/usr/bin/env bash
# Downloads a CPU-only LibTorch build into third_party/libtorch.
# Usage: scripts/setup_libtorch.sh [version] [variant]
#   version: default 2.14.1; variant: cpu (default) or a CUDA tag such as cu126
set -euo pipefail
VERSION="${1:-2.14.1}"
VARIANT="${2:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"
if [ -d "$DEST/libtorch" ]; then
  echo "third_party/libtorch already exists ($(cat "$DEST/libtorch/build-version" 2>/dev/null || echo unknown))"
  exit 0
fi
URL="https://download.pytorch.org/libtorch/${VARIANT}/libtorch-shared-with-deps-${VERSION}%2B${VARIANT}.zip"
echo "downloading $URL"
curl -sS -L -o "$DEST/libtorch.zip" "$URL"
(cd "$DEST" && unzip -q libtorch.zip && rm libtorch.zip)
echo "installed LibTorch $(cat "$DEST/libtorch/build-version")"
