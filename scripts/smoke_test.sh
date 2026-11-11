#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${1:-${TMPDIR:-/tmp}/sdreviewicde-build}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT/SkipDisk/tests/test_mixed_pivot_pipeline.py" \
  --codec-binary "$BUILD_DIR/tests/test_mixed_pivot" \
  --search-binary "$BUILD_DIR/tests/search_pcamixed_disk_index"
python3 "$ROOT/SkipDisk/tests/test_mixed_subpq_pipeline.py" \
  --search-binary "$BUILD_DIR/tests/search_pcamixed_disk_index"
python3 "$ROOT/SkipDisk/tests/test_mixed_filter_cache.py" \
  --search-binary "$BUILD_DIR/tests/search_pcamixed_disk_index"
