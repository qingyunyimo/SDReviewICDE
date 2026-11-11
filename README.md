# SkipDisk — ICDE Review Code

SkipDisk is an out-of-core approximate nearest-neighbor search implementation.
This source-only package contains the search engine, index preparation tools,
and small synthetic correctness tests. No datasets, pretrained indexes,
experimental measurements, cloud credentials, or historical Git commits are included.

## Quick start

Use Linux on an x86-64 CPU with AVX2 or AVX-512, GCC 11 or newer, CMake 3.15
or newer, and Python 3.10 or newer. Build on the machine that will run the code
because compilation uses `-march=native`.

On Ubuntu, install dependencies:

```bash
sudo apt-get install build-essential cmake libaio-dev libgoogle-perftools-dev libeigen3-dev libopenblas-dev
python3 -m venv /tmp/sdreviewicde-venv
source /tmp/sdreviewicde-venv/bin/activate
python3 -m pip install -r requirements.txt
bash scripts/build.sh
bash scripts/smoke_test.sh
```

The default build directory is `/tmp/sdreviewicde-build`. Both scripts accept
an alternative build directory as their first argument. Set `JOBS=4` to change
build parallelism (default: two). Tests create small random vectors and toy
graphs in temporary directories, then remove them automatically. They check
serialization, exact top-k IDs/distances at exhaustive search breadth,
independent PQ navigation, conservative filtering, and partial filtering caches.
Some negative tests intentionally print errors for malformed input; success is
reported by the final test messages and a zero exit status.

The quick start uses Linux AIO to work on systems that restrict io_uring.
It is a correctness demonstration, not a reproduction of paper latency numbers.
The bundled liburing backend remains available for performance experiments.

## Source map

| Path | Purpose |
| --- | --- |
| `SkipDisk/src/` and `SkipDisk/include/` | Graph traversal, filtering, exact distance evaluation, and disk I/O |
| `SkipDisk/tests/search_pcamixed_disk_index.cpp` | Unified mixed-precision search CLI |
| `SkipDisk/myscript/build_mixed_pivot.py` | Build filtering codes and residual radii |
| `SkipDisk/myscript/allocate_mixed_precision.py` | Precision allocation under a memory budget |
| `SkipDisk/myscript/train_allocation_pca.py` | Sampled PCA and eigenvalue preparation |
| `SkipDisk/myscript/build_query_access_ranking.py` | Generate access rankings for partial caches |
| `DiskANN/` | Modified graph and PQ index preparation source |
| `scripts/` | Portable build and synthetic smoke-test entry points |

## Representations and variants

Traversal representations generate candidates; filtering representations bound
or estimate distances; full-precision vectors provide final distances. These
roles are independent. PQ1/PQ2/PQ4/PQ8 mean **1/2/4/8 dimensions per subspace**,
with 256 centers and one byte per subspace, not bits per dimension.

The unified search executable supports `--no-bound` (no filtering),
`--filter-cache IDS COUNT` (partial filtering cache), and
`--estimation LMD EPSILON` (calibrated estimation). Use its printed usage and
the included synthetic tests for complete invocation examples. Estimation is
distinct from conservative lower-bound filtering and requires calibration inputs.

## Using your own data

Python tools expose their options with `--help`:

```bash
python3 SkipDisk/myscript/train_allocation_pca.py --help
python3 SkipDisk/myscript/build_mixed_pivot.py --help
python3 SkipDisk/myscript/build_query_access_ranking.py --help
```

Vector `.bin` inputs contain two little-endian uint32 values (row count,
dimension), followed by a row-major float32 matrix. Large-data execution needs
matching graph, navigation PQ, PCA, filtering, and full-precision sector files;
the toy graph in the tests is only a correctness fixture. The modified DiskANN
tools are included for graph/PQ construction. Their build additionally requires
Boost program_options and Intel MKL; see `docs/BUILD.md`.

No dataset download or cloud execution is performed by these scripts. Paper
datasets and measurement artifacts are deliberately excluded from this package.

## Third-party code

This implementation builds on DiskANN and PipeANN and includes third-party
headers and liburing. Existing copyright notices and bundled licenses are
retained; see `THIRD_PARTY.md`. This packaging does not grant a new blanket
license over third-party code.
