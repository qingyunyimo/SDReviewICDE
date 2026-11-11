# Package validation

Validated with GCC 15.2, CMake 4.2.3, Python 3.12.7, NumPy 1.26.4,
SciPy 1.13.1, scikit-learn 1.5.1, and threadpoolctl 3.5.0 on Linux x86-64.

Completed checks:

- SkipDisk mixed-search and codec executables compiled using OpenBLAS and AIO.
- All three scripts in `scripts/smoke_test.sh` completed successfully.
- The tests checked codec serialization, exhaustive top-k IDs/distances,
  independent navigation PQ, malformed input rejection, and partial filtering
  caches with conservative bounds and estimation, including multi-threaded I/O.
- DiskANN `build_disk_index` and `build_pq` compiled; `build_disk_index --help`
  completed successfully. Large-data graph construction was not rerun.
- Shell syntax and included Python syntax checked.
- Source package scanned for binary payloads, common credential patterns,
  private storage-bucket names, and local workstation paths.

No paper benchmark was rerun, and no benchmark measurements or generated test
inputs are included. The optional io_uring build instructions were not executed
as part of this package validation.
