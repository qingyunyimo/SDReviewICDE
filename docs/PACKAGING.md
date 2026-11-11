# Packaging scope

This package is a source snapshot of the working implementation, including
current uncommitted implementation changes. It starts a new Git history.
Datasets, measured results, indexes, figures, cloud task controllers, local
editor configuration, and machine credentials are excluded.

Packaging changes are limited to documentation, build/test entry points,
removal of commented local paths, a newer CMake minimum, removal of a
machine-specific CMake path, an ignored AIO context-flag declaration, and an
explicit standard integer header for newer C++ compilers. The AIO vector-submit
path also fixes `cb.data` to `cb[j].data`, so user data is assigned to each I/O
control block rather than to the vector object. Missing AIO completion-wait
and synchronous-context cleanup methods are supplied, with context reset on
cleanup. These changes enable the documented AIO smoke-test path.

The quick-start build uses OpenBLAS and Linux AIO. The paper's performance
experiments used their recorded hardware and I/O configuration; quick-start
timings should not be compared with those measurements. Synthetic tests are
correctness checks only, and generate their inputs outside this repository.
