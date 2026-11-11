# Build details

The quick-start build targets the mixed search executable and codec verifier.
To compile all SkipDisk utilities after configuring:

```bash
cmake --build /tmp/sdreviewicde-build --parallel 2
```

For the io_uring backend, build the bundled library in a writable copy outside
the source package, then point CMake to its headers and library:

```bash
cp -a SkipDisk/third_party/liburing /tmp/sdreviewicde-liburing
(cd /tmp/sdreviewicde-liburing && ./configure && make -C src -j2)
cmake -S SkipDisk -B /tmp/sdreviewicde-uring-build -DUSE_AIO=OFF \
  -DCMAKE_CXX_FLAGS=-I/tmp/sdreviewicde-liburing/src/include \
  -DCMAKE_EXE_LINKER_FLAGS=-L/tmp/sdreviewicde-liburing/src
cmake --build /tmp/sdreviewicde-uring-build --parallel 2
```

For optional DiskANN graph/PQ preparation, install `libboost-program-options-dev`
and Intel MKL (Ubuntu package `libmkl-full-dev` where available), then run:

```bash
cmake -S DiskANN -B /tmp/sdreviewicde-diskann-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/sdreviewicde-diskann-build --target build_disk_index build_pq --parallel 2
/tmp/sdreviewicde-diskann-build/apps/build_disk_index --help
/tmp/sdreviewicde-diskann-build/apps/build_pq --help
```

The optional DiskANN tree is packaged for Linux command-line tools. Windows
build support, Python bindings, and upstream development workflows are excluded;
the retained REST source files are not built by default.
