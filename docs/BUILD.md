# Build details

The quick-start build targets the mixed search executable and codec verifier.
To compile all SkipDisk utilities after configuring:

```bash
cmake --build /tmp/sdreviewicde-build --parallel 2
```

For the io_uring backend, install your distribution's `liburing-dev` package
(for example, `sudo apt-get install liburing-dev` on Ubuntu), then configure:

```bash
cmake -S SkipDisk -B /tmp/sdreviewicde-uring-build -DUSE_AIO=OFF \
  -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS
cmake --build /tmp/sdreviewicde-uring-build --parallel 2
```

CMake uses the installed headers and library. It does not automatically include
the bundled headers: mixing these with an older system library/Linux UAPI can
fail with `BLOCK_URING_CMD_DISCARD` undeclared. Use a fresh build directory when
changing the dependency. AIO builds do not require liburing.

To use a separately built copy instead, explicitly select both its headers and
library:

```bash
cp -a SkipDisk/third_party/liburing /tmp/sdreviewicde-liburing
(cd /tmp/sdreviewicde-liburing && ./configure && make -C src -j2)
cmake -S SkipDisk -B /tmp/sdreviewicde-uring-build -DUSE_AIO=OFF \
  -DLIBURING_INCLUDE_DIR=/tmp/sdreviewicde-liburing/src/include \
  -DLIBURING_LIBRARY=/tmp/sdreviewicde-liburing/src/liburing.a
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
