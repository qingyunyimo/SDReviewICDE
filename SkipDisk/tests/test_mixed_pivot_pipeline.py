#!/usr/bin/env python3
"""Builder/serialization/C++ ADC tests; optionally exercise real SSD search."""
import argparse
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mixed_builder", ROOT / "myscript/build_mixed_pivot.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def bin_write(path, array):
    with open(path, "wb") as dst:
        dst.write(struct.pack("<II", *array.shape))
        dst.write(array.tobytes())


def make_sectors(path, base, with_graph):
    count, dim = base.shape
    degree = 2 if with_graph else 0
    record = dim * 4 + 4 * (degree + 1)
    per_sector = 4096 // record
    sector_size = 4096 if per_sector else ((record + 4095) // 4096) * 4096
    slots = per_sector or 1
    size = 4096 + ((count + slots - 1) // slots) * sector_size
    data = bytearray(size)
    if with_graph:
        struct.pack_into("<II9Q", data, 0, 9, 1, count, dim, 0, record, per_sector, 0, 0, 0, size)
    else:
        struct.pack_into("<II6Q", data, 0, 6, 1, count, dim, 0, record, per_sector, size)
    for idx in range(count):
        offset = 4096 + (idx // slots) * sector_size + (idx % slots) * record
        data[offset:offset + dim * 4] = base[idx].tobytes()
        if with_graph:
            struct.pack_into("<III", data, offset + dim * 4, 2, (idx + 1) % count, (idx - 1) % count)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-binary", required=True)
    parser.add_argument("--search-binary")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="skipdisk-mixed-test-") as folder:
        work = Path(folder)
        rng = np.random.default_rng(51)
        base = (rng.normal(size=(400, 13)) * np.geomspace(8, 0.1, 13)).astype("<f4")
        bin_write(work / "base.bin", base)
        graph = work / "graph.index"
        make_sectors(graph, base, True)
        for dim, m, n in [(13, 3, 4), (13, 0, 0), (13, 13, 0), (13, 0, 13), (13, 2, 4),
                          (1537, 128, 256)]:
            if dim != base.shape[1]:
                base = rng.normal(size=(20, dim)).astype("<f4")
                bin_write(work / "base.bin", base)
                make_sectors(graph, base, True)
            prefix = str(work / f"mixed-{m}-{n}")
            config = SimpleNamespace(pca_base=str(work / "base.bin"), graph_index=str(graph),
                                     index_prefix=prefix, m=m, n=n, train_size=400, iterations=5,
                                     batch_size=73, threads=1, seed=42)
            builder.build(config)
            mixed = prefix + "_navi_mixed_graph.index"
            expected_bytes = 2 * m + n + (dim - m - n + 1) // 2
            row_bytes = ((expected_bytes + 3) & ~3) + 8 + 2 * 4
            assert Path(mixed).stat().st_size == 36 + (dim - m) * 256 * 4 + len(base) * row_bytes
            subprocess.run([args.codec_binary, mixed, str(work / "base.bin")], check=True)
            try:
                builder.build(config)
                raise AssertionError("Existing index overwritten")
            except FileExistsError:
                pass
            if args.search_binary:
                make_sectors(Path(prefix + "_data_sector.bin"), base, False)
                queries = np.vstack([base[:3], rng.normal(size=(9, dim)).astype("<f4")])
                bin_write(work / "queries.bin", queries)
                pca_arg = "-"
                if (dim, m, n) == (13, 3, 4):
                    # Exercise the raw-query PCA entry point with a nontrivial
                    # orthogonal transform, independently of the codec tests.
                    p = np.linalg.qr(rng.normal(size=(dim, dim)))[0].astype("<f4")
                    mean = rng.normal(size=(1, dim)).astype("<f4")
                    raw_queries = (queries @ p.T + mean).astype("<f4")
                    bin_write(work / "queries.bin", raw_queries)
                    bin_write(work / "3.bin", p)
                    bin_write(work / "pcamean.bin", mean)
                    queries = (raw_queries - mean) @ p
                    pca_arg = str(work)
                result = str(work / "result")
                subprocess.run([args.search_binary, prefix, str(work / "queries.bin"), pca_arg,
                                "10", str(len(base)), "3", "2", "-", result], check=True, timeout=90)
                with open(result + "_ids.bin", "rb") as src:
                    assert struct.unpack("<II", src.read(8)) == (len(queries), 10)
                    ids = np.frombuffer(src.read(), dtype="<u4").reshape(len(queries), 10)
                distances = builder.read_bin(result + "_dists.bin")
                exact = ((queries[:, None, :].astype(np.float64) - base.astype(np.float64)) ** 2).sum(axis=2)
                expected = np.argsort(exact, axis=1)[:, :10]
                assert np.array_equal(ids, expected), (m, n, ids, expected)
                np.testing.assert_allclose(distances, np.take_along_axis(exact, expected, axis=1), rtol=2e-6, atol=1e-6)

        # Fewer than 256 training vectors, constant dimensions, empty BF16 head.
        small = np.zeros((7, 5), dtype="<f4")
        small[:, 0] = np.arange(7)
        books = builder.train(small, 0, 2, 7, 2, 42)
        head, codes, radius = builder.encode(small, 0, builder.subspaces(5, 0, 2), books)
        assert head.shape == (7, 0) and codes.shape == (7, 4)
        assert np.all(radius == 0)
        for m, n in [(-1, 0), (0, -1), (5, 1)]:
            try:
                builder.subspaces(5, m, n)
                raise AssertionError("Invalid partition accepted")
            except ValueError:
                pass
    print("Mixed pivot pipeline tests passed")


if __name__ == "__main__":
    main()
