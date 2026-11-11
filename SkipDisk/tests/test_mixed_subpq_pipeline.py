#!/usr/bin/env python3
"""Real SSD regression: independent subPQ navigation + mixed-pivot bounds."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
from test_mixed_pivot_pipeline import builder, bin_write, make_sectors


def write_subpq(prefix, base, dim, chunks):
    """Write standard DiskANN 4-offset PQ files, with nonzero centering."""
    data = base[:, :dim]
    mean = data.mean(axis=0).astype('<f4')
    centered = data - mean
    offsets = np.linspace(0, dim, chunks + 1, dtype='<u4')
    centers = centered[np.arange(256) % len(base)].copy()
    codes = np.empty((len(base), chunks), dtype='u1')
    for c, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        dist = ((centered[:, None, lo:hi] - centers[None, :, lo:hi]) ** 2).sum(axis=2)
        codes[:, c] = dist.argmin(axis=1)
    sections = []
    for arr in (centers, mean[:, None], offsets[:, None]):
        sections.append(struct.pack('<II', *arr.shape) + arr.tobytes())
    positions = [4096]
    for section in sections:
        positions.append(positions[-1] + len(section))
    header = struct.pack('<II4Q', 4, 1, *positions)
    Path(prefix + '_navi_pq_pivots.bin').write_bytes(header.ljust(4096, b'\0') + b''.join(sections))
    bin_write(prefix + '_navi_pq_compressed.bin', codes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-binary', required=True)
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env.pop('SKIPDISK_BENCH_LEGACY_BF16', None)
    env.pop('SKIPDISK_BENCH_WARMUP', None)
    with tempfile.TemporaryDirectory(prefix='skipdisk-subpq-test-') as tmp:
        work = Path(tmp)
        rng = np.random.default_rng(471)
        base = (rng.normal(size=(320, 13)) * np.geomspace(4, .2, 13)).astype('<f4')
        queries = rng.normal(size=(12, 13)).astype('<f4')
        bin_write(work / 'base.bin', base)
        bin_write(work / 'queries.bin', queries)
        make_sectors(work / 'graph.index', base, True)
        prefix = str(work / 'mixed')
        with threadpool_limits(limits=1):
            builder.build(SimpleNamespace(pca_base=str(work / 'base.bin'),
                graph_index=str(work / 'graph.index'), index_prefix=prefix,
                m=3, n=4, train_size=320, iterations=3, batch_size=73, threads=1, seed=42))
        make_sectors(Path(prefix + '_data_sector.bin'), base, False)
        exact = ((queries[:, None].astype('f8') - base.astype('f8')) ** 2).sum(axis=2)
        expected = np.argsort(exact, axis=1)[:, :10]
        for dim, chunks in ((7, 3), (13, 5), (1, 1)):
            pq = str(work / f'pq-{dim}-{chunks}')
            write_subpq(pq, base, dim, chunks)
            out = str(work / 'result')
            for l in (len(base), 30):
                subprocess.run([args.search_binary, prefix, str(work / 'queries.bin'), '-',
                    '10', str(l), '3', '2', '-', out, pq], check=True, timeout=90, env=env)
                ids = np.fromfile(out + '_ids.bin', dtype='<u4', offset=8).reshape(12, 10)
                distances = builder.read_bin(out + '_dists.bin')
                assert np.all(ids < len(base))
                assert all(len(set(row)) == 10 for row in ids)
                assert np.all(np.diff(distances, axis=1) >= 0)
                np.testing.assert_allclose(distances, np.take_along_axis(exact, ids, axis=1),
                                           rtol=2e-6, atol=1e-6)
                if l == len(base):
                    assert np.array_equal(ids, expected)
        # Wrong point count and truncated codes must be rejected, not searched.
        bad = str(work / 'bad')
        write_subpq(bad, base[:-1], 7, 3)
        command = [args.search_binary, prefix, str(work / 'queries.bin'), '-',
                   '10', '30', '3', '2', '-', '-', bad]
        assert subprocess.run(command, env=env, timeout=90).returncode != 0
        Path(bad + '_navi_pq_compressed.bin').write_bytes(struct.pack('<II', 320, 3))
        assert subprocess.run(command, env=env, timeout=90).returncode != 0
        for malformed in ('bounds', 'nan', 'offset'):
            write_subpq(bad, base, 7, 3)
            path = Path(bad + '_navi_pq_pivots.bin')
            payload = bytearray(path.read_bytes())
            offsets = struct.unpack_from('<4Q', payload, 8)
            if malformed == 'bounds':
                struct.pack_into('<I', payload, offsets[2] + 8 + 4, 1000)
            elif malformed == 'nan':
                struct.pack_into('<f', payload, offsets[0] + 8, float('nan'))
            else:
                struct.pack_into('<Q', payload, 8, len(payload) + 1)
            path.write_bytes(payload)
            assert subprocess.run(command, env=env, timeout=90).returncode != 0
    print('Mixed subPQ pipeline tests passed')


if __name__ == '__main__':
    main()
