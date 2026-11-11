#!/usr/bin/env python3
"""Rank cache points by candidate frequency from sampled indexed base vectors.

Uses existing subPQ graph traversal, not ground truth or exact reranking.
The base must have the same row IDs as the index. By default it is already in
the index's PCA coordinates; --pca-dir transforms sampled original-space rows.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile

import numpy as np


def matrix_shape(path, itemsize):
    with Path(path).open('rb') as src:
        header = src.read(8)
    if len(header) != 8:
        raise ValueError(f'Truncated matrix header: {path}')
    n, d = struct.unpack('<II', header)
    if not n or not d or Path(path).stat().st_size != 8 + n*d*itemsize:
        raise ValueError(f'Invalid matrix shape/size: {path}')
    return n, d


def index_shape(prefix):
    with Path(str(prefix) + '_navi_mixed_graph.index').open('rb') as src:
        header = src.read(48)
    if len(header) < 36 or header[:8] != b'SKMIXPQ1':
        raise ValueError('Expected an existing mixed-pivot graph index')
    version, n, dim = struct.unpack_from('<III', header, 8)
    if version not in (1, 2, 3) or len(header) < {1: 36, 2: 40, 3: 48}[version]:
        raise ValueError('Unsupported or truncated mixed-pivot header')
    return n, dim if version == 1 else struct.unpack_from('<I', header, 36)[0]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def build(args):
    base_path = args.base.resolve()
    prefix = args.index_prefix.resolve()
    subpq = args.subpq_prefix.resolve()
    binary = args.search_binary.resolve()
    n, d = matrix_shape(base_path, 4)
    if (n, d) != index_shape(prefix) or matrix_shape(str(subpq) + '_navi_pq_compressed.bin', 1)[0] != n:
        raise ValueError('Base shape/IDs must match the mixed index and subPQ point count')
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('Search binary does not exist or is not executable')
    if not 1 <= args.L <= 2**32-2 or not 1 <= args.threads <= 1024:
        raise ValueError('Require 1 <= L <= 2^32-2 and 1 <= threads <= 1024')
    if args.sample_ratio is not None:
        if not math.isfinite(args.sample_ratio) or not 0 < args.sample_ratio <= 1:
            raise ValueError('sample-ratio must be in (0, 1]')
        count = math.ceil(n*args.sample_ratio)
    else:
        count = args.sample_size
    if not 1 <= count <= n:
        raise ValueError(f'sample-size must be in [1, {n}]')
    pca_dir = '-' if args.pca_dir == '-' else str(Path(args.pca_dir).resolve())
    if pca_dir != '-':
        if matrix_shape(Path(pca_dir)/'3.bin', 4) != (d, d) or matrix_shape(Path(pca_dir)/'pcamean.bin', 4) != (1, d):
            raise ValueError('PCA projection/mean must match the indexed dimension')

    # Sampling without replacement, independent of held-out evaluation queries.
    # Sorting selected IDs only changes query order and makes base reads local.
    sampled = np.asarray(random.Random(args.seed).sample(range(n), count), dtype='<u4')
    sampled.sort()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior ranking.
    sample_path = out/'sampled_base_ids.id'
    sample_path.write_bytes(struct.pack('<I', count) + sampled.tobytes())
    metadata = {
        'status': 'running', 'index_prefix': str(prefix), 'subpq_prefix': str(subpq),
        'base': str(base_path), 'base_shape': [n, d], 'pca_dir': pca_dir,
        'sample_size': count, 'requested_sample_ratio': args.sample_ratio, 'seed': args.seed,
        'sampling': 'uniform without replacement; sampled IDs searched in ascending order',
        'sampled_ids_sha256': file_hash(sample_path), 'L': args.L, 'threads': args.threads,
        'candidate_semantics': 'expanded graph candidates before filtering, including sampled query self if expanded',
        'ranking': 'frequency descending, then ID ascending (including zero-frequency tail)',
        'ground_truth': False, 'exact_reranking': False, 'filtering': False,
        'search_binary': str(binary), 'search_binary_sha256': file_hash(binary),
        'counts_format': 'uint32 N, uint32 1, then N uint64 counts in original ID order',
        'ranking_format': 'uint32 N, then N uint32 IDs, compatible with --filter-cache',
    }
    manifest = out/'metadata.json'
    manifest.write_text(json.dumps(metadata, indent=2)+'\n')
    try:
        with tempfile.TemporaryDirectory(prefix='sampling-', dir=out) as tmp:
            tmp = Path(tmp)
            queries = tmp/'sampled_queries.bin'
            base = np.memmap(base_path, dtype='<f4', mode='r', offset=8, shape=(n, d))
            with queries.open('wb') as dst:
                dst.write(struct.pack('<II', count, d))
                for start in range(0, count, 4096):
                    block = np.asarray(base[sampled[start:start+4096]], dtype='<f4')
                    if not np.isfinite(block).all():
                        raise ValueError('Non-finite sampled base vector')
                    dst.write(block.tobytes())
            del base
            empty = tmp/'empty_cache.id'
            empty.write_bytes(struct.pack('<I', 0))
            # K=1 is only an API placeholder: all expanded candidates, not the
            # top-K results, are counted. No filtering payloads need live in RAM.
            cmd = [str(binary), str(prefix), str(queries), pca_dir, '1', str(args.L),
                   '1', str(args.threads), '-', '-', str(subpq), '--quantized-only',
                   '--filter-cache', str(empty), '0', '--candidate-frequency', str(out/'rank')]
            metadata['command'] = cmd
            env = {k: v for k, v in os.environ.items() if not k.startswith('SKIPDISK_BENCH_')}
            env['OMP_NUM_THREADS'] = str(args.threads)
            env['OPENBLAS_NUM_THREADS'] = '1'
            print(f'Sampled {count}/{n} base points; L={args.L}. Progress log: {out / "search.log"}', flush=True)
            with (out/'search.log').open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        counts_path = out/'rank_access_counts.bin'
        rank_path = out/'rank_access_sorted.id'
        if matrix_shape(counts_path, 8) != (n, 1):
            raise ValueError('Invalid generated count file')
        with rank_path.open('rb') as src:
            if src.read(4) != struct.pack('<I', n) or rank_path.stat().st_size != 4+4*n:
                raise ValueError('Invalid generated rank file')
        counts = np.memmap(counts_path, dtype='<u8', mode='r', offset=8, shape=(n,))
        visited, total = 0, 0
        for start in range(0, n, 1024*1024):
            block = counts[start:start+1024*1024]
            visited += int(np.count_nonzero(block))
            total += int(block.sum(dtype=np.uint64))
        del counts
        metadata.update(status='complete', accessed_points=visited, accessed_fraction=visited/n, zero_frequency_points=n-visited,
                        total_candidates=total, mean_candidates=total/count,
                        ranking_sha256=file_hash(rank_path), counts_sha256=file_hash(counts_path))
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Complete: {visited}/{n} points accessed; ranking: {rank_path}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-prefix', type=Path, required=True)
    parser.add_argument('--subpq-prefix', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True, help='Indexed float32 .bin base in original ID order')
    parser.add_argument('--pca-dir', default='-', help="'-' for PCA base; otherwise PCA model directory for raw base")
    parser.add_argument('--search-binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are refused')
    sample = parser.add_mutually_exclusive_group(required=True)
    sample.add_argument('--sample-size', type=int)
    sample.add_argument('--sample-ratio', type=float)
    parser.add_argument('--L', type=int, required=True, help='Fixed graph search list length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Access ranking failed: {exc}\n')


if __name__ == '__main__':
    main()
