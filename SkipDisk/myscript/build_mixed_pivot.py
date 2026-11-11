#!/usr/bin/env python3
"""Build BF16 + 256-center PQ tiers with 1/2/4/8 dimensions per code.

Input base vectors must already be PCA-transformed float32, in original ID
order and in the same coordinate system as PREFIX_data_sector.bin.
"""
import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def read_bin(path):
    with open(path, "rb") as src:
        header = src.read(8)
    if len(header) != 8:
        raise ValueError(f"Truncated bin header: {path}")
    rows, dim = struct.unpack("<II", header)
    if not rows or not dim or os.path.getsize(path) != 8 + rows * dim * 4:
        raise ValueError(f"Invalid float32 bin: {path}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def subspaces(dim, m, n, pq2_dims=None, pq4_dims=None):
    if dim <= 0 or dim > 65536 or m < 0 or n < 0 or m + n > dim:
        raise ValueError("Require 0 <= m, 0 <= n, m+n <= D <= 65536")
    if pq2_dims is not None or pq4_dims is not None:
        n2 = 0 if pq2_dims is None else pq2_dims
        n4 = 0 if pq4_dims is None else pq4_dims
        n8 = dim - m - n - n2 - n4
        if min(n2, n4, n8) < 0 or n2 % 2 or n4 % 4 or n8 % 8:
            raise ValueError('Require PQ2 dimensions divisible by 2, PQ4 by 4, and remaining PQ8 by 8')
        groups, lo = [], m
        for count, width in ((n, 1), (n2, 2), (n4, 4), (n8, 8)):
            groups.extend((d, d + width) for d in range(lo, lo + count, width))
            lo += count
        return groups
    return [(d, d + 1) for d in range(m, m + n)] + [
        (d, min(d + 2, dim)) for d in range(m + n, dim, 2)]


def train(base, m, n, train_size, iterations, seed, pq2_dims=None, pq4_dims=None):
    groups = subspaces(base.shape[1], m, n, pq2_dims, pq4_dims)
    if not groups:
        return []
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(base), min(train_size, len(base)), replace=False)
    sample = np.asarray(base[ids], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("Non-finite training vector")
    books = []
    for i, (lo, hi) in enumerate(groups):
        # Small datasets and constant coordinates are valid, too. Duplicate
        # unused centers so the on-disk and lookup-table stride stays 256.
        data = np.ascontiguousarray(sample[:, lo:hi])
        unique = np.unique(data, axis=0)
        if len(unique) <= 256:
            centers = unique
        else:
            model = MiniBatchKMeans(n_clusters=256, random_state=seed + i,
                                   n_init=1, max_iter=iterations,
                                   batch_size=min(4096, len(data)),
                                   reassignment_ratio=0.0)
            model.fit(data)
            centers = model.cluster_centers_
        centers = np.asarray(centers[np.arange(256) % len(centers)], dtype="<f4")
        books.append(centers)
        if (i + 1) % 32 == 0 or i + 1 == len(groups):
            print(f"Trained {i + 1}/{len(groups)} subspaces", flush=True)
    return books


def encode(block, m, groups, books, trees=None):
    if not np.isfinite(block).all():
        raise ValueError("Non-finite base vector")
    # Match the existing SkipDisk BF16 high-16-bit truncation policy.
    head = (np.ascontiguousarray(block[:, :m], dtype="<f4").view("<u4") >> 16).astype("<u2")
    restored = (head.astype("<u4") << 16).view("<f4").astype(np.float64)
    error = np.sum((block[:, :m].astype(np.float64) - restored) ** 2, axis=1)
    codes = np.empty((len(block), len(groups)), dtype=np.uint8)
    if trees is None:
        trees = [cKDTree(book.astype(np.float64)) for book in books]
    for c, ((lo, hi), book, tree) in enumerate(zip(groups, books, trees)):
        values = block[:, lo:hi].astype(np.float64)
        _, ids = tree.query(values, k=1)
        codes[:, c] = ids
        error += np.sum((values - book[ids].astype(np.float64)) ** 2, axis=1)
    radius = np.sqrt(error).astype("<f4")
    if not np.isfinite(radius).all():
        raise ValueError("Pivot radius exceeds float32 range")
    return head, codes, radius


def build(args):
    if min(args.train_size, args.iterations, args.batch_size, args.threads) <= 0:
        raise ValueError("Training, batch and thread counts must be positive")
    base = read_bin(args.pca_base)
    count, dim = base.shape
    pivot_dim = getattr(args, 'pivot_dim', None)
    pivot_dim = dim if pivot_dim is None else pivot_dim
    if not 0 < pivot_dim <= dim:
        raise ValueError('Require 0 < pivot_dim <= full PCA dimension')
    n2, n4 = getattr(args, 'pq2_dims', None), getattr(args, 'pq4_dims', None)
    extended = n2 is not None or n4 is not None
    groups = subspaces(pivot_dim, args.m, args.n, n2, n4)
    pivot_base = base[:, :pivot_dim]
    output = Path(args.index_prefix + "_navi_mixed_graph.index")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose another prefix or move it first")
    with open(args.graph_index, "rb") as graph_file:
        with mmap.mmap(graph_file.fileno(), 0, access=mmap.ACCESS_READ) as graph:
            if len(graph) < 4096:
                raise ValueError("Truncated DiskANN graph")
            nr, nc = struct.unpack_from("<II", graph)
            if nr != 9 or nc != 1:
                raise ValueError("Expected the 9x1 DiskANN metadata format")
            meta = struct.unpack_from("<9Q", graph, 8)
            points, graph_dim, _, record_bytes, per_sector = meta[:5]
            if points != count or graph_dim != dim or meta[-1] != len(graph):
                raise ValueError("Graph/base count, dimension or graph file size mismatch")
            if record_bytes < dim * 4 + 4 or (record_bytes - dim * 4 - 4) % 4:
                raise ValueError("Invalid DiskANN node size")
            degree = (record_bytes - dim * 4 - 4) // 4
            if degree > 1024 or (per_sector and per_sector * record_bytes > 4096):
                raise ValueError("Invalid graph degree or sector layout")
            io_bytes = 4096 * ((record_bytes + 4095) // 4096) if not per_sector else 4096
            slots = per_sector or 1
            if 4096 + ((count + slots - 1) // slots) * io_bytes > len(graph):
                raise ValueError("Truncated graph sectors")
            with threadpool_limits(limits=args.threads):
                books = train(pivot_base, args.m, args.n, args.train_size, args.iterations, args.seed, n2, n4)
            trees = [cKDTree(book.astype(np.float64)) for book in books]
            code_bytes = 2 * args.m + len(groups)
            padded = (code_bytes + 3) & ~3
            node_bytes = padded + 8 + degree * 4
            dtype = np.dtype({"names": ["head", "codes", "radius", "degree", "neighbors"],
                              "formats": [("<u2", (args.m,)), ("u1", (len(groups),)),
                                          "<f4", "<u4", ("<u4", (degree,))],
                              "offsets": [0, 2 * args.m, padded, padded + 4, padded + 8],
                              "itemsize": node_bytes})
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as dst:
                    temporary = Path(dst.name)
                    dst.write(b"SKMIXPQ1")
                    version = 3 if extended else (1 if pivot_dim == dim else 2)
                    dst.write(struct.pack("<7I", version, count, pivot_dim, args.m, args.n, degree, code_bytes))
                    if version >= 2:
                        dst.write(struct.pack('<I', dim))
                    if version == 3:
                        dst.write(struct.pack('<II', n2 or 0, n4 or 0))
                    for book in books:
                        dst.write(book.tobytes())
                    for start in range(0, count, args.batch_size):
                        end = min(start + args.batch_size, count)
                        records = np.zeros(end - start, dtype=dtype)
                        head, codes, radius = encode(pivot_base[start:end], args.m, groups, books, trees)
                        records["head"], records["codes"], records["radius"] = head, codes, radius
                        for row, idx in enumerate(range(start, end)):
                            offset = 4096 + (idx // slots) * io_bytes + (idx % slots) * record_bytes + dim * 4
                            neighbors = struct.unpack_from("<I", graph, offset)[0]
                            if neighbors > degree:
                                raise ValueError(f"Invalid degree at ID {idx}")
                            ids = np.frombuffer(graph, dtype="<u4", count=neighbors, offset=offset + 4).copy()
                            if np.any(ids >= count):
                                raise ValueError(f"Invalid neighbor ID at ID {idx}")
                            records["degree"][row] = neighbors
                            records["neighbors"][row, :neighbors] = ids
                        dst.write(records.tobytes())
                        print(f"Encoded {end}/{count}", flush=True)
                # Exclusive publication avoids accidentally replacing an existing index.
                os.link(temporary, output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    print(f"Wrote {output}: {code_bytes} pivot bytes/vector, {node_bytes} bytes/node, "
          f"{(pivot_dim - args.m) * 256 * 4} shared codebook bytes, "
          f"pivot dimensions={pivot_dim}, full dimensions={dim}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-base", required=True)
    parser.add_argument("--graph-index", help="Original-ID-order float32 DiskANN _disk.index")
    parser.add_argument("--index-prefix")
    parser.add_argument("--m", type=int, help="BF16 dimensions")
    parser.add_argument("--n", type=int, help="One-dimensional PQ dimensions")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--budget-bytes', type=int, nargs='+',
                        help='Automatic allocation: aligned code bytes per point; multiple budgets require --plan-only')
    budget.add_argument('--mixed-budget-mib', type=float, nargs='+',
                        help='Automatic allocation: total code+radius+codebook+header MiB (excludes graph/traversal/runtime)')
    parser.add_argument('--plan-only', action='store_true', help='Print JSON configurations without training or writing an index')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Automatic allocation exponent: nonnegative, default 1; alpha=0 gives all axes equal weight')
    parser.add_argument('--pq-only', action='store_true',
                        help='Automatic allocation using only PQ1/PQ2/PQ4/PQ8; disable BF16')
    parser.add_argument('--fixed-c', help='Use exactly this nonnegative c with raw lambda**alpha; budget is only a ceiling, no search or refill')
    parser.add_argument('--c-mode', choices=('discrete','continuous','initial-support'), default='discrete',
                        help='Discrete search (default), continuous without refill, or freeze initial >=1bit support then discrete search')
    parser.add_argument('--pca-model', help='Sample-trained PCA model from train_allocation_pca.py; use its eigenvalues, verify projection provenance')
    parser.add_argument('--allocation-output', help='Also save automatic allocation JSON (refuses to overwrite)')
    parser.add_argument('--allocation-mode', choices=('paper','precision-truncate'), default='paper',
                        help='Default paper rule, or explicitly fix --pivot-dim before solving c')
    # Diagnose removed options explicitly instead of silently changing old runs.
    parser.add_argument('--c-scale', nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--c', nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--pq2-dims', type=int, default=None,
                        help='Enable v3: dimensions assigned to 2D PQ (multiple of 2; omitted tier is zero)')
    parser.add_argument('--pq4-dims', type=int, default=None,
                        help='Enable v3: dimensions assigned to 4D PQ (multiple of 4); remaining dimensions use 8D PQ')
    parser.add_argument('--pivot-dim', type=int, default=None,
                        help='Manual retained dimensions, or maximum candidate dimension in automatic mode (default: all)')
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    automatic = args.budget_bytes is not None or args.mixed_budget_mib is not None
    if args.c_mode != 'discrete' and (not automatic or args.fixed_c is not None):
        parser.error('--c-mode '+args.c_mode+' requires a budget and cannot use --fixed-c')
    if args.fixed_c is not None and not automatic:
        parser.error('--fixed-c requires a budget ceiling')
    if args.pca_model and not automatic:
        parser.error('--pca-model requires an automatic budget')
    if args.alpha is not None and not automatic:
        parser.error('--alpha requires an automatic budget')
    if args.pq_only and not automatic:
        parser.error('--pq-only requires an automatic budget')
    if args.alpha is None:
        args.alpha = 1.0
    if not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error('--alpha must be finite and nonnegative')
    if args.c is not None or args.c_scale is not None:
        parser.error('--c-scale/--c are no longer supported: specify --pivot-dim before allocation; c is solved from the budget')
    if args.allocation_mode == 'precision-truncate':
        if not automatic or args.pivot_dim is None:
            parser.error('precision-truncate requires a budget and --pivot-dim')
    if automatic:
        from allocate_mixed_precision import coordinate_variances, allocate, allocate_precision_truncate
        if any(v is not None for v in (args.m,args.n,args.pq2_dims,args.pq4_dims)):
            parser.error('Automatic budgets cannot be combined with manual tier counts')
        requested = args.budget_bytes if args.budget_bytes is not None else args.mixed_budget_mib
        if len(requested)>1 and not args.plan_only:
            parser.error('Multiple budgets require --plan-only; build each configuration with its own prefix')
        if not args.plan_only and (not args.graph_index or not args.index_prefix):
            parser.error('Building requires --graph-index and --index-prefix')
        base=read_bin(args.pca_base)
        if args.pivot_dim is not None and not 1<=args.pivot_dim<=base.shape[1]:
            parser.error('--pivot-dim must be between 1 and the full PCA dimension')
        provenance=None
        if args.pca_model:
            from train_allocation_pca import allocation_eigenvalues
            try:
                variances,provenance=allocation_eigenvalues(args.pca_model,args.pca_base)
            except (ValueError,OSError,KeyError) as error:
                parser.error(str(error))
            if args.pivot_dim is not None:variances=variances[:args.pivot_dim]
        else:
            variances=coordinate_variances(base,args.pivot_dim,args.batch_size)
        reports=[]
        for value in requested:
            if args.budget_bytes is not None:
                budget_args=dict(budget_bytes=value)
            else:
                if not np.isfinite(value) or value < 0:
                    parser.error('Mixed budget must be finite and nonnegative')
                budget_args=dict(mixed_budget_bytes=int(value*1024*1024))
            try:
                if args.allocation_mode=='paper':
                    report=allocate(variances,len(base),alpha=args.alpha,pq_only=args.pq_only,fixed_c=args.fixed_c,c_mode=args.c_mode,**budget_args)
                else:
                    report=allocate_precision_truncate(variances,len(base),pivot_dim=args.pivot_dim,alpha=args.alpha,pq_only=args.pq_only,fixed_c=args.fixed_c,c_mode=args.c_mode,**budget_args)
            except ValueError as error:
                parser.error(str(error))
            report['full_dim']=base.shape[1]
            report['omitted_full_dims']=base.shape[1]-report['pivot_dim']
            reports.append(report)
        payload=json.dumps(dict(method=reports[0]['method'],alpha=args.alpha,pq_only=args.pq_only,
              weight_source='sample_pca_eigenvalues' if provenance else 'projected_population_variance',
              pca_provenance=provenance,
              variance_rows=provenance['sample_count'] if provenance else len(base),configurations=reports),indent=2,allow_nan=False)
        if args.allocation_output:
            destination=Path(args.allocation_output)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('x') as dst:
                dst.write(payload+'\n')
        print(payload,flush=True)
        if args.plan_only:
            return
        selected=reports[0]
        if selected['pivot_dim']==0:
            parser.error('Budget/data produced no retained dimensions; no index can be built')
        for key in ('pivot_dim','m','n','pq2_dims','pq4_dims'):
            setattr(args,key,selected[key])
    else:
        if args.plan_only or args.allocation_output:
            parser.error('--plan-only and --allocation-output require a budget')
        if args.m is None or args.n is None or not args.graph_index or not args.index_prefix:
            parser.error('Manual build requires --m, --n, --graph-index and --index-prefix')
    build(args)


if __name__ == "__main__":
    main()
