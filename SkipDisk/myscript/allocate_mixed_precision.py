"""Variance allocation: largest feasible discrete c=t/(lambda**alpha) threshold.

Each trial maps all coordinates at one common c, repairs incomplete groups
downward, and tests aligned code cost. There are no independent promotions.
"""
import math
import numpy as np

WIDTHS = (1, 1, 2, 4, 8)
RATES = (16, 8, 4, 2, 1)


def coordinate_variances(base, max_dim=None, batch_size=8192):
    """Population variance, full-data streaming parallel-variance merge in float64."""
    dim = base.shape[1] if max_dim is None else max_dim
    if not 0 < dim <= base.shape[1] or batch_size <= 0:
        raise ValueError('Invalid variance dimensions or batch size')
    mean, m2, count = np.zeros(dim), np.zeros(dim), 0
    for lo in range(0, len(base), batch_size):
        block = np.array(base[lo:lo+batch_size, :dim], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError('Non-finite PCA input')
        k = len(block)
        local = block.mean(axis=0)
        delta = local - mean
        block -= local
        m2 += np.einsum('ij,ij->j', block, block) + delta**2 * (count*k/(count+k))
        mean += delta * (k/(count+k))
        count += k
    if not count:
        raise ValueError('Empty PCA input')
    return m2/count


def continuous_allocation(variances, bits):
    """Continuous reference only; the paper's discrete allocator does not use it."""
    v = np.asarray(variances, dtype=np.float64)
    positive = v > 0
    target = min(float(bits), 16*int(positive.sum()))
    if target <= 0 or not positive.any():
        return np.zeros_like(v), 0.0
    if target == 16*int(positive.sum()):
        # Any c >= 16/min positive variance saturates all positive coordinates.
        coefficient = 16/float(v[positive].min())
        return positive.astype(float)*16, coefficient if math.isfinite(coefficient) else None
    weights = v/v.max()
    lower, upper = 0.0, 16.0
    while np.minimum(16, upper*weights).sum() < target:
        upper *= 2
        if not math.isfinite(upper):
            raise ValueError('Variance dynamic range exceeds floating-point allocation limits')
    for _ in range(100):
        middle = (lower+upper)/2
        if np.minimum(16, middle*weights).sum() <= target:
            lower = middle
        else:
            upper = middle
    coefficient = lower/float(v.max())
    return np.minimum(16, lower*weights), coefficient if math.isfinite(coefficient) else None


def footprint(counts, points):
    m,n,n2,n4,n8 = map(int, counts)
    d = sum(counts)
    code = 2*m+n+n2//2+n4//4+n8//8
    padded = (code+3)//4*4
    books = (d-m)*256*4
    # This is filtering data only, NOT graph, traversal codes, or process RSS.
    total = points*(padded+4)+books+48 if d else 0
    return dict(pivot_dim=d,m=m,n=n,pq2_dims=n2,pq4_dims=n4,pq8_dims=n8,
                code_bytes=code,aligned_code_bytes=padded,codebook_bytes=books,
                radius_bytes=4*points if d else 0,header_bytes=48 if d else 0,
                mixed_bytes=total)


def variance_weights(variances, alpha=1.0):
    """Unnormalized lambda**alpha; alpha=0 gives every axis weight 1, including zeros."""
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError('alpha must be finite and nonnegative')
    try:
        alpha = float(alpha)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('alpha must be finite and nonnegative') from None
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    v = np.asarray(variances, dtype=np.longdouble)
    if v.ndim != 1 or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('Variances must be a finite nonnegative vector')
    if alpha == 0:
        return np.ones_like(v)  # Explicit uniform endpoint, independent of 0**0 conventions.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        weights = v.copy() if alpha == 1 else np.power(v, np.longdouble(alpha))
    if not np.isfinite(weights).all() or np.any((v > 0) & (weights == 0)):
        raise ValueError('alpha/variance range exceeds supported weight arithmetic')
    return weights


def allocation_thresholds(variances, alpha=1.0, *, pq_only=False):
    """Thresholds for every supported tier, plus the sorted unique event list.

    Use extended precision for very small positive variances. Comparing c with
    the precomputed thresholds avoids (t/lambda)*lambda rounding below t at the
    very event that should activate a tier. Zero variance stays inactive for alpha>0.
    """
    if not isinstance(pq_only, (bool, np.bool_)):
        raise ValueError('pq_only must be boolean')
    v=variance_weights(variances, alpha)
    thresholds=np.full((len(v),len(RATES)),np.inf,dtype=np.longdouble)
    positive=v>0
    first=1 if pq_only else 0  # Keep the BF16 column inactive; preserve the existing layout.
    thresholds[positive,first:]=np.asarray(RATES[first:],dtype=np.longdouble)[None,:]/v[positive,None]
    events=np.unique(np.concatenate((np.zeros(1,dtype=np.longdouble),thresholds[positive,first:].ravel())))
    if not np.isfinite(events).all():
        raise ValueError('Variance range exceeds supported threshold arithmetic')
    return thresholds,events


def grouped_counts(thresholds, coefficient):
    """Map at one c, carry incomplete suffixes downward, omit final remainder."""
    reached=coefficient>=thresholds
    active=reached.any(axis=1)
    tier=reached.argmax(axis=1)  # Rates ordered 16 -> 1.
    counts=[int(np.sum(active & (tier==i))) for i in range(len(RATES))]
    for i in range(2,5):
        remainder=counts[i]%WIDTHS[i]
        counts[i]-=remainder
        if i<4:
            counts[i+1]+=remainder
    return counts


def serial_coefficient(value):
    # c can exceed float64 for valid subnormal variances. Preserve its decimal
    # form separately instead of emitting nonstandard JSON Infinity.
    return float(value) if abs(value)<=np.finfo('f8').max else None


def budget_coefficient(weights, bits, cap):
    """Solve sum(min(cap, c*w))=bits before discretization, with saturation."""
    active = weights[weights > 0].copy()
    remaining = np.longdouble(min(bits, cap*len(active)))
    if not len(active) or remaining == 0:
        return np.longdouble(0)
    while len(active):
        scale = active.max()
        coefficient = (remaining / (active/scale).sum()) / scale
        saturated = coefficient >= np.longdouble(cap)/active
        if not saturated.any() or saturated.all():
            if saturated.all():
                coefficient = np.longdouble(cap)/active.min()
            return coefficient
        remaining -= cap*int(saturated.sum())
        active = active[~saturated]


def allocate(variances, points, *, budget_bytes=None, mixed_budget_bytes=None, alpha=1.0,
             pq_only=False, fixed_c=None, c_mode='discrete'):
    """Exactly one budget: aligned code bytes/point OR total filtering bytes.

    Total filtering bytes include codebooks, one float32 radius/point, and the
    v3 header, but exclude graph adjacency/degree, traversal data and runtime RAM.
    No rotation/reordering is performed: input must be ordered PCA variances.
    alpha defaults to 1 (original rule); alpha>1 emphasizes higher variances.
    pq_only disables BF16 before solving c, capping precision at PQ1 (8 bits/axis).
    fixed_c bypasses budget search: map at exactly this coefficient and reject
    an over-budget result. Never increase or reduce the supplied coefficient.
    c_mode='continuous' solves the capped continuous budget, then rounds down
    once without spending the remainder. It does not accept an external c.
    c_mode='initial-support' freezes axes with uncapped c0*weight >= 1, then
    searches discrete thresholds only on those axes, never reviving the tail.
    """
    v = np.asarray(variances, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('Variances must be a finite nonnegative vector')
    if np.any(v[1:] > v[:-1]):
        raise ValueError('PCA variances must be nonincreasing; do not pass untransformed or unordered coordinates')
    if not isinstance(points, (int,np.integer)) or points <= 0:
        raise ValueError('Point count must be positive')
    if (budget_bytes is None) == (mixed_budget_bytes is None):
        raise ValueError('Specify exactly one budget mode')
    budget = budget_bytes if budget_bytes is not None else mixed_budget_bytes
    if not isinstance(budget,(int,np.integer)) or budget < 0:
        raise ValueError('Budget must be a nonnegative integer number of bytes')
    if c_mode not in ('discrete', 'continuous', 'initial-support'):
        raise ValueError('c_mode must be discrete, continuous or initial-support')
    if c_mode != 'discrete' and fixed_c is not None:
        raise ValueError('Budget-derived c_mode cannot be combined with fixed_c')
    weights=variance_weights(v, alpha)
    positive_dim = int(np.count_nonzero(weights))
    # Reserve shared storage BEFORE choosing the fixed per-point code budget R,
    # as in the paper. Do not binary-search total footprint: decreasing codebook
    # size when moving coordinates into BF16 can make that quantity nonmonotone.
    capacity = (budget if budget_bytes is not None else
                max(0,(budget-48-points*4-positive_dim*1024)//points))
    capacity = capacity//4*4
    thresholds,events=allocation_thresholds(v, alpha, pq_only=pq_only)
    support_report = {}
    if c_mode == 'initial-support':
        # Determine eligibility BEFORE capping rates or repairing PQ groups.
        # Sorted PCA weights ensure that this mask is a contiguous prefix.
        scale = weights.max()
        c0 = ((np.longdouble(capacity*8)/(weights/scale).sum())/scale
              if scale > 0 and capacity else np.longdouble(0))
        if not np.isfinite(c0):
            raise ValueError('Initial coefficient exceeds supported arithmetic')
        kept = (weights > 0) & (c0 >= thresholds[:, -1])
        eligible_dim = int(kept.sum())
        assert np.all(kept[:eligible_dim]) and not np.any(kept[eligible_dim:])
        initial = footprint(grouped_counts(thresholds, c0), points)
        assert initial['aligned_code_bytes'] <= capacity
        thresholds[~kept, :] = np.inf
        # Include c0 itself, even when every eligible coordinate is saturated.
        events = np.unique(np.concatenate(([c0], thresholds[np.isfinite(thresholds) & (thresholds >= c0)])))
        support_report = dict(method='initial_support_discrete_threshold_search',
                              c_source='initial_support_budget',
                              initial_c=serial_coefficient(c0), initial_c_decimal=str(c0),
                              eligible_dim=eligible_dim, initial_excluded_dims=len(v)-eligible_dim,
                              initial_pivot_dim=initial['pivot_dim'],
                              initial_aligned_code_bytes=initial['aligned_code_bytes'])
    if c_mode == 'continuous':
        fixed_c = budget_coefficient(weights, capacity*8, 8 if pq_only else 16)
    if fixed_c is not None:
        if isinstance(fixed_c, (bool, np.bool_)) or not np.isscalar(fixed_c):
            raise ValueError('fixed_c must be finite and nonnegative')
        try:
            coefficient=np.longdouble(fixed_c)
        except (ValueError, TypeError, OverflowError):
            raise ValueError('fixed_c must be finite and nonnegative') from None
        if not np.isfinite(coefficient) or coefficient<0:
            raise ValueError('fixed_c must be finite and nonnegative')
        counts=grouped_counts(thresholds,coefficient)
        result=footprint(counts,points)
        used=result['aligned_code_bytes'] if budget_bytes is not None else result['mixed_bytes']
        if used>budget:
            raise ValueError(f'Fixed c allocation exceeds budget: {used} bytes required, {budget} bytes available')
        with np.errstate(over='ignore'):
            ideal=np.minimum(np.longdouble(8 if pq_only else 16),coefficient*weights)
        result.update(budget_mode='aligned_code_bytes_per_point' if budget_bytes is not None else 'total_mixed_bytes',
                      budget=int(budget),points=int(points),candidate_dim=len(v),alpha=float(alpha),
                      pq_only=bool(pq_only),method='fixed_c',c_source='fixed',
                      omitted_dims=len(v)-sum(counts),
                      code_capacity_bytes=int(budget//4*4) if budget_bytes is not None else None,
                      c=serial_coefficient(coefficient),c_decimal=str(coefficient),
                      ideal_total_bits=float(ideal.sum()),search_trials=0,
                      unused_budget_bytes=int(budget-used),
                      variance_fraction_scope='candidate PCA prefix',
                      retained_variance_fraction=float(v[:sum(counts)].sum()/v.sum()) if v.sum()>0 else 0.0)
        if c_mode == 'continuous':
            result.update(method='continuous_budget_then_round_down', c_source='continuous_budget',
                          code_capacity_bytes=int(capacity),
                          reserved_codebook_bytes=positive_dim*1024 if mixed_budget_bytes is not None else None)
        return result
    low,high,best,trials=0,len(events)-1,0,0
    while low<=high:
        middle=(low+high)//2
        candidate=grouped_counts(thresholds,events[middle])
        trials+=1
        if footprint(candidate,points)['aligned_code_bytes']<=capacity:
            best=middle
            low=middle+1
        else:
            high=middle-1
    coefficient=events[best]
    counts=grouped_counts(thresholds,coefficient)
    with np.errstate(over='ignore'):
        ideal=np.minimum(np.longdouble(8 if pq_only else 16),coefficient*weights)
    if c_mode == 'initial-support':
        ideal[~kept] = 0
    next_coefficient=events[best+1] if best+1<len(events) else None
    next_cost=(footprint(grouped_counts(thresholds,next_coefficient),points)['aligned_code_bytes']
               if next_coefficient is not None else None)
    assert next_cost is None or next_cost>capacity
    result=footprint(counts,points)
    result.update(budget_mode='aligned_code_bytes_per_point' if budget_bytes is not None else 'total_mixed_bytes',
                  budget=int(budget),points=int(points),candidate_dim=len(v),alpha=float(alpha),
                  pq_only=bool(pq_only),
                  method='paper_discrete_threshold_search',
                  omitted_dims=len(v)-sum(counts),code_capacity_bytes=int(capacity),
                  c=serial_coefficient(coefficient),c_decimal=str(coefficient),
                  ideal_total_bits=float(ideal.sum()),
                  threshold_count=len(events),selected_threshold_index=best,search_trials=trials,
                  next_c=serial_coefficient(next_coefficient) if next_coefficient is not None else None,
                  next_c_decimal=str(next_coefficient) if next_coefficient is not None else None,
                  next_aligned_code_bytes=next_cost,
                  reserved_codebook_bytes=positive_dim*1024 if mixed_budget_bytes is not None else None,
                  variance_fraction_scope='candidate PCA prefix',
                  retained_variance_fraction=float(v[:sum(counts)].sum()/v.sum()) if v.sum()>0 else 0.0)
    used=result['aligned_code_bytes'] if budget_bytes is not None else result['mixed_bytes']
    assert used<=budget
    result['unused_budget_bytes']=int(budget-used)
    result.update(support_report)
    return result


def allocate_precision_truncate(variances, points, *, pivot_dim, budget_bytes=None,
                                mixed_budget_bytes=None, alpha=1.0, pq_only=False, fixed_c=None,
                                c_mode='discrete'):
    """Fix the PCA prefix BEFORE computing c and allocating precision.

    Only the first pivot_dim original variances, raised to alpha, participate in threshold search
    and codebook reservation. Do not normalize them or derive c from a full-D
    baseline. The final retained count can be smaller due to zero/low rates and
    incomplete groups. This is the paper allocator applied to a fixed prefix.
    """
    v=np.asarray(variances,dtype=np.float64)
    if v.ndim!=1 or not len(v):
        raise ValueError('Variances must be a nonempty vector')
    if (not isinstance(pivot_dim,(int,np.integer)) or isinstance(pivot_dim,(bool,np.bool_))
            or not 0<pivot_dim<=len(v)):
        raise ValueError('pivot_dim must be an integer in [1, number of input dimensions]')
    result=allocate(v[:pivot_dim],points,budget_bytes=budget_bytes,
                    mixed_budget_bytes=mixed_budget_bytes,alpha=alpha,pq_only=pq_only,fixed_c=fixed_c,c_mode=c_mode)
    method = ('initial_support_fixed_prefix' if c_mode == 'initial-support' else
              'continuous_budget_fixed_prefix' if c_mode == 'continuous' else
              'fixed_c_fixed_prefix' if fixed_c is not None else 'variance_fixed_prefix')
    result.update(method=method,allocation_order='truncate_then_allocate',
                  requested_pivot_dim=int(pivot_dim))
    assert result['pivot_dim']<=pivot_dim
    return result
