#!/usr/bin/env python3
"""
Broadcasting benchmark worker — runs one strategy and writes metrics to JSON.

Usage:
  python3 broadcasting_worker.py <strategy> <duration_s> <seed> <output_json>

Strategies
----------
  non_graph_arrays     – fix ALL variables (both ndims + both shapes) randomly, SAT check
  non_graph_integers   – same, with named integer dimension variables
  graph_arrays         – fix BOTH ndims randomly, Z3 solves all shapes (arrays)
  graph_integers       – same, with named integer dimension variables
  fix_one_side_arrays  – fix ONE tensor (input_ndim + input_shape) randomly,
                         Z3 solves the other tensor (other_ndim + other_shape), arrays
  fix_one_side_integers– same, with named integer dimension variables
"""

import sys
import json
import time
import numpy as np
from z3 import *

MAX_NDIM = 6
MAX_DIM  = 64

STRATEGY_LABELS = {
    'non_graph_arrays':      'Non-graph + Arrays',
    'non_graph_integers':    'Non-graph + Integers',
    'graph_arrays':          'Graph-based + Arrays',
    'graph_integers':        'Graph-based + Integers',
    'fix_one_side_arrays':   'Fix-one-side + Arrays',
    'fix_one_side_integers': 'Fix-one-side + Integers',
}


# ── constraint builders ───────────────────────────────────────────────────────

def _build_array():
    s     = Solver()
    n_in  = Int('input_ndim')
    n_ot  = Int('other_ndim')
    sh_in = Array('input_shape', IntSort(), IntSort())
    sh_ot = Array('other_shape', IntSort(), IntSort())

    s.add(n_in >= 1, n_in <= MAX_NDIM)
    s.add(n_ot >= 1, n_ot <= MAX_NDIM)
    for i in range(MAX_NDIM):
        s.add(Implies(n_in > i, And(Select(sh_in, i) >= 1, Select(sh_in, i) <= MAX_DIM)))
        s.add(Implies(n_ot > i, And(Select(sh_ot, i) >= 1, Select(sh_ot, i) <= MAX_DIM)))

    # Broadcasting: align from the right end of the wider tensor's shape.
    s.add(If(
        n_in > n_ot,
        And([Implies(i < n_ot,
             Or(Select(sh_in, n_in - n_ot + i) == Select(sh_ot, i),
                Select(sh_ot, i) == 1))
             for i in range(MAX_NDIM)]),
        And([Implies(i < n_in,
             Or(Select(sh_ot, n_ot - n_in + i) == Select(sh_in, i),
                Select(sh_in, i) == 1))
             for i in range(MAX_NDIM)])))

    return s, n_in, n_ot, sh_in, sh_ot


def _build_integer():
    s    = Solver()
    n_in = Int('input_ndim')
    n_ot = Int('other_ndim')
    d_in = [Int(f'in_dim{k}') for k in range(MAX_NDIM)]
    d_ot = [Int(f'ot_dim{k}') for k in range(MAX_NDIM)]

    s.add(n_in >= 1, n_in <= MAX_NDIM)
    s.add(n_ot >= 1, n_ot <= MAX_NDIM)
    for i in range(MAX_NDIM):
        s.add(Implies(n_in > i, And(d_in[i] >= 1, d_in[i] <= MAX_DIM)))
        s.add(Implies(n_ot > i, And(d_ot[i] >= 1, d_ot[i] <= MAX_DIM)))

    # Per-pair Implies: input dim j and other dim i align when n_in - n_ot == j - i.
    s.add(And([
        Implies(And(n_in - n_ot == j - i, i < n_ot, j < n_in),
                Or(d_in[j] == d_ot[i], d_ot[i] == 1))
        for i in range(MAX_NDIM) for j in range(MAX_NDIM)
    ]))

    return s, n_in, n_ot, d_in, d_ot


# ── fix functions — each owns all random assignments for its strategy ─────────
#
# Signature: fix_fn(solver, rng, n_in, n_ot, *shape_vars)
#   n_in, n_ot    Z3 Int variables for tensor rank
#   shape_vars    either [sh_in, sh_ot] (arrays) or [d_in, d_ot] (int lists)
#
# Each function adds equality constraints to the solver for the variables it
# controls. Variables not constrained here are left for Z3 to solve freely.

def _fix_non_graph_array(s, rng, n_in, n_ot, sh_in, sh_ot):
    """Fix both ndims and both full shape arrays randomly."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    s.add(n_ot == int(rng.integers(1, MAX_NDIM + 1)))
    for k in range(MAX_NDIM):
        s.add(Select(sh_in, k) == int(rng.integers(1, MAX_DIM + 1)))
        s.add(Select(sh_ot, k) == int(rng.integers(1, MAX_DIM + 1)))


def _fix_non_graph_integer(s, rng, n_in, n_ot, d_in, d_ot):
    """Fix both ndims and all individual dimension integers randomly."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    s.add(n_ot == int(rng.integers(1, MAX_NDIM + 1)))
    for k in range(MAX_NDIM):
        s.add(d_in[k] == int(rng.integers(1, MAX_DIM + 1)))
        s.add(d_ot[k] == int(rng.integers(1, MAX_DIM + 1)))


def _fix_graph_array(s, rng, n_in, n_ot, sh_in, sh_ot):
    """Fix both ndims randomly; Z3 solves all shape values."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    s.add(n_ot == int(rng.integers(1, MAX_NDIM + 1)))


def _fix_graph_integer(s, rng, n_in, n_ot, d_in, d_ot):
    """Fix both ndims randomly; Z3 solves all dimension integers."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    s.add(n_ot == int(rng.integers(1, MAX_NDIM + 1)))


def _fix_one_side_array(s, rng, n_in, n_ot, sh_in, sh_ot):
    """Fix input tensor (ndim + full shape) randomly; Z3 solves other tensor."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    for k in range(MAX_NDIM):
        s.add(Select(sh_in, k) == int(rng.integers(1, MAX_DIM + 1)))
    # other_ndim and other_shape left entirely for Z3


def _fix_one_side_integer(s, rng, n_in, n_ot, d_in, d_ot):
    """Fix input tensor (ndim + all dims) randomly; Z3 solves other tensor."""
    s.add(n_in == int(rng.integers(1, MAX_NDIM + 1)))
    for k in range(MAX_NDIM):
        s.add(d_in[k] == int(rng.integers(1, MAX_DIM + 1)))
    # other_ndim and other dims left entirely for Z3


# ── feature extraction ────────────────────────────────────────────────────────

def _ei(m, expr):
    try:
        return m.eval(expr, model_completion=True).as_long()
    except Exception:
        return 0


def _array_feat(m, n_in, n_ot, sh_in, sh_ot):
    ni, no = _ei(m, n_in), _ei(m, n_ot)
    # Zero out inactive dimensions so uniqueness is semantically meaningful.
    s_in = [_ei(m, Select(sh_in, k)) if k < ni else 0 for k in range(MAX_NDIM)]
    s_ot = [_ei(m, Select(sh_ot, k)) if k < no else 0 for k in range(MAX_NDIM)]
    return (ni, no, *s_in, *s_ot)


def _integer_feat(m, n_in, n_ot, d_in, d_ot):
    ni, no = _ei(m, n_in), _ei(m, n_ot)
    s_in = [_ei(m, d_in[k]) if k < ni else 0 for k in range(MAX_NDIM)]
    s_ot = [_ei(m, d_ot[k]) if k < no else 0 for k in range(MAX_NDIM)]
    return (ni, no, *s_in, *s_ot)


# ── diversity metrics ─────────────────────────────────────────────────────────

def _mpd(mat, n_pairs=5000):
    n = len(mat)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    col_rng = mat.max(0) - mat.min(0)
    col_rng[col_rng == 0] = 1.0
    pairs = min(n_pairs, n * (n - 1) // 2)
    i1 = rng.integers(0, n, pairs * 3)
    i2 = rng.integers(0, n, pairs * 3)
    keep = i1 != i2
    i1, i2 = i1[keep][:pairs], i2[keep][:pairs]
    diffs = np.abs(mat[i1] - mat[i2]) / col_rng
    return float(diffs.sum(axis=1).mean())


def _entropy(mat):
    if len(mat) == 0:
        return 0.0
    h = 0.0
    for c in range(mat.shape[1]):
        _, cnt = np.unique(mat[:, c], return_counts=True)
        p = cnt / cnt.sum()
        h += float(-np.sum(p * np.log2(p + 1e-12)))
    return h / mat.shape[1]


# ── strategy registry ─────────────────────────────────────────────────────────

STRATEGY_MAP = {
    'non_graph_arrays':      (_build_array,   _array_feat,   _fix_non_graph_array),
    'non_graph_integers':    (_build_integer, _integer_feat, _fix_non_graph_integer),
    'graph_arrays':          (_build_array,   _array_feat,   _fix_graph_array),
    'graph_integers':        (_build_integer, _integer_feat, _fix_graph_integer),
    'fix_one_side_arrays':   (_build_array,   _array_feat,   _fix_one_side_array),
    'fix_one_side_integers': (_build_integer, _integer_feat, _fix_one_side_integer),
}


# ── main loop ─────────────────────────────────────────────────────────────────

def run(strategy, duration, seed, output_path):
    build_fn, feat_fn, fix_fn = STRATEGY_MAP[strategy]
    label = STRATEGY_LABELS[strategy]
    rng   = np.random.default_rng(seed)

    base, n_in, n_ot, *sh_vars = build_fn()
    samples, seen = [], set()
    sat_n = unsat_n = 0
    start     = time.time()
    next_tick = start + 60

    print(f"[{label}] starting  duration={duration}s  seed={seed}", flush=True)

    try:
        while time.time() - start < duration:
            s = Solver()
            s.add(*base.assertions())
            fix_fn(s, rng, n_in, n_ot, *sh_vars)

            if s.check() == sat:
                sat_n += 1
                feat = feat_fn(s.model(), n_in, n_ot, *sh_vars)
                samples.append(feat)
                seen.add(feat)
            else:
                unsat_n += 1

            now = time.time()
            if now >= next_tick:
                elapsed = int(now - start)
                total   = sat_n + unsat_n
                pct     = 100.0 * sat_n / total if total else 0.0
                print(
                    f"[{label}] {elapsed:3d}s  "
                    f"models={sat_n:>7}  attempts={total:>8}  SAT%={pct:.1f}%",
                    flush=True)
                next_tick = now + 60

    finally:
        n       = len(samples)
        total   = sat_n + unsat_n
        sat_pct = 100.0 * sat_n / total if total else 0.0

        if n >= 2:
            mat = np.array(samples, dtype=float)
            mpd = _mpd(mat)
            ent = _entropy(mat)
        else:
            mpd = ent = 0.0

        result = {
            'strategy': strategy,
            'label':    label,
            'n_models': n,
            'n_unique': len(seen),
            'sat_n':    sat_n,
            'unsat_n':  unsat_n,
            'sat_pct':  round(sat_pct, 4),
            'mpd':      round(mpd, 6),
            'entropy':  round(ent, 6),
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - start
        print(
            f"[{label}] finished  elapsed={elapsed:.0f}s  "
            f"models={n}  unique={len(seen)}  MPD={mpd:.3f}  Entropy={ent:.3f}",
            flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(
            'Usage: python3 broadcasting_worker.py '
            '<strategy> <duration_s> <seed> <output_json>\n'
            f'  strategy: {" | ".join(STRATEGY_MAP)}')
        sys.exit(1)

    strat, dur, sd, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

    if strat not in STRATEGY_MAP:
        print(f"Unknown strategy '{strat}'. Choose from: {', '.join(STRATEGY_MAP)}")
        sys.exit(1)

    run(strat, dur, sd, out)
