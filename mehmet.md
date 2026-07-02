# Bug Report: `torch.cdist(p=2)` returns `NaN` instead of `Inf` on the matmul path

---

## Title

`torch.cdist(p=2)` returns `NaN` instead of `Inf` for inputs with an `Inf` coordinate when the matmul compute path is selected (rows > 25)

## Summary

The Euclidean distance from a point that has a `+Inf` coordinate to any finite point is
mathematically `+Inf`. `torch.cdist(x1, x2, p=2)` returns this correctly with its direct
computation path, but returns `NaN` once the input is large enough (`r1 > 25` or `r2 > 25`) to
trigger the default matrix-multiplication path. The result is silently wrong and depends only on
the **number of rows**, not on the data.

## To Reproduce

```python
import torch

q = torch.zeros(1, 4)
q[0, 0] = float("inf")        # query point with an Inf coordinate
B = torch.randn(6, 4)         # finite reference points

def dist_row0(num_rows):
    A = torch.cat([q, torch.randn(num_rows - 1, 4)], dim=0)
    return torch.cdist(A, B, p=2)[0]

print("r1=25 (direct path):", dist_row0(25).tolist())
print("r1=26 (matmul path):", dist_row0(26).tolist())
print("donot_use_mm       :",
      torch.cdist(torch.cat([q, torch.randn(25, 4)], 0), B, p=2,
                  compute_mode="donot_use_mm_for_euclid_dist")[0].tolist())
```

Output:

```
r1=25 (direct path): [inf, inf, inf, inf, inf, inf]
r1=26 (matmul path): [inf, nan, nan, nan, nan, nan]
donot_use_mm       : [inf, inf, inf, inf, inf, inf]
```

## Expected behavior

All distances are `Inf` (the distance to a point at infinity), as produced by the direct path,
by `compute_mode='donot_use_mm_for_euclid_dist'`, and by a float64 brute-force reference.

## Actual behavior

On the default `compute_mode='use_mm_for_euclid_dist_if_necessary'`, once `r1 > 25` (or
`r2 > 25`) the matmul path returns `NaN` for those distances.

## Root cause

For `p == 2` the matmul path computes squared distances via the identity
`||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`. When a coordinate is `Inf`, this becomes
`Inf + finite - 2*Inf = Inf - Inf = NaN`, then `sqrt(NaN) = NaN`. The direct path computes
`sqrt(sum((a - b)^2))`, where `(Inf - finite)^2 = Inf` and `sqrt(Inf) = Inf` (correct). The path
is chosen by the documented threshold (`P > 25 or R > 25`) of
`compute_mode='use_mm_for_euclid_dist_if_necessary'`. The accuracy tradeoff of this mode is
documented, but the `Inf -> NaN` special-value change is not.

## Trigger condition

Default `compute_mode`. The same input is correct with `r1 <= 25` and wrong with `r1 > 25`;
only the row count changes the algorithm.

## Workaround

Pass `compute_mode='donot_use_mm_for_euclid_dist'`.

## Environment

Run `python -m torch.utils.collect_env` and paste the output. Observed on PyTorch built from
source at revision `040e3bf86e7b9bb63e56f968ff880fefd83cbd58` (CPU). Reproduces with the default
`compute_mode`; expected to affect released versions with the same matmul path.

## How this was found

Branch-distance instrumentation of `aten/src/ATen/native/Distance.cpp` flagged
`cdist/cdist_impl/r1_gt_25` as a fork between two implementations that should agree. Differential
testing of that fork against a float64 reference, extended with `Inf`/`NaN` inputs, surfaced the
discrepancy. Repro script: `agent_space/cdist_inf_repro.py`.[11:04 AM]What we did

  Starting from one instrumentation-guided discovery (the cdist Inf→NaN bug), we generalized its pattern into a systematic campaign. The recipe:

  1. Target algorithm-selection forks — branches that choose between two implementations of the same math (compute-mode flags, size thresholds, contiguous
  vs channels_last kernels, fast paths, exact vs approximate variants), drawn from our branch-distance instrumentation and ranked candidate analysis
  (api_candidates*.md).
  2. Exercise both sides of each fork with identical data, seeded with special values: ±Inf, NaN, and near-overflow magnitudes (~1e20).
  3. Apply a categorical oracle — compare NaN/Inf placement (not just numeric tolerance) against a float64 brute-force reference or the op's documented
  contract.
  4. Triage every hit against the upstream tracker before claiming it as new.

  Two rounds, ~40 fork checks (agent_space/fork_special_value_hunt.py, fork_special_value_hunt2.py), verified on PyTorch 2.12.0 stable and 2.13.0.dev
  nightly (CPU). Zero false claims: every flagged discrepancy was either confirmed new or attributed to an existing issue.

  New bugs found (no existing upstream report)

  - torch.cdist(p=2) returns NaN instead of Inf on the matmul path. A point with an Inf coordinate yields correct Inf distances when r1 ≤ 25, but NaN once
  r1 > 25 flips the default compute mode onto the mm path (‖a‖² + ‖b‖² − 2a·b → Inf − Inf). The result depends only on row count, not data. Distinct from
  the known finite-precision issues #42479/#123082. Repro: agent_space/cdist_inf_repro.py.
  - F.hardswish(-inf) returns NaN, violating its own documented definition. The docs define hardswish piecewise with output exactly 0 for x ≤ −3, but the
  kernel computes x * relu6(x+3)/6 in the saturated region, so −inf × 0 = NaN. Strongest claim of the set: a documented-contract violation, not just a
  limit argument. Repro: agent_space/activation_inf_repro.py.
  - F.mish(-inf) returns NaN. Mathematical limit of x·tanh(softplus(x)) at −∞ is 0; the kernel produces −inf × 0 = NaN. Same repro script.
  - F.softsign(±inf) returns NaN. x/(1+|x|) evaluates as inf/inf = NaN; the mathematical limits are ±1. Same repro script.

  - Context for the three activation bugs: upstream already accepts this bug class — silu(-inf) is open as #160876, and a gelu Inf fix (PR #185790) is in
  flight — but we verified that PR patches gelu only; mish, hardswish, and softsign are not covered by any issue or pending fix.
  - F.group_norm: contiguous and channels_last CPU kernels disagree categorically. On finite fp32 input of magnitude ~1e20, the contiguous kernel returns
  NaN for the entire group (one-pass E[x²]−mean² accumulation → inf − inf), while the channels_last kernel returns 0.0 on identical data; the fp64
  reference is finite and O(1). The one-pass variance instability is known (#54293), but the memory-format kernel disagreement is undocumented. Repro:
  agent_space/group_norm_layout_repro.py.

  Discrepancies found but attributed to existing reports

  - grid_sample (bilinear, padding_mode='zeros') with an Inf coordinate returns NaN while nearest mode and ordinary out-of-bounds coordinates correctly
  return 0 — covered by umbrella issue #24823.
  - lerp with an Inf endpoint at weight 0/1 returns NaN instead of the exact endpoint — adjacent to triaged issues #78484 and #111374.
  - pdist vs cdist(x,x) disagree on Inf inputs — same root cause as the cdist bug above, not a separate finding.
  - matrix_power (n ≥ 4) NaN placement differs from sequential multiplication — repeated-squaring vs sequential evaluation; recorded as an observation
  only, since no equivalence contract is documented.

  Checks that came back clean (~30)

  addmm/addmv/baddbmm beta=0 Inf contract; all vector_norm ord paths; softmax last-dim vs generic kernel with ±inf; logsumexp all--inf; matmul batch
  folding; conv2d memory formats and the 1×1 fast path; batch_norm layouts; avg_pool divisor fork; searchsorted/bucketize/sort/median NaN handling;
  smooth_l1/huber; softplus threshold fork; interpolate modes; remainder/fmod IEEE semantics.

  Status

  Nothing has been filed upstream yet (pending human review per project policy). Recommended filing plan: the cdist report as a standalone issue (already
  polished), and mish/hardswish/softsign as a single grouped issue citing #160876 and PR #185790 as precedent.
[4:15 PM] Summary statistics

  - Absolute branch coverage: KD 86.0% vs DLL 44.9% of the instrumented surface.
  - Combined branch coverage: 89.3% (union of 159/178); residual uncovered = 10.7%.
  - Coverage efficiency: DLL is ~2× higher per input (6.1 vs 3.1 branches/100 inputs) and ~5× higher per subject (1.25 vs 0.25 branches/op) — its
  specialized generators are statistically more concentrated.
  - Coverage breadth vs depth: KD's higher total coverage is a breadth effect (10× the subject count); normalize by inputs or by subject and DLL has
  the higher coverage rate.
  - Marginal/incremental coverage: DLL contributes only +6 branches (+3.9% relative) over KD, all at conv2d guard predicates — its incremental
  coverage value is small but non-zero.
[4:16 PM] Across the 178 instrumented branches, KD attains higher absolute branch coverage than the DLL generators (153 vs. 80, 86% vs. 45%) purely through
  breadth — exercising ~10× more operators — whereas the DLL generators achieve higher coverage density per subject and per input and uniquely reach 6
  boundary-guarded branches KD's random sampling misses, so KD wins on aggregate coverage while DLL wins on per-API thoroughness.