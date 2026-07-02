# Model Generation Strategies

Two strategies are available for generating abstract inputs from an SMT formula.
Both read the same API signature and inferred invariants and write the same
`abstract_inputs.jsonl` format — the only difference is _how_ the solver is
driven to produce diverse assignments.

---

## `z3.py` — Constraint-guided generation (default)

### What it does

Builds a Z3 formula from the API's inferred invariants and drives Z3 toward
diverse solutions using three layered strategies applied on every iteration:

1. **Boolean partitioning** (`parition_solvers`) — if the signature contains
   boolean parameters, the solver is cloned once per truth-value combination,
   guaranteeing coverage of all boolean corners before repeating.

2. **Probabilistic blocking** — every satisfying assignment is recorded; a
   random `blocking_proba` fraction of those past assignments is added as
   `≠` constraints on the next call, nudging Z3 away from regions already
   explored.  When too many consecutive UNSAT results are detected (staleness),
   `blocking_proba` is reduced adaptively so the search can escape.

3. **Partition-based steering** — variable bounds are computed via
   `variable_bounds` and partitioned into buckets.  On each iteration a random
   `partition_proba` fraction of variables is steered to a value drawn from a
   gap between two previously-seen bucket values, guiding Z3 into unexplored
   sub-intervals.

The result is a corpus with high validity rate (almost all samples pass the
oracle) and good diversity, at the cost of maintaining solver state and running
multiple Z3 queries per sample.

### Corpus location

```
corpus_torch/<api>/abstract_inputs.jsonl
corpus_torch/<api>/timestamps.csv
```

### How to run — single API

```bash
python -m generator.z3 <variant> <duration_s> [n_max] [lib] [seed] [regen] [use_reference]
```

| Argument | Default | Meaning |
|---|---|---|
| `variant` | — | API variant string, e.g. `torch.conv2d` |
| `duration_s` | — | Wall-clock budget in seconds |
| `n_max` | `0` | Stop after this many valid models (0 = unlimited) |
| `lib` | `torch` | `torch` or `tf` |
| `seed` | `200` | NumPy RNG seed |
| `regen` | `0` | `1` to overwrite an existing corpus |
| `use_reference` | `0` | `1` to use reference invariants |

```bash
# Examples
python -m generator.z3 torch.conv2d 300
python -m generator.z3 torch.nn.functional.relu 600 0 torch 42 1
```

### How to run — all APIs via Slurm

```bash
bash scripts/generate_models_with_slurm.sh [duration] [n_max] [lib] [seed] [regen]

# Examples
bash scripts/generate_models_with_slurm.sh 300             # 5 min budget, torch
bash scripts/generate_models_with_slurm.sh 3600 0 torch 42 0   # 1 h budget

# To run on a custom subset of APIs:
export elements_file=my_apis.txt
bash scripts/generate_models_with_slurm.sh 300
```

The script reads API variants from `${lib}_variations.txt` by default and
submits one Slurm job per variant.  Results are aggregated into:

```
.tmp/model_generation_torch.csv
```

---

## `z3_random.py` — Random group-selection sampling

### What it does

Rather than fixing every variable independently, this strategy reasons about
which variables are **linked by cross-parameter constraints** and uses Z3 to
satisfy those constraints — not just verify them.

#### Step 1: build the variable dependency graph

Every variable that appears in any assertion becomes a node. Two variables share
an edge if they co-occur in the **same atomic sub-assertion**.

Before building edges, every top-level assertion is recursively split on `And()`
nodes (`_flatten_and`). This is a critical correctness fix: without splitting,
a compound assertion like

```
And(input_dtype < 10, other_dtype < 10)
```

would create a spurious edge between `input_dtype` and `other_dtype`, falsely
placing them in the same connected component and treating their two independent
single-parameter constraints as a cross-parameter dependency. After splitting,
each child assertion is considered independently, so the edge only appears if
the two variables genuinely share an atomic constraint.

Variable collection is also done via `_collect_var_names`, which walks the
expression tree and captures both plain scalar constants (e.g. `input_ndim`,
`input_dtype`) and array element accesses expressed as Z3 `Select` nodes
(e.g. `Select(input_shape, 0)` → `input_shape[0]`).

For example, the broadcasting invariant
`input_shape[j] == other_shape[i] OR other_shape[i] == 1`
is a single atomic assertion (an `Or`, not an `And`), so it links
`input_shape[j]` to `other_shape[i]`, correctly placing them in the same
connected component.

#### Step 2: group variables by API parameter

Within each connected component, variables are grouped by the API parameter they
belong to. The group is inferred from the variable name:

| Variable name | Parameter group |
|---|---|
| `input_ndim` | `input` |
| `input_shape[0]`, `input_shape[1]`, … | `input` |
| `other_ndim` | `other` |
| `other_shape[0]`, … | `other` |
| `kernel_size_value` | `kernel_size` |

#### Step 3: classify components

- **Single-group component** — all variables belong to the same API parameter
  (e.g. `input_ndim` and `input_shape[k]` are linked only by the per-tensor
  bounds constraint). These variables are always fixed randomly, the same as
  plain random sampling.
- **Multi-group component** — variables from two or more parameters are linked
  by a cross-parameter constraint (e.g. broadcasting links `input` and `other`).
  One parameter group is randomly selected as the **driver** for this iteration;
  Z3 is left to find values for all remaining groups.

#### Step 4: each iteration

For each multi-group component a driver group is drawn uniformly at random.
The driver's variables are fixed to uniform random values from their bounds.
All other variables in the component are left unconstrained so Z3 can find
compatible values. For APIs with no cross-parameter constraints the behaviour
is identical to fixing every variable randomly.

#### Why this helps

Consider `torch.le` (two tensors that must satisfy broadcasting):

- **Old approach (fix all randomly):** both tensor shapes are independently
  drawn. The probability that a random `(input_shape, other_shape)` pair
  satisfies broadcasting is roughly 1 in 100. 99% of Z3 calls are wasted
  confirming UNSAT.
- **New approach (fix one group, solve the other):** `input_shape` and
  `input_ndim` are fixed randomly. Z3 is asked to find `other_shape` and
  `other_ndim` that broadcast with the fixed input. A valid other tensor always
  exists (e.g. a scalar broadcasts with everything), so the SAT rate is 100%
  and diverse inputs are produced at 347 unique models per second instead of 9.

#### Why the driver is chosen at random each iteration

The broadcasting constraint is symmetric — there is no reason to always fix
`input` rather than `other`. Randomly alternating the driver produces a corpus
that spans both directions. For APIs where one parameter group is the natural
anchor (e.g. `input` drives `out` in element-wise APIs), that group will tend
to be chosen half the time and contributes the majority of the useful models.
The other iterations are not wasted — they still produce valid models, just
with lower diversity for that component.

**Trade-off** — high throughput and 100% SAT rate for cross-parameter APIs.
For very complex constraints (non-linear arithmetic, tight divisibility), Z3
may still take many seconds per solve call; in those cases the SAT rate
advantage is reduced.

### Corpus location

Writes to a **separate** directory to avoid overwriting the z3 corpus:

```
corpus_torch_random/<api>/abstract_inputs.jsonl
corpus_torch_random/<api>/timestamps.csv
```

### How to run — single API

Arguments are identical to `z3.py`:

```bash
python -m generator.z3_random <variant> <duration_s> [n_max] [lib] [seed] [regen] [use_reference]

# Example
python -m generator.z3_random torch.conv2d 300
```

### How to run — all APIs via Slurm

```bash
bash scripts/generate_models_random_with_slurm.sh [duration] [n_max] [lib] [seed] [regen]

# Examples
bash scripts/generate_models_random_with_slurm.sh 300            # 5 min budget, torch
bash scripts/generate_models_random_with_slurm.sh 3600 0 tf 42   # TensorFlow, 1 h

# To run on a custom subset of APIs:
export elements_file=my_apis.txt
bash scripts/generate_models_random_with_slurm.sh 300
```

Results are aggregated into:

```
.tmp/model_generation_random_torch.csv
```

---

## Comparing strategies with `compare_corpora.py`

The root-level `compare_corpora.py` measures three complementary diversity
metrics and produces plots for one or many APIs at once.

### Metrics

| Metric | Definition |
|---|---|
| **unique ratio** | Fraction of distinct samples (exact JSON match). Low values mean the sampler revisits the same assignments. |
| **mean pairwise distance** | Mean normalised Manhattan distance over random pairs. Each dimension is normalised by its observed range so wide integer axes don't dominate. |
| **mean feature entropy** | Mean Shannon entropy (bits) across all numeric features extracted from the abstract inputs. Higher = more spread. |

---

### Single-API comparison

Pass the two `abstract_inputs.jsonl` files directly:

```bash
python compare_corpora.py \
    corpus_torch/torch.conv2d/abstract_inputs.jsonl \
    corpus_torch_random/torch.conv2d/abstract_inputs.jsonl \
    --label1 z3 --label2 random \
    --out comparison_results/torch.conv2d
```

**Outputs** in `--out/`:

| File | Contents |
|---|---|
| `z3_vs_random_scatter.png` | Side-by-side PCA scatter (PCA fitted on the union). Each panel subtitle shows all three metrics. |
| `z3_vs_random_entropy.png` | Bar chart of per-feature Shannon entropy. |
| `z3_vs_random_report.txt` | Plain-text metrics and deltas. |

---

### Multi-API comparison

Pass the two corpus **root directories** instead of individual files.

```bash
# Compare all APIs present in both roots
python compare_corpora.py corpus_torch corpus_torch_random \
    --label1 z3 --label2 random \
    --out comparison_results/

# Filter to a specific set of APIs
python compare_corpora.py corpus_torch corpus_torch_random \
    --label1 z3 --label2 random \
    --apis-file torch_variations.txt \
    --out comparison_results/
```

When `--apis-file` is omitted the script auto-discovers every API
subdirectory present in **both** roots.

**Per-API outputs** in `--out/<api>/`:
- `z3_vs_random_scatter.png`
- `z3_vs_random_entropy.png`
- `z3_vs_random_report.txt`

**Aggregate outputs** in `--out/`:

| File | Contents |
|---|---|
| `z3_vs_random_summary.csv` | One row per API with all six metric values. |
| `z3_vs_random_summary.txt` | Human-readable summary with per-API deltas. |
| `z3_vs_random_unique_ratio.png` | Grouped bar chart of unique ratio across APIs. |
| `z3_vs_random_mpd.png` | Grouped bar chart of mean pairwise distance. |
| `z3_vs_random_entropy.png` | Grouped bar chart of mean feature entropy. |

A summary table is also printed to stdout:

```
            API                    n(z3)  n(rand)  uniq(z3)  uniq(rand)   mpd(z3)  mpd(rand)   H(z3) b  H(rand) b
------------------------------------  -------  -------  --------  ----------  --------  ---------  --------  ---------
torch.conv2d                              347      500    0.9160      1.0000    0.3120     0.4810    3.2140     4.1020
torch.nn.functional.relu                  412      500    0.9800      1.0000    0.2870     0.4120    2.9870     3.8850
...
AVERAGE                                   ...      ...    0.9400      0.9970    0.3010     0.4510    3.1000     4.0100
```

---

### Interpreting results

- **Higher unique ratio** in z3_random is expected — pure random sampling
  never repeats the same point unless the feasible region is very small.
- **Higher mean pairwise distance and entropy** confirm inputs are more spread
  across the parameter space.
- **Lower hit rate** (logged during generation) in z3_random means more wasted
  attempts for tightly constrained APIs — the solver is not guiding samples
  into feasible regions.
- For fuzzing purposes, the relevant question is which corpus finds more bugs.
  Use `pipeline.sh` / `harness_z3.py` with each corpus directory to measure
  crash / exception counts.
