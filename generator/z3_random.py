"""
Random group-selection strategy for Z3-guided API input generation.

Each iteration picks one parameter group uniformly at random as the driver,
fixes all other variables to values sampled from their computed bounds, then
calls solver.check() once.  Z3 only needs to solve for the free driver group,
keeping per-iteration latency near-constant regardless of formula complexity.
"""

import os
import sys
import json
import time
import resource
import shutil

import numpy as np
from z3 import *

from .input_generators import abstract_print
from .definitions import get_definition
from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, int_buckets, float_buckets
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root, bcolors, read_file_in_root
from utils.new_api_utils import get_lib_version, get_api_suffix
from utils.z3_utils import (
    instantiate_args, create_z3_args, initial_constraints, collect_constraints,
    add_negative_buckets, clip_buckets, append_abstract_to_jsonl,
)
from eval.oracle import oracle_crash
from utils.proc import get_memory_usage



# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _peak_rss_mb():
    """OS-level peak RSS high-water mark in MB (Linux: getrusage in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def save_state_models(api, suffix, unsat, nominal, invalid, crash, excp, tmp_results):
    api = f"{api}_{suffix}" if suffix > 0 else api
    total = nominal + invalid + crash + excp
    valid_prcnt = round((total - invalid) * 100 / total, 2) if total > 0 else 0
    csv_file = os.path.join(tmp_results, f"{api}.csv")
    with open(csv_file, "w") as f:
        f.write(f"{api},{unsat},{nominal},{invalid},{crash},{excp},{total},{valid_prcnt}\n")


def is_const_num(expr):
    return is_int_value(expr) or is_rational_value(expr)


def is_nonlinear_expr(expr):
    kind = expr.decl().kind()
    if kind in {Z3_OP_MUL, Z3_OP_DIV, Z3_OP_POWER}:
        if sum(1 for c in expr.children() if not is_const_num(c)) >= 2:
            return True
    return any(is_nonlinear_expr(c) for c in expr.children())


def is_nonlinear_assertion(assertion):
    if assertion.decl().kind() in {Z3_OP_AND, Z3_OP_OR, Z3_OP_IMPLIES}:
        return any(is_nonlinear_assertion(c) for c in assertion.children())
    return is_nonlinear_expr(assertion)


# ---------------------------------------------------------------------------
# Variable collection and graph analysis
# ---------------------------------------------------------------------------

# Variable-name → parameter-group mapping
_FIELD_SUFFIXES = ('_ndim', '_dtype', '_value', '_length')
_ARRAY_INFIX    = ('_shape[', '_range[', '_values[')


def _param_group(var_name: str) -> str:
    """Return the API parameter name that owns this variable."""
    for suffix in _FIELD_SUFFIXES:
        if var_name.endswith(suffix):
            return var_name[: -len(suffix)]
    for pattern in _ARRAY_INFIX:
        idx = var_name.find(pattern)
        if idx != -1:
            return var_name[:idx]
    return var_name


def _collect_var_names(expr) -> dict:
    """Return {str(var): z3_expr} for all scalar and Select(arr, idx) sub-expressions."""
    found = {}

    def walk(e):
        if is_const(e) and e.decl().kind() == Z3_OP_UNINTERPRETED:
            if e.sort().kind() not in (Z3_ARRAY_SORT, Z3_BOOL_SORT):
                found[str(e)] = e
        elif e.decl().kind() == Z3_OP_SELECT:
            arr, idx = e.children()
            if is_const(arr) and is_int_value(idx):
                sel = Select(arr, idx)
                found[str(sel)] = sel
        for ch in e.children():
            walk(ch)

    walk(expr)
    return found


def _flatten_and(assertion):
    """Yield atomic sub-assertions by recursively splitting And() nodes.

    Splitting prevents And(dtype_a < 10, dtype_b < 10) from creating a
    spurious edge between dtype_a and dtype_b in the dependency graph.
    """
    if assertion.decl().kind() == Z3_OP_AND:
        for child in assertion.children():
            yield from _flatten_and(child)
    else:
        yield assertion


def _build_var_adjacency(assertions):
    """Build {var_name: {neighbour_names}} from variables that share an assertion.

    And() is split first so only variables in the *same atomic* sub-assertion
    are considered neighbours.
    """
    all_vars = {}
    adj = {}
    for assertion in assertions:
        for sub in _flatten_and(assertion):
            var_map = _collect_var_names(sub)
            names = list(var_map)
            for name, expr in var_map.items():
                all_vars[name] = expr
                adj.setdefault(name, set())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    adj[names[i]].add(names[j])
                    adj[names[j]].add(names[i])
    return all_vars, adj


def _find_connected_components(adj):
    """Return a list of frozensets, one per BFS connected component."""
    visited = set()
    components = []
    for start in adj:
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(adj[node] - visited)
        components.append(frozenset(comp))
    return components


def _precompute_components(assertions):
    """Identify always-fix variables and cross-parameter groups.

    Returns
    -------
    always_fix : set of variable name strings
        Variables that are not entangled with any other parameter group;
        fixed randomly every iteration.
    cross_comps : list of dicts  {param_group_name: {var_name, ...}}
        Each element is one connected component that spans multiple parameter
        groups.  One group per component acts as the driver (left free for Z3).
    """
    _, adj = _build_var_adjacency(assertions)
    components = _find_connected_components(adj)

    always_fix = set()
    cross_comps = []

    for comp in components:
        groups = {}
        for vname in comp:
            g = _param_group(vname)
            groups.setdefault(g, set()).add(vname)
        if len(groups) <= 1:
            always_fix.update(comp)
        else:
            cross_comps.append(groups)

    return always_fix, cross_comps


# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------

def variable_bounds(assertions, mem_log=None):
    """Compute per-variable (lo, hi) bounds using Z3 Optimize() on linear assertions.

    Fixes vs. the original z3.py implementation:
      - And() is split before variable collection so every variable is found.
      - _shape[ variables: min clamped to 0, max clamped to MAX_SZ_DIM.
        Prevents corrupted Implies-guarded bounds from producing negative
        sizes or shapes that exceed the tensor-element cap in instantiate_args.
    """
    def _vblog(tag):
        if mem_log is not None:
            mem_log.write(f"-1,{tag},{get_memory_usage():.1f},{_peak_rss_mb():.1f}\n")
            mem_log.flush()

    all_vars = {}
    for a in assertions:
        all_vars.update(_collect_var_names(a))

    # Use original (unflattened) assertions for the linear filter so that
    # ITE and other hard sub-terms are still excluded via their parent And().
    linear_assertions = [a for a in assertions if not is_nonlinear_assertion(a)]

    bounds = {}
    for name, var in all_vars.items():
        sort_kind = var.sort().kind()
        buckets = (float_buckets if sort_kind == Z3_REAL_SORT else int_buckets).copy()
        buckets = add_negative_buckets(buckets)

        _vblog(f"vb_before_min_{name}")
        opt_min = Optimize()
        opt_min.add(linear_assertions)
        opt_min.minimize(var)
        if opt_min.check() != sat:
            del opt_min
            continue
        raw = opt_min.model().eval(var, model_completion=True)
        minv = float(raw.as_fraction()) if sort_kind == Z3_REAL_SORT else raw.as_long()
        del opt_min
        _vblog(f"vb_after_min_{name}")

        opt_max = Optimize()
        opt_max.add(linear_assertions)
        opt_max.maximize(var)
        if opt_max.check() != sat:
            del opt_max
            continue
        raw = opt_max.model().eval(var, model_completion=True)
        maxv = float(raw.as_fraction()) if sort_kind == Z3_REAL_SORT else raw.as_long()
        del opt_max
        _vblog(f"vb_after_max_{name}")

        # Shape dimensions are always ≥ 0.  Clamp the upper bound to MAX_SZ_DIM
        # so the tensor-element guard in instantiate_args works correctly.
        if '_shape[' in name:
            minv = max(minv, 0)
            maxv = min(maxv, MAX_SZ_DIM)

        if minv > maxv:
            continue

        val_set = (
            set(sorted(clip_buckets(buckets, minv, maxv)))
            if minv != maxv
            else {minv}
        )
        bounds[var] = (sorted(val_set)[0], sorted(val_set)[-1], sort_kind)

    return bounds


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def gen_models(definition, api, z3_args, model_gen_duration, max_model=0, seed=42,
               print_details=False, lib="torch", corpus_dir=None, return_models=True,
               use_reference=False):

    abstract_inputs_path = os.path.join(corpus_dir, "abstract_inputs.jsonl")
    timestamp_file = os.path.join(corpus_dir, "timestamps.csv")
    with open(timestamp_file, "w") as ft:
        ft.write("model,timestamp\n")

    # Per-run memory log (iter, phase, current_rss_mb, peak_rss_mb)
    mem_log_path = os.path.join(get_tmp_dir(), f"mem_log_{api}.csv")
    _mem_log = open(mem_log_path, "w")
    _mem_log.write("iter,phase,cur_mb,peak_mb\n")
    _mem_log.flush()

    def _mlog(phase, _iter):
        _mem_log.write(f"{_iter},{phase},{get_memory_usage():.1f},{_peak_rss_mb():.1f}\n")
        _mem_log.flush()

    # Build the main solver once; push/pop for per-iteration variable assignments.
    solver_main = Solver()
    initial_constraints(solver_main, definition["signature"], z3_args, lib=lib)
    collect_constraints(solver_main, api, definition["ruleset"], z3_args,
                        use_reference=use_reference, lib=lib)

    smtlib_file = os.path.join(get_tmp_dir(), f"{api}_{definition['suffix']}_formula.smt2")
    with open(smtlib_file, "w") as f:
        f.write(solver_main.to_smt2())

    rng = np.random.default_rng(seed)
    models = []
    num_model = nominal = invalid = crash = excp = unsat = 0
    exceptions = set()
    tmp_results = create_subdir(get_tmp_dir(), "model_results")

    # -----------------------------------------------------------------------
    # One-time pre-computation
    # -----------------------------------------------------------------------
    var_bounds = variable_bounds(solver_main.assertions(), mem_log=_mem_log)
    always_fix, cross_comps = _precompute_components(solver_main.assertions())
    group_lists = [sorted(comp.keys()) for comp in cross_comps]

    if cross_comps:
        print(f"Cross-parameter components: {len(cross_comps)} ({group_lists})")

    # -----------------------------------------------------------------------
    # Generation loop
    # -----------------------------------------------------------------------
    elapsed = 0
    start = time.time()
    _iter = 0

    while elapsed < model_gen_duration and (num_model < max_model or max_model == 0):
        _mlog("loop_start", _iter)

        # Each iteration: pick the driver group uniformly at random.
        dependent_this_iter: set = set()
        for i, comp_groups in enumerate(cross_comps):
            chosen = group_lists[i][int(rng.integers(len(group_lists[i])))]
            for g, names in comp_groups.items():
                if g != chosen:
                    dependent_this_iter.update(names)

        # Push a backtracking point; fix driver group + always_fix, leave
        # dependent group variables free for Z3 to solve.
        solver_main.push()
        for var, (lo, hi, sort_kind) in var_bounds.items():
            if str(var) in dependent_this_iter:
                continue   # leave free — Z3 solves for these given the driver
            if sort_kind == Z3_REAL_SORT:
                v = lo if lo == hi else rng.uniform(lo, hi)
                solver_main.add(var == RealVal(str(v)))
            else:
                v = int(lo) if lo == hi else int(rng.integers(int(lo), int(hi) + 1))
                solver_main.add(var == v)

        result = solver_main.check()
        model = solver_main.model() if result == sat else None
        solver_main.pop()

        if result != sat:
            unsat += 1
            elapsed = time.time() - start
            _iter += 1
            _mlog("after_unsat", _iter)
            save_state_models(definition["api"], definition["suffix"],
                              unsat, nominal, invalid, crash, excp, tmp_results)
            continue

        _mlog("after_sat", _iter)

        concrete_input, abstract_input = instantiate_args(
            model, definition["signature"], z3_args, lib=lib)
        _mlog("after_instantiate", _iter)

        oracle_status, exception_message = oracle_crash(
            api, concrete_input, cpu=True, lib=lib)
        del concrete_input   # release numpy arrays immediately
        _mlog("after_oracle", _iter)

        if oracle_status != "invalid":
            if oracle_status == "cpu_crash":
                crash += 1
            elif oracle_status == "cpu_excp":
                excp += 1
            elif oracle_status == "nominal":
                nominal += 1

            if return_models:
                models.append(model)
            if corpus_dir:
                timestamp = time.time() - start
                with open(timestamp_file, "a") as ft:
                    ft.write(f"{num_model},{timestamp}\n")
                append_abstract_to_jsonl(abstract_input, abstract_inputs_path)
            num_model += 1

            print(f"Valid models: {num_model} | Memory: {get_memory_usage():.2f} MB",
                  end="\r", flush=True)

            if oracle_status != "nominal":
                print(f"\n[{oracle_status}]\n{exception_message}")
                print(f"\nPotential bug. Input:\n"
                      f"{abstract_print(abstract_input, definition['signature'])}")
        else:
            invalid += 1
            exceptions.add(exception_message)
            if print_details:
                print(abstract_print(abstract_input, definition["signature"]))
                print(f"\nStatus: {oracle_status}\n{exception_message}")

        elapsed = time.time() - start
        _iter += 1
        save_state_models(definition["api"], definition["suffix"],
                          unsat, nominal, invalid, crash, excp, tmp_results)

    _mem_log.close()

    exc_dir = create_subdir(get_tmp_dir(), f"exceptions_models_{lib}")
    exc_file = os.path.join(exc_dir, f"{api}_{definition['suffix']}.txt")
    with open(exc_file, "w") as f:
        f.write("\n".join(sorted(exceptions)))

    print(f"\nGenerated {num_model} models for {api} (suffix {definition['suffix']})")
    return models


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_model_gen(variant, duration, n_max, lib, seed, regen, use_reference=False):
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"

    api, suffix = get_api_suffix(variant)
    api = get_lib_version(api, lib=lib)
    definition = get_definition(api, z3=True, lib=lib, suffix=suffix,
                                use_reference=use_reference)

    if len(definition["ruleset"]) == 0:
        known_trivial = read_file_in_root(f"True_invariants_{lib}")
        if variant in known_trivial:
            print(f"No additional invariants for {variant} — using TRUE.")
        else:
            print(f"No invariants learned for {api}, skipping.")
            return
    else:
        print("─" * 60)
        print(f"Rulesets for {api} (suffix {suffix}):")
        for arity, rule_name, *args in definition["ruleset"]:
            print(f"  {rule_name}, arity={arity}, args={args}")
        print("─" * 60)

    corpus_root = "corpus_tf_random" if lib == "tf" else "corpus_torch_random"
    corpus_dir = os.path.join(
        get_dir_in_root(corpus_root),
        f"{api}_{suffix}" if suffix > 0 else api,
    )
    z3_args = create_z3_args(definition["signature"])

    if os.path.exists(os.path.join(corpus_dir, "abstract_inputs.jsonl")) and not regen:
        print("Skipping — existing corpus found. Pass regen=1 to overwrite.")
        return

    if os.path.exists(corpus_dir):
        print(f"Removing existing corpus: {corpus_dir}")
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)

    t0 = time.time()
    gen_models(definition, api, z3_args, duration, max_model=n_max,
               seed=seed, lib=lib, corpus_dir=corpus_dir,
               return_models=False, use_reference=use_reference)
    print(f"{bcolors.OKBLUE}Done in {time.time() - t0:.1f} s{bcolors.ENDC}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m generator.z3_random <variant> <duration> "
              "[n_max] [lib] [seed] [regen] [use_reference]")
        return

    variant       = sys.argv[1]
    duration      = int(sys.argv[2])
    n_max         = int(sys.argv[3])      if len(sys.argv) > 3 else 0
    lib           = sys.argv[4]           if len(sys.argv) > 4 else "torch"
    seed          = int(sys.argv[5])      if len(sys.argv) > 5 else 200
    regen         = int(sys.argv[6]) == 1 if len(sys.argv) > 6 else False
    use_reference = int(sys.argv[7]) == 1 if len(sys.argv) > 7 else False

    run_model_gen(variant, duration, n_max, lib, seed, regen, use_reference)


if __name__ == "__main__":
    main()
