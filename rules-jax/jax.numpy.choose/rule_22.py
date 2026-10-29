import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when mode is clip choices tensor needs at least one choice, otherwise indices must be strictly within bounds (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 34, Select(v["arg2_shape"], 0) >= 1, And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) < Select(v["arg2_shape"], 0)))) if n else
          If(v["arg3_value"] == 34, Select(v["arg2_shape"], 0) >= 1, And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) < Select(v["arg2_shape"], 0))))
)

def rule_22_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Constraints for rule 22
        rule_22(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
