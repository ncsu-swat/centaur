import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# In promise_in_bounds mode, single integer index must be strictly in bounds to avoid index errors (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == 33, (If(v["arg3_value"] >= 0, And(v["arg2_value"] >= 0, v["arg2_value"] < Select(v["arg1_shape"], v["arg3_value"])), And(v["arg2_value"] >= 0, v["arg2_value"] < Select(v["arg1_shape"], v["arg3_value"] + v["arg1_ndim"])))), True)) if n else
          If(v["arg4_value"] == 33, (If(v["arg3_value"] >= 0, And(v["arg2_value"] >= 0, v["arg2_value"] < Select(v["arg1_shape"], v["arg3_value"])), And(v["arg2_value"] >= 0, v["arg2_value"] < Select(v["arg1_shape"], v["arg3_value"] + v["arg1_ndim"])))), True))
)

def rule_41_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        solver.add(arg4_value == list_of_string_values_jax.index(arg4))

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_string_values_jax.index(arg4))

        # Constraints for rule 41
        rule_41(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
