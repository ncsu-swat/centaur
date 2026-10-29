import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Signature: arr: tensor, obj: tensor, values: tensor, axis: int - values rank cannot exceed arr rank and obj indices must not exceed axis dimension size (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(And(0 <= v["arg4_value"], v["arg4_value"] < v["arg1_ndim"]), And(v["arg3_ndim"] <= v["arg1_ndim"], Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg4_value"])), v["arg3_ndim"] <= v["arg1_ndim"])) if n else
          If(And(0 <= v["arg4_value"], v["arg4_value"] < v["arg1_ndim"]), And(v["arg3_ndim"] <= v["arg1_ndim"], Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg4_value"])), v["arg3_ndim"] <= v["arg1_ndim"]))
)

def rule_22_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 22
        rule_22(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
