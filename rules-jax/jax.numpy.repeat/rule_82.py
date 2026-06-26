import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# validation of tensor repeats, axis, and total_repeat_length (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg4_value"] >= 0, v["arg2_ndim"] == 1), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0), (Or((And(v["arg3_value"] >= 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg3_value"]))), (And(v["arg3_value"] < 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg3_value"] + v["arg1_ndim"]))))))) if n else
          And(And(And(And(And(v["arg4_value"] >= 0, v["arg2_ndim"] == 1), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0), (Or((And(v["arg3_value"] >= 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg3_value"]))), (And(v["arg3_value"] < 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg3_value"] + v["arg1_ndim"])))))))
)

def rule_82_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 82
        rule_82(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
