import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if any index is out-of-bounds with respect to choices tensor, mode must be clip (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] >= 1, (Or(Select(v["arg1_range"], 1) >= Select(v["arg2_shape"], 0), Select(v["arg1_range"], 0) < 0))), v["arg3_value"] == 34, True)) if n else
          If(And(v["arg2_ndim"] >= 1, (Or(Select(v["arg1_range"], 1) >= Select(v["arg2_shape"], 0), Select(v["arg1_range"], 0) < 0))), v["arg3_value"] == 34, True))
)

def rule_28_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Constraints for rule 28
        rule_28(solver, {'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
