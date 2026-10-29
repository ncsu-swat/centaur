import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# joint consistency for full tensor signature with scalar discont and period (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_ndim"] == 0, v["arg4_ndim"] == 0), Select(v["arg4_range"], 0) > 0), (If(v["arg3_value"] >= 0, v["arg3_value"] < v["arg1_ndim"], v["arg3_value"] >= 0 - v["arg1_ndim"])))) if n else
          And(And(And(v["arg2_ndim"] == 0, v["arg4_ndim"] == 0), Select(v["arg4_range"], 0) > 0), (If(v["arg3_value"] >= 0, v["arg3_value"] < v["arg1_ndim"], v["arg3_value"] >= 0 - v["arg1_ndim"]))))
)

def rule_44_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 44
        rule_44(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_range': arg4_range, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_range': arg4['range'], 'arg4_ndim': arg4['ndim']}, neg)
