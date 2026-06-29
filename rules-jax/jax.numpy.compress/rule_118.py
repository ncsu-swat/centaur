import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# constraints on condition, input tensor, axis, static size, and tensor fill_value (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] >= 1), -1 * v["arg2_ndim"] <= v["arg3_value"]), v["arg3_value"] <= v["arg2_ndim"] - 1), v["arg4_value"] >= 0), v["arg5_ndim"] == 0)) if n else
          And(And(And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] >= 1), -1 * v["arg2_ndim"] <= v["arg3_value"]), v["arg3_value"] <= v["arg2_ndim"] - 1), v["arg4_value"] >= 0), v["arg5_ndim"] == 0))
)

def rule_118_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 118
        rule_118(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_ndim': arg5['ndim']}, neg)
