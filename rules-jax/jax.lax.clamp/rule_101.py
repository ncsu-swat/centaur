import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# min value must be less than or equal to max value, and input tensor must be non-scalar (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] <= v["arg3_value"], v["arg2_ndim"] >= 1)) if n else
          And(v["arg1_value"] <= v["arg3_value"], v["arg2_ndim"] >= 1))
)

def rule_101_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 101
        rule_101(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
