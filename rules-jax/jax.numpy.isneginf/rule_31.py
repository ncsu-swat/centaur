import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for scalar input, the output tensor must be 0-dimensional (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_ndim"] == 0, v["arg1_value"] - v["arg1_value"] == 0)) if n else
          And(v["arg2_ndim"] == 0, v["arg1_value"] - v["arg1_value"] == 0))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
