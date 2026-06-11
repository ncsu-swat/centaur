import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The order parameter must be "none" if the tensor is non-empty (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] >= 1, v["arg2_value"] == 5, v["arg2_value"] == 5)) if n else
          If(v["arg1_ndim"] >= 1, v["arg2_value"] == 5, v["arg2_value"] == 5))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 26
        rule_26(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
