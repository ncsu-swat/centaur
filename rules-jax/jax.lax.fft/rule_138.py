import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Valid FFT types and minimum rank constraint on the input tensor (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] >= 1, (Or(Or(Or(v["arg2_value"] == 29, v["arg2_value"] == 30), v["arg2_value"] == 31), v["arg2_value"] == 32)))) if n else
          And(v["arg1_ndim"] >= 1, (Or(Or(Or(v["arg2_value"] == 29, v["arg2_value"] == 30), v["arg2_value"] == 31), v["arg2_value"] == 32))))
)

def rule_138_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 138
        rule_138(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
