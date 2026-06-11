import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Data format/layout signature implies 4-dimensional input tensor (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == 15, v["arg1_value"] == 16), v["arg2_ndim"] == 4, True)) if n else
          If(Or(v["arg1_value"] == 15, v["arg1_value"] == 16), v["arg2_ndim"] == 4, True))
)

def rule_13_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
