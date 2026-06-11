import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# 3D image layouts require 5-dimensional input tensor (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 21, v["arg2_value"] == 22), v["arg1_ndim"] == 5, True)) if n else
          If(Or(v["arg2_value"] == 21, v["arg2_value"] == 22), v["arg1_ndim"] == 5, True))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 25
        rule_25(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
