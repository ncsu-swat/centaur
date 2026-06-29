import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Input tensors must have dimensions between 1 and 4 to prevent high-dimensional complexity (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] >= 1, v["arg1_ndim"] <= 4), v["arg2_ndim"] >= 1), v["arg2_ndim"] <= 4)) if n else
          And(And(And(v["arg1_ndim"] >= 1, v["arg1_ndim"] <= 4), v["arg2_ndim"] >= 1), v["arg2_ndim"] <= 4))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 78
        rule_78(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
