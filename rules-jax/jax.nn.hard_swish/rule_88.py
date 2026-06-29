import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# input tensor should be a multi-dimensional array with 1 to 4 dimensions (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(And(1 <= v["arg1_ndim"], v["arg1_ndim"] <= 4)) if n else
          And(1 <= v["arg1_ndim"], v["arg1_ndim"] <= 4))
)

def rule_88_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 88
        rule_88(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_ndim': arg1['ndim']}, neg)
