import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# tensor dtype categorized into integer, floating-point, or other valid types (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg1_ndim"] >= 0, If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg1_ndim"] >= 0, v["arg1_dtype"] <= 10))) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg1_ndim"] >= 0, If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg1_ndim"] >= 0, v["arg1_dtype"] <= 10)))
)

def rule_45_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']}, neg)
