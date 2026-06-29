import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# valid float/complex input tensor paired with a tuple of absolute and relative tolerances (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), v["arg2_length"] == 2)) if n else
          And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), v["arg2_length"] == 2))
)

def rule_94_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 94
        rule_94(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length']}, neg)
