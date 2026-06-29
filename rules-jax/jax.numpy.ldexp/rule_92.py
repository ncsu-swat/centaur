import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# real-valued input constraint and exponent range limit when x2 is an integer (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(And(And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg2_value"] >= -1024), v["arg2_value"] <= 1024)) if n else
          And(And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg2_value"] >= -1024), v["arg2_value"] <= 1024))
)

def rule_92_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 92
        rule_92(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
