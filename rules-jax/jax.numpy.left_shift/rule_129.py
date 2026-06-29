import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the shift amount y is an integer, it must be non-negative, and the input tensor x must be integer-typed (Rule 129)

rule_129 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_value"] >= 0, 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5)) if n else
          And(And(v["arg2_value"] >= 0, 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5))
)

def rule_129_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 129
        rule_129(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_129(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
