import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The matrix order n must be strictly positive, and kind cannot be "none" (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] > 0.0, v["arg2_value"] != 5)) if n else
          And(v["arg1_value"] > 0.0, v["arg2_value"] != 5))
)

def rule_32_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
