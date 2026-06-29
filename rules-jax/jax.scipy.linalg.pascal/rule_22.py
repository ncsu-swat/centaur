import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# size of the Pascal matrix must be positive and kind cannot be sum or mean (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 0, v["arg2_value"] != 6), v["arg2_value"] != 9)) if n else
          And(And(v["arg1_value"] > 0, v["arg2_value"] != 6), v["arg2_value"] != 9))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
