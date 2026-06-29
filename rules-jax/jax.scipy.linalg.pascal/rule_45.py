import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# size parameter must be positive and kind must not be "clip" or "drop" (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 1, False, And(v["arg2_value"] != 34, v["arg2_value"] != 35))) if n else
          If(v["arg1_value"] < 1, False, And(v["arg2_value"] != 34, v["arg2_value"] != 35)))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
