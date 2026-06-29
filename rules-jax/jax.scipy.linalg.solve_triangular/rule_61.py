import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The integer parameter trans must satisfy 0 ≤ trans and trans^2 ≤ 4, restricting its value to 0, 1, or 2 (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(And(0 <= v["arg1_value"], v["arg1_value"] * v["arg1_value"] <= 4)) if n else
          And(0 <= v["arg1_value"], v["arg1_value"] * v["arg1_value"] <= 4))
)

def rule_61_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value']}, neg)
