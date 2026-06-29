import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the input is an integer, it should reside within the range [-1, 1] to ensure the output is real (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 1, False, (If(v["arg1_value"] < -1, False, True)))) if n else
          If(v["arg1_value"] > 1, False, (If(v["arg1_value"] < -1, False, True))))
)

def rule_118_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 118
        rule_118(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_value': arg1['value']}, neg)
