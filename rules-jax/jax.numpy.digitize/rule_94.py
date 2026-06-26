import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# method must be a valid option for searchsorted (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == 34, v["arg1_value"] == 36), v["arg1_value"] == 33), v["arg1_value"] == 5)) if n else
          Or(Or(Or(v["arg1_value"] == 34, v["arg1_value"] == 36), v["arg1_value"] == 33), v["arg1_value"] == 5))
)

def rule_94_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))

        # Constraints for rule 94
        rule_94(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_value': arg1['value']}, neg)
