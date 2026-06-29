import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# signature must be one of the recognized vectorization or einsum-like patterns (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(v["arg1_value"] == 5, v["arg1_value"] == 0), v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4)) if n else
          Or(Or(Or(Or(Or(v["arg1_value"] == 5, v["arg1_value"] == 0), v["arg1_value"] == 1), v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4))
)

def rule_107_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 107
        rule_107(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_value': arg1['value']}, neg)
