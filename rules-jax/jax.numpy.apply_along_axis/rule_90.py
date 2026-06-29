import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The func1d string parameter must not be an einsum subscript pattern (Rule 90)

rule_90 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2), v["arg1_value"] != 3), v["arg1_value"] != 4)) if n else
          And(And(And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2), v["arg1_value"] != 3), v["arg1_value"] != 4))
)

def rule_90_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 90
        rule_90(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_90(solver, {'arg1_value': arg1['value']}, neg)
