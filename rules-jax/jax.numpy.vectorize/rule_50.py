import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Layout format strings are not valid signatures for vectorize (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] != 15, v["arg1_value"] != 16), v["arg1_value"] != 21), v["arg1_value"] != 22)) if n else
          And(And(And(v["arg1_value"] != 15, v["arg1_value"] != 16), v["arg1_value"] != 21), v["arg1_value"] != 22))
)

def rule_50_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value']}, neg)
