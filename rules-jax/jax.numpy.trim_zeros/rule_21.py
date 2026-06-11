import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# trim argument must not be any of the unsupported standard string values like none, sum, or mean (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] != 5, v["arg1_value"] != 6), v["arg1_value"] != 9)) if n else
          And(And(v["arg1_value"] != 5, v["arg1_value"] != 6), v["arg1_value"] != 9))
)

def rule_21_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value']}, neg)
