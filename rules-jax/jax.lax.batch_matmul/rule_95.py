import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# precision string must be a valid JAX precision option (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 37, True, (If(v["arg1_value"] == 38, True, v["arg1_value"] == 39)))) if n else
          If(v["arg1_value"] == 37, True, (If(v["arg1_value"] == 38, True, v["arg1_value"] == 39))))
)

def rule_95_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 95
        rule_95(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_value': arg1['value']}, neg)
