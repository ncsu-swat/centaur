import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# both arguments are strings representing valid dtypes (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 38, Or(v["arg2_value"] == 38, v["arg2_value"] == 39), And(v["arg1_value"] == 39, (Or(v["arg2_value"] == 38, v["arg2_value"] == 39))))) if n else
          If(v["arg1_value"] == 38, Or(v["arg2_value"] == 38, v["arg2_value"] == 39), And(v["arg1_value"] == 39, (Or(v["arg2_value"] == 38, v["arg2_value"] == 39)))))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 73
        rule_73(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
