import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Padding string must be valid, same, or SAME_LOWER regardless of consistent padding setting (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg2_value"] == True, v["arg2_value"] == False)), (Or(Or(v["arg1_value"] == 19, v["arg1_value"] == 20), v["arg1_value"] == 28)))) if n else
          And((Or(v["arg2_value"] == True, v["arg2_value"] == False)), (Or(Or(v["arg1_value"] == 19, v["arg1_value"] == 20), v["arg1_value"] == 28))))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
