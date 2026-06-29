import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If input is a string, converting to a JAX array always requires a copy, so copy must be true (Rule 139)

rule_139 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, And(v["arg1_value"] == 5, v["arg1_value"] != 5), True)) if n else
          If(v["arg2_value"] == False, And(v["arg1_value"] == 5, v["arg1_value"] != 5), True))
)

def rule_139_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 139
        rule_139(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_139(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
