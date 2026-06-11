import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# 1D integer shape with valid string dtype and sharding compatibility (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] >= 0, (Or(v["arg2_value"] == 38, v["arg2_value"] == 39))), v["arg3_length"] == 1)) if n else
          And(And(v["arg1_value"] >= 0, (Or(v["arg2_value"] == 38, v["arg2_value"] == 39))), v["arg3_length"] == 1))
)

def rule_53_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, tuple) and all(isinstance(e, str) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 53
        rule_53(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
