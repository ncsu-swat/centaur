import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if fill value is an integer, it must be within the bounds of 8-bit types if the input tensor is int8 or uint8 (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, And(v["arg2_value"] >= -128, v["arg2_value"] <= 127), If(v["arg1_dtype"] == 5, And(v["arg2_value"] >= 0, v["arg2_value"] <= 255), True))) if n else
          If(v["arg1_dtype"] == 1, And(v["arg2_value"] >= -128, v["arg2_value"] <= 127), If(v["arg1_dtype"] == 5, And(v["arg2_value"] >= 0, v["arg2_value"] <= 255), True)))
)

def rule_113_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 113
        rule_113(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
