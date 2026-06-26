import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if supertype is bfloat16, subtype must be floating-point or bfloat16/float32 (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 39, Or(Or((And(6 <= v["arg1_value"], v["arg1_value"] <= 8)), v["arg1_value"] == 39), v["arg1_value"] == 38), True)) if n else
          If(v["arg2_value"] == 39, Or(Or((And(6 <= v["arg1_value"], v["arg1_value"] <= 8)), v["arg1_value"] == 39), v["arg1_value"] == 38), True))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)) or isinstance(arg1, str)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
