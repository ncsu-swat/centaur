import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the function is an activation like relu, sigmoid, or softmax, the input tensor cannot have a boolean or string data type (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 13), v["arg1_value"] == 14), And(v["arg2_dtype"] != 0, v["arg2_dtype"] != 11), True)) if n else
          If(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 13), v["arg1_value"] == 14), And(v["arg2_dtype"] != 0, v["arg2_dtype"] != 11), True))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
