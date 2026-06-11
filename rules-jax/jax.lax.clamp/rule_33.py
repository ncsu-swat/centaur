import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for numeric bounds, min must be less than or equal to max, and the input tensor must have a compatible numeric data type (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] <= v["arg3_value"], 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8)) if n else
          And(And(v["arg1_value"] <= v["arg3_value"], 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 8))
)

def rule_33_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 33
        rule_33(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
