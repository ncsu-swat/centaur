import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when x is an integer and y is a tensor, x must be a valid 32-bit signed integer, y must be integer-typed, and its values must be non-negative (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] > -2147483648, v["arg1_value"] < 2147483648), v["arg2_dtype"] >= 1), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0)) if n else
          And(And(And(And(v["arg1_value"] > -2147483648, v["arg1_value"] < 2147483648), v["arg2_dtype"] >= 1), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
