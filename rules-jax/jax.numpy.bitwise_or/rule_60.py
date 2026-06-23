import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If v_1 is an integer and v_2 is a tensor, v_2 must be of boolean or integer type and v_1 must be a valid integer (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_dtype"] <= 5, (And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 2147483647)))) if n else
          And(v["arg2_dtype"] <= 5, (And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 2147483647))))
)

def rule_60_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
