import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when accuracy is specified, input tensor elements must be non-negative and accuracy mode must not be an activation string (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) >= 0, v["arg2_value"] != 10)) if n else
          And(Select(v["arg1_range"], 0) >= 0, v["arg2_value"] != 10))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 25
        rule_25(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
