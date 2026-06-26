import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# accuracy string should specify a valid accuracy mode and input tensor must have floating-point or complex dtype (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), (Or(Or(v["arg2_value"] == 37, v["arg2_value"] == 38), v["arg2_value"] == 39)))) if n else
          And((And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)), (Or(Or(v["arg2_value"] == 37, v["arg2_value"] == 38), v["arg2_value"] == 39))))
)

def rule_53_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 53
        rule_53(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
