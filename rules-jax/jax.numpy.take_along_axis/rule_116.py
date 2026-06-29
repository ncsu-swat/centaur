import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# type compatibility when fill_value is float (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_value"] == v["arg3_value"], (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)))) if n else
          And(And(v["arg3_value"] == v["arg3_value"], (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), (And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8))))
)

def rule_116_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
