import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# a tensor and scalar union input where the tensor is numeric and the scalar is within a standard numeric range (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(v["arg2_value"] > -1000000, v["arg2_value"] < 1000000)))) if n else
          And((And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 8)), (And(v["arg2_value"] > -1000000, v["arg2_value"] < 1000000))))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 43
        rule_43(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
