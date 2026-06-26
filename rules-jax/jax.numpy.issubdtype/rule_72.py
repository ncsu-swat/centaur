import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# parameters must be valid dtypes or dtype strings (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or((And(0 <= v["arg1_value"], v["arg1_value"] <= 12)), v["arg1_value"] == 38), v["arg1_value"] == 39)), (Or(Or((And(0 <= v["arg2_value"], v["arg2_value"] <= 12)), v["arg2_value"] == 38), v["arg2_value"] == 39)))) if n else
          And((Or(Or((And(0 <= v["arg1_value"], v["arg1_value"] <= 12)), v["arg1_value"] == 38), v["arg1_value"] == 39)), (Or(Or((And(0 <= v["arg2_value"], v["arg2_value"] <= 12)), v["arg2_value"] == 38), v["arg2_value"] == 39))))
)

def rule_72_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)) or isinstance(arg1, str)):
            return False
        if not ((isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
