import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# When both arguments are a union of dtype and string, they must represent valid floating-point dtypes for comparison (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 38), v["arg1_value"] == 39)), (Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 38), v["arg2_value"] == 39)))) if n else
          And((Or(Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 38), v["arg1_value"] == 39)), (Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 38), v["arg2_value"] == 39))))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 121
        rule_121(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
