import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# equivalence comparison of union types based on casting constraint (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 20, v["arg1_value"] == v["arg2_value"], v["arg1_value"] != v["arg2_value"])) if n else
          If(v["arg3_value"] == 20, v["arg1_value"] == v["arg2_value"], v["arg1_value"] != v["arg2_value"]))
)

def rule_14_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)) or isinstance(arg1, str)):
            return False
        if not ((isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)) or isinstance(arg2, str)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Value assignments
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
