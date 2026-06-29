import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# kind is a union of dtype and string (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 12), (If(Or(v["arg2_value"] == 38, v["arg2_value"] == 39), True, And(v["arg2_value"] >= 0, v["arg2_value"] <= 12))))) if n else
          And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 12), (If(Or(v["arg2_value"] == 38, v["arg2_value"] == 39), True, And(v["arg2_value"] >= 0, v["arg2_value"] <= 12)))))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)):
            return False
        if not ((isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
