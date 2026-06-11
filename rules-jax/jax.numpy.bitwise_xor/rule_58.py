import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# first operand of union type int or bool must be valid and tensor operand must be integer or boolean (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(And((And(v["arg2_dtype"] >= 0, v["arg2_dtype"] <= 5)), (Or((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 4294967295)))))) if n else
          And((And(v["arg2_dtype"] >= 0, v["arg2_dtype"] <= 5)), (Or((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 4294967295))))))
)

def rule_58_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
