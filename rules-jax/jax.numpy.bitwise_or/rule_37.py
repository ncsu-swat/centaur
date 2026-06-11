import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Type compatibility for scalar integer/boolean and tensor inputs (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(And(And(0 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), (Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] > -2147483648)))) if n else
          And(And(0 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), (Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] > -2147483648))))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 37
        rule_37(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
