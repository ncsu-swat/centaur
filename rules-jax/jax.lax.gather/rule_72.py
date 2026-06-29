import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if mode is fill and fill_value is boolean, operand must have boolean dtype (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 36, And(v["arg1_dtype"] == 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), Or(v["arg2_value"] == True, v["arg2_value"] == False))) if n else
          If(v["arg3_value"] == 36, And(v["arg1_dtype"] == 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), Or(v["arg2_value"] == True, v["arg2_value"] == False)))
)

def rule_72_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Constraints for rule 72
        rule_72(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
