import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# String-based dtype casting with copy option (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 38, And(v["arg1_ndim"] >= 1, Or(v["arg3_value"] == True, v["arg3_value"] == False)), And(v["arg1_ndim"] >= 0, v["arg2_value"] == 39))) if n else
          If(v["arg2_value"] == 38, And(v["arg1_ndim"] >= 1, Or(v["arg3_value"] == True, v["arg3_value"] == False)), And(v["arg1_ndim"] >= 0, v["arg2_value"] == 39)))
)

def rule_72_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 72
        rule_72(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
