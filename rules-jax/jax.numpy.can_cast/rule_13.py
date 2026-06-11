import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# dimension and destination type constraints when casting is same (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 20, And(v["arg1_ndim"] >= 0, v["arg2_value"] == 20), And(v["arg1_ndim"] >= 0, v["arg2_value"] != 20))) if n else
          If(v["arg3_value"] == 20, And(v["arg1_ndim"] >= 0, v["arg2_value"] == 20), And(v["arg1_ndim"] >= 0, v["arg2_value"] != 20)))
)

def rule_13_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))

        # Constraints for rule 13
        rule_13(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
