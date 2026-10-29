import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# matrix norm with tensor, keepdims and ord parameters (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, And(v["arg1_ndim"] >= 2, v["arg3_value"] != 0), And(v["arg1_ndim"] >= 2, (Or(v["arg3_value"] * v["arg3_value"] == 1, v["arg3_value"] * v["arg3_value"] == 4))))) if n else
          If(v["arg2_value"] == True, And(v["arg1_ndim"] >= 2, v["arg3_value"] != 0), And(v["arg1_ndim"] >= 2, (Or(v["arg3_value"] * v["arg3_value"] == 1, v["arg3_value"] * v["arg3_value"] == 4)))))
)

def rule_25_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 25
        rule_25(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
