import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If mode is a statistical mode, the integer stat_length must not exceed any of the array's dimension sizes (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg2_value"] == 9, v["arg2_value"] == 8), v["arg2_value"] == 7), And([Implies(i < (v["arg1_ndim"] - 1 + 1), v["arg3_value"] <= Select(v["arg1_shape"], i)) for i in range(6)]), True)) if n else
          If(Or(Or(v["arg2_value"] == 9, v["arg2_value"] == 8), v["arg2_value"] == 7), And([Implies(i < (v["arg1_ndim"] - 1 + 1), v["arg3_value"] <= Select(v["arg1_shape"], i)) for i in range(6)]), True))
)

def rule_53_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 53
        rule_53(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
