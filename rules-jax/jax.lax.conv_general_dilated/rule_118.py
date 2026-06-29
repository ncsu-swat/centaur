import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# dimension layout specifications must match the input tensor rank (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If(Or(Select(v["arg2_values"], 0) == 16, Select(v["arg2_values"], 0) == 15), v["arg1_ndim"] == 4, If(Or(Select(v["arg2_values"], 0) == 22, Select(v["arg2_values"], 0) == 21), v["arg1_ndim"] == 5, True))) if n else
          If(Or(Select(v["arg2_values"], 0) == 16, Select(v["arg2_values"], 0) == 15), v["arg1_ndim"] == 4, If(Or(Select(v["arg2_values"], 0) == 22, Select(v["arg2_values"], 0) == 21), v["arg1_ndim"] == 5, True)))
)

def rule_118_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_values = Array('arg2_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 118
        rule_118(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values']}, neg)
