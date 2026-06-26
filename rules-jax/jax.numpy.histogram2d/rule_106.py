import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# weights tensor must be numeric and have the same shape as x (Rule 106)

rule_106 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 10)) if n else
          And(And(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 10))
)

def rule_106_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 106
        rule_106(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_106(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype']}, neg)
