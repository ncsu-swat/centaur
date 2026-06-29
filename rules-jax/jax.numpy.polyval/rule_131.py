import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# evaluation point scalar must be within a safe numerical range to avoid underflow/overflow during high-degree polynomial expansion (Rule 131)

rule_131 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 10, And(v["arg1_value"] >= -1000, v["arg1_value"] <= 1000), True)) if n else
          If(Select(v["arg2_shape"], 0) > 10, And(v["arg1_value"] >= -1000, v["arg1_value"] <= 1000), True))
)

def rule_131_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 131
        rule_131(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_131(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
