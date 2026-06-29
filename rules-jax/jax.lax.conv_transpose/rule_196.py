import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# channel dimension alignment for NCHW and OIHW layout conventions (Rule 196)

rule_196 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg3_values"], 0) == 16, Select(v["arg3_values"], 1) == 18), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), True)) if n else
          If(And(Select(v["arg3_values"], 0) == 16, Select(v["arg3_values"], 1) == 18), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), True))
)

def rule_196_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all(isinstance(e, str) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), StringSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 196
        rule_196(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_196(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values']}, neg)
