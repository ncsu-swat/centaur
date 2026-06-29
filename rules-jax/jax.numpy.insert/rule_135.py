import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when obj is a tuple of integers and values is a 1-D tensor, their lengths must be compatible (Rule 135)

rule_135 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 1, Or(v["arg1_length"] == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == 1), True)) if n else
          If(v["arg2_ndim"] == 1, Or(v["arg1_length"] == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == 1), True))
)

def rule_135_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 135
        rule_135(solver, {'arg1_length': arg1_length, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_135(solver, {'arg1_length': arg1['length'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
