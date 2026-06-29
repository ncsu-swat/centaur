import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# output tensor shape and inexact dtype constraints for tuple input (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == v["arg1_length"]), v["arg2_dtype"] >= 6), v["arg2_dtype"] <= 8)) if n else
          And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == v["arg1_length"]), v["arg2_dtype"] >= 6), v["arg2_dtype"] <= 8))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)) or (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 79
        rule_79(solver, {'arg1_length': arg1_length, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_length': arg1['length'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
