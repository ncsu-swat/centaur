import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if the prepended tensor v_2 has at least one dimension, its first dimension should be bounded by the input tuple length (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) < v["arg1_length"], True)) if n else
          If(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) < v["arg1_length"], True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
