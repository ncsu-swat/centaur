import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when values is a tensor and obj is a tuple of integers, the shape of values along the specified insertion axis must match the length of the tuple or be 1 (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg3_ndim"] == v["arg1_ndim"], v["arg4_value"] >= 0), v["arg4_value"] < v["arg1_ndim"]), Or(Select(v["arg3_shape"], v["arg4_value"]) == 1, Select(v["arg3_shape"], v["arg4_value"]) == v["arg2_length"]), True)) if n else
          If(And(And(v["arg3_ndim"] == v["arg1_ndim"], v["arg4_value"] >= 0), v["arg4_value"] < v["arg1_ndim"]), Or(Select(v["arg3_shape"], v["arg4_value"]) == 1, Select(v["arg3_shape"], v["arg4_value"]) == v["arg2_length"]), True))
)

def rule_66_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 66
        rule_66(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
