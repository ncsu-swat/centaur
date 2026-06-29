import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the combined size of the input array, prepend, and append tensors along the axis must be strictly greater than the order of difference to ensure a non-empty result (Rule 105)

rule_105 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg4_ndim"] != v["arg1_ndim"], v["arg5_ndim"] != v["arg1_ndim"]), Select(v["arg1_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg1_ndim"], v["arg3_value"]))) + Select(v["arg4_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg4_ndim"], v["arg3_value"]))) + Select(v["arg5_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg5_ndim"], v["arg3_value"]))) > v["arg2_value"])) if n else
          Or(Or(v["arg4_ndim"] != v["arg1_ndim"], v["arg5_ndim"] != v["arg1_ndim"]), Select(v["arg1_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg1_ndim"], v["arg3_value"]))) + Select(v["arg4_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg4_ndim"], v["arg3_value"]))) + Select(v["arg5_shape"], (If(v["arg3_value"] < 0, v["arg3_value"] + v["arg5_ndim"], v["arg3_value"]))) > v["arg2_value"]))
)

def rule_105_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 105
        rule_105(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim, 'arg5_shape': arg5_shape, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_105(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim'], 'arg5_shape': arg5['shape'], 'arg5_ndim': arg5['ndim']}, neg)
