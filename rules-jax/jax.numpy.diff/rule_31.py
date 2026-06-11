import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the total size along the axis of difference after prepending and appending must be greater than n (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_ndim"] - 1 + 1), (Or((And(i != v["arg3_value"], i != v["arg3_value"] + v["arg1_ndim"])), Select(v["arg1_shape"], i) + Select(v["arg4_shape"], i) + Select(v["arg5_shape"], i) > v["arg2_value"]))) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_ndim"] - 1 + 1), (Or((And(i != v["arg3_value"], i != v["arg3_value"] + v["arg1_ndim"])), Select(v["arg1_shape"], i) + Select(v["arg4_shape"], i) + Select(v["arg5_shape"], i) > v["arg2_value"]))) for i in range(6)]))
)

def rule_31_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 31
        rule_31(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg5_shape': arg5['shape']}, neg)
