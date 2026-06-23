import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for valid convolution, the first input must be at least as large as the second input along all specified axes in tuple (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 19, And([Implies(i < (v["arg4_length"] - 1 + 1), (If(Select(v["arg4_values"], i) >= 0, Select(v["arg1_shape"], Select(v["arg4_values"], i)) >= Select(v["arg2_shape"], Select(v["arg4_values"], i)), Select(v["arg1_shape"], v["arg1_ndim"] + Select(v["arg4_values"], i)) >= Select(v["arg2_shape"], v["arg1_ndim"] + Select(v["arg4_values"], i))))) for i in range(6)]), True)) if n else
          If(v["arg3_value"] == 19, And([Implies(i < (v["arg4_length"] - 1 + 1), (If(Select(v["arg4_values"], i) >= 0, Select(v["arg1_shape"], Select(v["arg4_values"], i)) >= Select(v["arg2_shape"], Select(v["arg4_values"], i)), Select(v["arg1_shape"], v["arg1_ndim"] + Select(v["arg4_values"], i)) >= Select(v["arg2_shape"], v["arg1_ndim"] + Select(v["arg4_values"], i))))) for i in range(6)]), True))
)

def rule_29_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 29
        rule_29(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_values': arg4_values, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length']}, neg)
