import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Valid 1D or higher floating-point input array with tuple of integer axes and matching shape boolean mask (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] >= 1, v["arg2_length"] >= 1), (And([Implies(i < (v["arg2_length"] - 1 + 1), And(-1 * v["arg1_ndim"] <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)) for i in range(6)]))), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 8), v["arg1_ndim"] == v["arg3_ndim"]), v["arg3_dtype"] == 0), And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) == Select(v["arg3_shape"], j)) for j in range(6)]))) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] >= 1, v["arg2_length"] >= 1), (And([Implies(i < (v["arg2_length"] - 1 + 1), And(-1 * v["arg1_ndim"] <= Select(v["arg2_values"], i), Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)) for i in range(6)]))), 6 <= v["arg1_dtype"]), v["arg1_dtype"] <= 8), v["arg1_ndim"] == v["arg3_ndim"]), v["arg3_dtype"] == 0), And([Implies(j < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], j) == Select(v["arg3_shape"], j)) for j in range(6)])))
)

def rule_86_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 86
        rule_86(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
