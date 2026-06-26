import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the elements of non-empty input tensors must be within a safe numerical range to prevent floating-point overflow (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And((If(And(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) > 0), And(Select(v["arg1_range"], 0) >= -1000000, Select(v["arg1_range"], 1) <= 1000000), True)), (If(And(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) > 0), And(Select(v["arg2_range"], 0) >= -1000000, Select(v["arg2_range"], 1) <= 1000000), True)))) if n else
          And((If(And(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) > 0), And(Select(v["arg1_range"], 0) >= -1000000, Select(v["arg1_range"], 1) <= 1000000), True)), (If(And(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) > 0), And(Select(v["arg2_range"], 0) >= -1000000, Select(v["arg2_range"], 1) <= 1000000), True))))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 14
        rule_14(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_range': arg2['range']}, neg)
