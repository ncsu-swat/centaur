import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# shift amount must be strictly less than the bit width of the input tensor's data type (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(Or((Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 0) for i in range(6)])), (If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), Select(v["arg2_range"], 1) < 8, If(v["arg1_dtype"] == 2, Select(v["arg2_range"], 1) < 16, If(v["arg1_dtype"] == 3, Select(v["arg2_range"], 1) < 32, If(v["arg1_dtype"] == 4, Select(v["arg2_range"], 1) < 64, True))))))) if n else
          Or((Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 0) for i in range(6)])), (If(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), Select(v["arg2_range"], 1) < 8, If(v["arg1_dtype"] == 2, Select(v["arg2_range"], 1) < 16, If(v["arg1_dtype"] == 3, Select(v["arg2_range"], 1) < 32, If(v["arg1_dtype"] == 4, Select(v["arg2_range"], 1) < 64, True)))))))
)

def rule_8_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 8
        rule_8(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
