import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for three tensor arguments, the scale tensor must contain positive values or be empty (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(And((And(v["arg1_ndim"] == v["arg2_ndim"], v["arg2_ndim"] == v["arg3_ndim"])), (Or((Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 0) for i in range(6)])), Select(v["arg3_range"], 0) > 0)))) if n else
          And((And(v["arg1_ndim"] == v["arg2_ndim"], v["arg2_ndim"] == v["arg3_ndim"])), (Or((Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 0) for i in range(6)])), Select(v["arg3_range"], 0) > 0))))
)

def rule_16_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 16
        rule_16(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_range': arg3_range, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_range': arg3['range'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
