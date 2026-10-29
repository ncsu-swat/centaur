import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Signature: arr: tensor, obj: tuple, values: tensor - values rank cannot exceed arr rank and tuple of indices must be non-empty (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(And(v["arg3_ndim"] <= v["arg1_ndim"], v["arg2_length"] > 0)) if n else
          And(v["arg3_ndim"] <= v["arg1_ndim"], v["arg2_length"] > 0))
)

def rule_16_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 16
        rule_16(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim']}, neg)
