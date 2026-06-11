import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Preferred element type must represent a valid datatype index when LHS input is at least 3D (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_ndim"] >= 3, (And(0 <= v["arg1_value"], v["arg1_value"] <= 12)))) if n else
          And(v["arg2_ndim"] >= 3, (And(0 <= v["arg1_value"], v["arg1_value"] <= 12))))
)

def rule_32_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
