import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Rank constraint and dtype range for tuple shape with tensor fill value (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_ndim"] <= v["arg1_length"], 0 <= v["arg3_value"]), v["arg3_value"] <= 10)) if n else
          And(And(v["arg2_ndim"] <= v["arg1_length"], 0 <= v["arg3_value"]), v["arg3_value"] <= 10))
)

def rule_18_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType) or isinstance(arg3, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 18
        rule_18(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
