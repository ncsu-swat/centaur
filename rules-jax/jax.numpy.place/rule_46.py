import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# full API constraints when vals is a tuple: matching array and mask dimensions, boolean mask, non-empty tuple, and inplace set to False (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] == v["arg2_ndim"], v["arg2_dtype"] == 0), v["arg3_length"] > 0), v["arg4_value"] == False)) if n else
          And(And(And(v["arg1_ndim"] == v["arg2_ndim"], v["arg2_dtype"] == 0), v["arg3_length"] > 0), v["arg4_value"] == False))
)

def rule_46_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not ((isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)) or (isinstance(arg3, tuple) and all(isinstance(e, (float, np.floating)) for e in arg3))):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_length = Int('arg3_length')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 46
        rule_46(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_length': arg3_length, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value']}, neg)
