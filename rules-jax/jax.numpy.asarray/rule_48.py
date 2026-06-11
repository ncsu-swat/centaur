import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Convert a tensor with valid options where copy mode affects allowable dimensions (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(And(And(And((If(v["arg4_value"] == True, v["arg1_ndim"] >= 0, v["arg1_ndim"] > 0)), v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_value"] == 5)) if n else
          And(And(And((If(v["arg4_value"] == True, v["arg1_ndim"] >= 0, v["arg1_ndim"] > 0)), v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_value"] == 5))
)

def rule_48_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 48
        rule_48(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
