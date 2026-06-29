import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# When the first argument is an integer dtype, the second must also be an integer dtype or np.dtype (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_value"] >= 1, v["arg1_value"] <= 5)), Or((And(v["arg2_value"] >= 1, v["arg2_value"] <= 5)), v["arg2_value"] == 12), True)) if n else
          If((And(v["arg1_value"] >= 1, v["arg1_value"] <= 5)), Or((And(v["arg2_value"] >= 1, v["arg2_value"] <= 5)), v["arg2_value"] == 12), True))
)

def rule_119_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 119
        rule_119(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
