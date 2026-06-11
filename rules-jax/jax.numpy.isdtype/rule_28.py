import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# valid input dtype and kind is a valid category string from the allowed list (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(And(0 <= v["arg1_value"], v["arg1_value"] <= 12), (Or(Or(v["arg2_value"] == 38, v["arg2_value"] == 39), v["arg2_value"] == 37)))) if n else
          And(And(0 <= v["arg1_value"], v["arg1_value"] <= 12), (Or(Or(v["arg2_value"] == 38, v["arg2_value"] == 39), v["arg2_value"] == 37))))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
