import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when arg1 is a string and arg2 is a dtype, they must be a valid dtype string and a valid dtype index respectively (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_value"] == 38, v["arg1_value"] == 39)), (And(0 <= v["arg2_value"], v["arg2_value"] <= 12)))) if n else
          And((Or(v["arg1_value"] == 38, v["arg1_value"] == 39)), (And(0 <= v["arg2_value"], v["arg2_value"] <= 12))))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
