import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Combined parameter validation for JAX DFT matrix generation (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] >= 1, v["arg2_value"] == 5), 9 <= v["arg3_value"]), v["arg3_value"] <= 10)) if n else
          And(And(And(v["arg1_value"] >= 1, v["arg2_value"] == 5), 9 <= v["arg3_value"]), v["arg3_value"] <= 10))
)

def rule_12_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType) or isinstance(arg3, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
