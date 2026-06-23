import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# ensure floats with fractional parts are cast to floating-point or complex types to avoid loss of precision (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] % 1 != 0, And(6 <= v["arg2_value"], v["arg2_value"] <= 10), And(0 <= v["arg2_value"], v["arg2_value"] <= 10))) if n else
          If(v["arg1_value"] % 1 != 0, And(6 <= v["arg2_value"], v["arg2_value"] <= 10), And(0 <= v["arg2_value"], v["arg2_value"] <= 10)))
)

def rule_32_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
