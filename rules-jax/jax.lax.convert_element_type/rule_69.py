import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# when casting to integer types, the input tensor values should be within a safe integer representation range (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg2_value"] >= 1, v["arg2_value"] <= 5)), (And(Select(v["arg1_range"], 0) >= -100000, Select(v["arg1_range"], 1) <= 100000)), (And(v["arg2_value"] >= 0, v["arg2_value"] <= 10)))) if n else
          If((And(v["arg2_value"] >= 1, v["arg2_value"] <= 5)), (And(Select(v["arg1_range"], 0) >= -100000, Select(v["arg1_range"], 1) <= 100000)), (And(v["arg2_value"] >= 0, v["arg2_value"] <= 10))))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 69
        rule_69(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
