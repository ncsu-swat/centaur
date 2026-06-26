import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if subtype is complex, supertype cannot be an integer or unsigned integer type (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If((And(9 <= v["arg1_value"], v["arg1_value"] <= 10)), (And([Implies(i < (5 + 1), v["arg2_value"] != i) for i in range(6)])), (And(And(And(0 <= v["arg1_value"], v["arg1_value"] <= 12), 0 <= v["arg2_value"]), v["arg2_value"] <= 12)))) if n else
          If((And(9 <= v["arg1_value"], v["arg1_value"] <= 10)), (And([Implies(i < (5 + 1), v["arg2_value"] != i) for i in range(6)])), (And(And(And(0 <= v["arg1_value"], v["arg1_value"] <= 12), 0 <= v["arg2_value"]), v["arg2_value"] <= 12))))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 79
        rule_79(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
