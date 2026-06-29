import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if ord is a float, the axis tuple must be of length 1 since float ord is only for vector norms (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == v["arg1_value"], v["arg2_length"] == 1, True)) if n else
          If(v["arg1_value"] == v["arg1_value"], v["arg2_length"] == 1, True))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 15
        rule_15(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
