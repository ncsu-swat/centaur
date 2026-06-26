import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Convert a list of floats to a target dtype name (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= -1000.0, Select(v["arg1_values"], i) <= 1000.0)) for i in range(6)]))), (Or(v["arg2_value"] == 38, v["arg2_value"] == 39)))) if n else
          And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= -1000.0, Select(v["arg1_values"], i) <= 1000.0)) for i in range(6)]))), (Or(v["arg2_value"] == 38, v["arg2_value"] == 39))))
)

def rule_53_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 53
        rule_53(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)
