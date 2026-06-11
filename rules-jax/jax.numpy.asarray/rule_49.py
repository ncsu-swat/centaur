import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Convert a list of non-negative integers to array with valid options (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0) for i in range(6)]))), v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_value"] == 5), (Or(v["arg4_value"] == True, v["arg4_value"] == False)))) if n else
          And(And(And(And(And(v["arg1_length"] > 0, (And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) >= 0) for i in range(6)]))), v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_value"] == 5), (Or(v["arg4_value"] == True, v["arg4_value"] == False))))
)

def rule_49_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType) or isinstance(arg2, np.dtype)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values_jax.index(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 49
        rule_49(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
