import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the excluded tuple is non-empty, its first element must be a valid argument index (Rule 129)

rule_129 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, Select(v["arg1_values"], 0) < v["arg2_length"], True)) if n else
          If(v["arg1_length"] > 0, Select(v["arg1_values"], 0) < v["arg2_length"], True))
)

def rule_129_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 129
        rule_129(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_129(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_length': arg2['length']}, neg)
