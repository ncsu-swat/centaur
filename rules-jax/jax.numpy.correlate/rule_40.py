import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The precision tuple must contain exactly two elements and must not contain mode strings (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) != 19), Select(v["arg1_values"], 1) != 20)) if n else
          And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) != 19), Select(v["arg1_values"], 1) != 20))
)

def rule_40_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 40
        rule_40(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
