import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# output dimension spec in dimension_numbers must be a valid convention (Rule 193)

rule_193 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Select(v["arg1_values"], 2) == 15, Select(v["arg1_values"], 2) == 16), Select(v["arg1_values"], 2) == 21), Select(v["arg1_values"], 2) == 22), Select(v["arg1_values"], 2) == 26), Select(v["arg1_values"], 2) == 27)) if n else
          Or(Or(Or(Or(Or(Select(v["arg1_values"], 2) == 15, Select(v["arg1_values"], 2) == 16), Select(v["arg1_values"], 2) == 21), Select(v["arg1_values"], 2) == 22), Select(v["arg1_values"], 2) == 26), Select(v["arg1_values"], 2) == 27))
)

def rule_193_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), StringSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 193
        rule_193(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_193(solver, {'arg1_values': arg1['values']}, neg)
