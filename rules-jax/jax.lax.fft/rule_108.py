import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the FFT type must be a valid option and the FFT length sequence must have at least one element (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(v["arg1_value"] == 29, v["arg1_value"] == 30), v["arg1_value"] == 31), v["arg1_value"] == 32)), v["arg2_length"] >= 1)) if n else
          And((Or(Or(Or(v["arg1_value"] == 29, v["arg1_value"] == 30), v["arg1_value"] == 31), v["arg1_value"] == 32)), v["arg2_length"] >= 1))
)

def rule_108_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not ((isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)) or (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2))):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 108
        rule_108(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
