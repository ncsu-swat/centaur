import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for any valid FFT operation type, fft_lengths list must contain 1 to 3 dimensions (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(Or(v["arg1_value"] == 29, v["arg1_value"] == 30), v["arg1_value"] == 31), v["arg1_value"] == 32)), (And(v["arg2_length"] >= 1, v["arg2_length"] <= 3)), True)) if n else
          If((Or(Or(Or(v["arg1_value"] == 29, v["arg1_value"] == 30), v["arg1_value"] == 31), v["arg1_value"] == 32)), (And(v["arg2_length"] >= 1, v["arg2_length"] <= 3)), True))
)

def rule_40_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_jax.index(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
