import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# float input should be within safe mathematical boundaries to avoid overflow or underflow (Rule 178)

rule_178 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, v["arg1_value"] < 1024, v["arg1_value"] > -1024)) if n else
          If(v["arg1_value"] > 0, v["arg1_value"] < 1024, v["arg1_value"] > -1024))
)

def rule_178_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 178
        rule_178(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_178(solver, {'arg1_value': arg1['value']}, neg)
