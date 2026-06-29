import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the input is a scalar float or int, its value should be within the range of negative to positive pi (Rule 148)

rule_148 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= -3.14, v["arg1_value"] <= 3.14)) if n else
          And(v["arg1_value"] >= -3.14, v["arg1_value"] <= 3.14))
)

def rule_148_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 148
        rule_148(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_148(solver, {'arg1_value': arg1['value']}, neg)
