import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# scalar input should be within reasonable bounds to avoid extreme precision loss (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(And(-10000 < v["arg1_value"], v["arg1_value"] < 10000)) if n else
          And(-10000 < v["arg1_value"], v["arg1_value"] < 10000))
)

def rule_62_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 62
        rule_62(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_value': arg1['value']}, neg)
