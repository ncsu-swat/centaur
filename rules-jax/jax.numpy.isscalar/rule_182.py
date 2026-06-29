import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# element as a union of int and float (Rule 182)

rule_182 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] < 1000000, v["arg1_value"] >= 1000000)) if n else
          Or(v["arg1_value"] < 1000000, v["arg1_value"] >= 1000000))
)

def rule_182_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 182
        rule_182(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_182(solver, {'arg1_value': arg1['value']}, neg)
