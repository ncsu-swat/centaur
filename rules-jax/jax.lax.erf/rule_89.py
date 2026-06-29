import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The scalar input value should be bounded within a safe range to prevent extreme calculations (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= -500.0, v["arg1_value"] <= 500.0)) if n else
          And(v["arg1_value"] >= -500.0, v["arg1_value"] <= 500.0))
)

def rule_89_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 89
        rule_89(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_value': arg1['value']}, neg)
