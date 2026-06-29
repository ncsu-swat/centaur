import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# real-valued scalar input must be within the closed interval [-1, 1] to avoid NaN values (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1)) if n else
          And(v["arg1_value"] >= -1, v["arg1_value"] <= 1))
)

def rule_93_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 93
        rule_93(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_value': arg1['value']}, neg)
