import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# float input angle should be within one full rotation in radians (Rule 146)

rule_146 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= -3.14159, v["arg1_value"] <= 3.14159)) if n else
          And(v["arg1_value"] >= -3.14159, v["arg1_value"] <= 3.14159))
)

def rule_146_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 146
        rule_146(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_146(solver, {'arg1_value': arg1['value']}, neg)
