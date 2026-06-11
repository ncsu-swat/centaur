import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# scalar inputs safe range (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And(And(And(-1000000 <= v["arg1_value"], v["arg1_value"] <= 1000000), -1000000 <= v["arg2_value"]), v["arg2_value"] <= 1000000)) if n else
          And(And(And(-1000000 <= v["arg1_value"], v["arg1_value"] <= 1000000), -1000000 <= v["arg2_value"]), v["arg2_value"] <= 1000000))
)

def rule_29_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
