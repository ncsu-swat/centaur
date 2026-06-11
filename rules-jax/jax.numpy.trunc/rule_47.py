import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# scalar input must be within a safe limit (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] > -999999, v["arg1_value"] < 999999)) if n else
          And(v["arg1_value"] > -999999, v["arg1_value"] < 999999))
)

def rule_47_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 47
        rule_47(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_value': arg1['value']}, neg)
