import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the input is a list of floats, its length must be positive and not exceed 100000 (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] >= 1, v["arg1_length"] <= 100000)) if n else
          And(v["arg1_length"] >= 1, v["arg1_length"] <= 100000))
)

def rule_73_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 73
        rule_73(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_length': arg1['length']}, neg)
