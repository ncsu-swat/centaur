import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Input tuple must contain between 1 and 10 elements (Rule 110)

rule_110 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] >= 1, v["arg1_length"] <= 10)) if n else
          And(v["arg1_length"] >= 1, v["arg1_length"] <= 10))
)

def rule_110_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)) or (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 110
        rule_110(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_110(solver, {'arg1_length': arg1['length']}, neg)
