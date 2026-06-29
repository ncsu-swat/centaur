import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# subset_by_index, if provided, must contain exactly 2 elements; otherwise, it must be empty (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_length"] == 2, v["arg1_length"] == 0)) if n else
          Or(v["arg1_length"] == 2, v["arg1_length"] == 0))
)

def rule_15_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 15
        rule_15(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_length': arg1['length']}, neg)
