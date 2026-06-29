import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for positive scalar numeric inputs, doubling the value increases it (Rule 117)

rule_117 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, v["arg1_value"] * 2 > v["arg1_value"], True)) if n else
          If(v["arg1_value"] > 0, v["arg1_value"] * 2 > v["arg1_value"], True))
)

def rule_117_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 117
        rule_117(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_117(solver, {'arg1_value': arg1['value']}, neg)
