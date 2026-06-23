import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# optimize parameter can be "none" or boolean (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_value"] == 5, v["arg1_value"] == True), v["arg1_value"] == False)) if n else
          Or(Or(v["arg1_value"] == 5, v["arg1_value"] == True), v["arg1_value"] == False))
)

def rule_71_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str) or isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 71
        rule_71(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_value': arg1['value']}, neg)
