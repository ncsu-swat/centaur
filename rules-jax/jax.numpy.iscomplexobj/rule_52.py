import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# A union variable of float and bool must be a boolean or a float larger than -1000.0 (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not((If(Or(v["arg1_value"] == True, v["arg1_value"] == False), True, v["arg1_value"] > -1000.0))) if n else
          (If(Or(v["arg1_value"] == True, v["arg1_value"] == False), True, v["arg1_value"] > -1000.0)))
)

def rule_52_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value']}, neg)
