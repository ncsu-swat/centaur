import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# boolean or floating-point input must be a valid boolean or bounded float (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, True, If(v["arg1_value"] == False, True, And(v["arg1_value"] <= 1000, v["arg1_value"] >= -1000)))) if n else
          If(v["arg1_value"] == True, True, If(v["arg1_value"] == False, True, And(v["arg1_value"] <= 1000, v["arg1_value"] >= -1000))))
)

def rule_38_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value']}, neg)
