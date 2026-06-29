import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# For an input z of union type int or float, its value should be within safe bounds, and deg must be boolean (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > -500000, v["arg1_value"] < 500000), (Or(v["arg2_value"] == True, v["arg2_value"] == False)))) if n else
          And(And(v["arg1_value"] > -500000, v["arg1_value"] < 500000), (Or(v["arg2_value"] == True, v["arg2_value"] == False))))
)

def rule_94_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 94
        rule_94(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
