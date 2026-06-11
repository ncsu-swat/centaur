import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# floating-point or complex type input with string accuracy setting (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 10), (Or(v["arg2_value"] == 37, v["arg2_value"] == 5)))) if n else
          And(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 10), (Or(v["arg2_value"] == 37, v["arg2_value"] == 5))))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))

        # Constraints for rule 20
        rule_20(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
