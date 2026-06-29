import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# tensor-float addition requires tensor to have float dtype (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 0, (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8)), (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8)))) if n else
          If(v["arg2_value"] == 0, (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8)), (And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8))))
)

def rule_100_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 100
        rule_100(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
