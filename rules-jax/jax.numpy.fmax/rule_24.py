import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Float input range constraint depending on tensor dimensionality (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 1, And(v["arg1_value"] > -100000, v["arg1_value"] < 100000), And(v["arg1_value"] > -1000000, v["arg1_value"] < 1000000))) if n else
          If(v["arg2_ndim"] == 1, And(v["arg1_value"] > -100000, v["arg1_value"] < 100000), And(v["arg1_value"] > -1000000, v["arg1_value"] < 1000000)))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
