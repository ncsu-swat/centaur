import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# coefficients tensor must have a rank of 1 and support NaN representation if leading zeros are not stripped (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 1, If(v["arg2_value"] == False, And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 10)))) if n else
          And(v["arg1_ndim"] == 1, If(v["arg2_value"] == False, And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 10))))
)

def rule_103_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 103
        rule_103(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
