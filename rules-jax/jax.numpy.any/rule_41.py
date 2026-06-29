import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# axis parameter as tensor must contain valid dimension indices (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(v["arg2_ndim"] == 0, v["arg2_ndim"] == 1)), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0 - v["arg1_ndim"]), Select(v["arg2_range"], 1) < v["arg1_ndim"])) if n else
          And(And(And(And((Or(v["arg2_ndim"] == 0, v["arg2_ndim"] == 1)), 1 <= v["arg2_dtype"]), v["arg2_dtype"] <= 5), Select(v["arg2_range"], 0) >= 0 - v["arg1_ndim"]), Select(v["arg2_range"], 1) < v["arg1_ndim"]))
)

def rule_41_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 41
        rule_41(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
