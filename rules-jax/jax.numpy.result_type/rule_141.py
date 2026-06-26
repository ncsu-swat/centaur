import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# If the scalar numeric parameter is negative, the tensor must contain negative values (Rule 141)

rule_141 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 0, Select(v["arg2_range"], 0) < 0, Select(v["arg2_range"], 0) >= 0)) if n else
          If(v["arg1_value"] < 0, Select(v["arg2_range"], 0) < 0, Select(v["arg2_range"], 0) >= 0))
)

def rule_141_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 141
        rule_141(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_141(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range']}, neg)
