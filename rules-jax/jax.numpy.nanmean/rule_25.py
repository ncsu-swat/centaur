import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# where mask must be boolean and tuple axis elements within tensor dimension bounds (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg4_dtype"] == 0, (And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) >= 0 - v["arg1_ndim"], Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)) for i in range(6)]))), (Or(v["arg3_value"] == True, v["arg3_value"] == False)))) if n else
          And(And(v["arg4_dtype"] == 0, (And([Implies(i < (v["arg2_length"] - 1 + 1), And(Select(v["arg2_values"], i) >= 0 - v["arg1_ndim"], Select(v["arg2_values"], i) <= v["arg1_ndim"] - 1)) for i in range(6)]))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))))
)

def rule_25_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == arg3)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 25
        rule_25(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype']}, neg)
