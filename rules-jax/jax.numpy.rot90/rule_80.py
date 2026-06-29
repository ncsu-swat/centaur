import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# validity of the rotation axes in relation to rotation count (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 1, And(And(And(And(v["arg1_ndim"] >= 2, v["arg3_length"] == 2), Select(v["arg3_values"], 0) != Select(v["arg3_values"], 1)), -1 * v["arg1_ndim"] <= Select(v["arg3_values"], 0)), Select(v["arg3_values"], 0) < v["arg1_ndim"]), And(And(v["arg1_ndim"] >= 2, v["arg3_length"] == 2), Select(v["arg3_values"], 0) != Select(v["arg3_values"], 1)))) if n else
          If(v["arg2_value"] == 1, And(And(And(And(v["arg1_ndim"] >= 2, v["arg3_length"] == 2), Select(v["arg3_values"], 0) != Select(v["arg3_values"], 1)), -1 * v["arg1_ndim"] <= Select(v["arg3_values"], 0)), Select(v["arg3_values"], 0) < v["arg1_ndim"]), And(And(v["arg1_ndim"] >= 2, v["arg3_length"] == 2), Select(v["arg3_values"], 0) != Select(v["arg3_values"], 1))))
)

def rule_80_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 80
        rule_80(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
