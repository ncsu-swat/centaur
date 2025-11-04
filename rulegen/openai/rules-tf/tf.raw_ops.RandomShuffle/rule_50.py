import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Comprehensive rule for RandomShuffle parameters (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And(And(And((And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), Or((And(v["arg1_value"] == 0, v["arg2_value"] == 0)), (And(v["arg1_value"] != 0, v["arg2_value"] != 0))))), v["arg4_ndim"] >= 1), Select(v["arg4_shape"], 0) > 0), And([Implies(i < (v["arg4_ndim"] - 1 + 1), And(Select(v["arg4_shape"], i) > 0, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25)))) for i in range(6)]))) if n else
          And(And(And((And(And(v["arg1_value"] >= 0, v["arg2_value"] >= 0), Or((And(v["arg1_value"] == 0, v["arg2_value"] == 0)), (And(v["arg1_value"] != 0, v["arg2_value"] != 0))))), v["arg4_ndim"] >= 1), Select(v["arg4_shape"], 0) > 0), And([Implies(i < (v["arg4_ndim"] - 1 + 1), And(Select(v["arg4_shape"], i) > 0, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25)))) for i in range(6)])))
)

def rule_50_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim']}, neg)
