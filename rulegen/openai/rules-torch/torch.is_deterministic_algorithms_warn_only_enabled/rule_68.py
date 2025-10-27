import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor shape validation using tuple variables (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 32, Select(v["arg1_shape"], 0) == Select(v["arg2_values"], 0)), Select(v["arg1_shape"], 1) == Select(v["arg2_values"], 1)), Select(v["arg1_shape"], 2) == Select(v["arg2_values"], 2)), Select(v["arg1_shape"], 3) == Select(v["arg2_values"], 3)), Select(v["arg1_shape"], 4) == Select(v["arg2_values"], 4)), Select(v["arg1_shape"], 5) == Select(v["arg2_values"], 5)), Select(v["arg1_shape"], 6) == Select(v["arg2_values"], 6)), Select(v["arg1_shape"], 7) == Select(v["arg2_values"], 7)), Select(v["arg1_shape"], 8) == Select(v["arg2_values"], 8)), Select(v["arg1_shape"], 9) == Select(v["arg2_values"], 9)), Select(v["arg1_shape"], 10) == Select(v["arg2_values"], 10)), Select(v["arg1_shape"], 11) == Select(v["arg2_values"], 11)), Select(v["arg1_shape"], 12) == Select(v["arg2_values"], 12)), Select(v["arg1_shape"], 13) == Select(v["arg2_values"], 13)), Select(v["arg1_shape"], 14) == Select(v["arg2_values"], 14)), Select(v["arg1_shape"], 15) == Select(v["arg2_values"], 15)), Select(v["arg1_shape"], 16) == Select(v["arg2_values"], 16)), Select(v["arg1_shape"], 17) == Select(v["arg2_values"], 17)), Select(v["arg1_shape"], 18) == Select(v["arg2_values"], 18)), Select(v["arg1_shape"], 19) == Select(v["arg2_values"], 19)), Select(v["arg1_shape"], 20) == Select(v["arg2_values"], 20)), Select(v["arg1_shape"], 21) == Select(v["arg2_values"], 21)), Select(v["arg1_shape"], 22) == Select(v["arg2_values"], 22)), Select(v["arg1_shape"], 23) == Select(v["arg2_values"], 23)), Select(v["arg1_shape"], 24) == Select(v["arg2_values"], 24)), Select(v["arg1_shape"], 25) == Select(v["arg2_values"], 25)), Select(v["arg1_shape"], 26) == Select(v["arg2_values"], 26)), Select(v["arg1_shape"], 27) == Select(v["arg2_values"], 27)), Select(v["arg1_shape"], 28) == Select(v["arg2_values"], 28)), Select(v["arg1_shape"], 29) == Select(v["arg2_values"], 29)), Select(v["arg1_shape"], 30) == Select(v["arg2_values"], 30)), Select(v["arg1_shape"], 31) == Select(v["arg2_values"], 31))) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 32, Select(v["arg1_shape"], 0) == Select(v["arg2_values"], 0)), Select(v["arg1_shape"], 1) == Select(v["arg2_values"], 1)), Select(v["arg1_shape"], 2) == Select(v["arg2_values"], 2)), Select(v["arg1_shape"], 3) == Select(v["arg2_values"], 3)), Select(v["arg1_shape"], 4) == Select(v["arg2_values"], 4)), Select(v["arg1_shape"], 5) == Select(v["arg2_values"], 5)), Select(v["arg1_shape"], 6) == Select(v["arg2_values"], 6)), Select(v["arg1_shape"], 7) == Select(v["arg2_values"], 7)), Select(v["arg1_shape"], 8) == Select(v["arg2_values"], 8)), Select(v["arg1_shape"], 9) == Select(v["arg2_values"], 9)), Select(v["arg1_shape"], 10) == Select(v["arg2_values"], 10)), Select(v["arg1_shape"], 11) == Select(v["arg2_values"], 11)), Select(v["arg1_shape"], 12) == Select(v["arg2_values"], 12)), Select(v["arg1_shape"], 13) == Select(v["arg2_values"], 13)), Select(v["arg1_shape"], 14) == Select(v["arg2_values"], 14)), Select(v["arg1_shape"], 15) == Select(v["arg2_values"], 15)), Select(v["arg1_shape"], 16) == Select(v["arg2_values"], 16)), Select(v["arg1_shape"], 17) == Select(v["arg2_values"], 17)), Select(v["arg1_shape"], 18) == Select(v["arg2_values"], 18)), Select(v["arg1_shape"], 19) == Select(v["arg2_values"], 19)), Select(v["arg1_shape"], 20) == Select(v["arg2_values"], 20)), Select(v["arg1_shape"], 21) == Select(v["arg2_values"], 21)), Select(v["arg1_shape"], 22) == Select(v["arg2_values"], 22)), Select(v["arg1_shape"], 23) == Select(v["arg2_values"], 23)), Select(v["arg1_shape"], 24) == Select(v["arg2_values"], 24)), Select(v["arg1_shape"], 25) == Select(v["arg2_values"], 25)), Select(v["arg1_shape"], 26) == Select(v["arg2_values"], 26)), Select(v["arg1_shape"], 27) == Select(v["arg2_values"], 27)), Select(v["arg1_shape"], 28) == Select(v["arg2_values"], 28)), Select(v["arg1_shape"], 29) == Select(v["arg2_values"], 29)), Select(v["arg1_shape"], 30) == Select(v["arg2_values"], 30)), Select(v["arg1_shape"], 31) == Select(v["arg2_values"], 31)))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 68
        rule_68(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values']}, neg)
