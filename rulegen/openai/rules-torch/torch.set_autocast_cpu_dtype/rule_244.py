import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor has dimensions of size 219 (Rule 244)

rule_244 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_shape"], 0) == 219, Select(v["arg1_shape"], 1) == 219)) if n else
          And(Select(v["arg1_shape"], 0) == 219, Select(v["arg1_shape"], 1) == 219))
)

def rule_244_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 244
        rule_244(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_244(solver, {'arg1_shape': arg1['shape']}, neg)
