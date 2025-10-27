import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Uninitialized parameter must have valid shape (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(v_1 != v_1) if n else
          v_1 != v_1)
)

def rule_84_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 84
        rule_84(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {}, neg)
