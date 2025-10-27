import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor must be at least 2D with valid dimensions (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_ndim"] >= 2, v_2 >= 0), v_3 >= 0), v_2 < v["arg1_ndim"]), v_3 < v["arg1_ndim"])) if n else
          And(And(And(And(v["arg1_ndim"] >= 2, v_2 >= 0), v_3 >= 0), v_2 < v["arg1_ndim"]), v_3 < v["arg1_ndim"]))
)

def rule_12_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 12
        rule_12(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_ndim': arg1['ndim']}, neg)
