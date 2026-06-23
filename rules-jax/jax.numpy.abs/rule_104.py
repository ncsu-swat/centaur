import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Tensor elements must be bounded within a specific decimal range (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_range"], 0) > -123.45, Select(v["arg1_range"], 1) < 123.45)) if n else
          And(Select(v["arg1_range"], 0) > -123.45, Select(v["arg1_range"], 1) < 123.45))
)

def rule_104_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 104
        rule_104(solver, {'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_range': arg1['range']}, neg)
