import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# The input tensor x must not be a complex-valued tensor, meaning its dtype index is neither 9 nor 10 (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (1 + 1), v["arg1_dtype"] != i + 9) for i in range(6)])) if n else
          And([Implies(i < (1 + 1), v["arg1_dtype"] != i + 9) for i in range(6)]))
)

def rule_46_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype']}, neg)
