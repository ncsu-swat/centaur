import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the input tensor rank must be at least equal to the length of the fft_lengths tuple or list (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (3 + 1), If(v["arg2_length"] == i, v["arg1_ndim"] >= i, True)) for i in range(6)])) if n else
          And([Implies(i < (3 + 1), If(v["arg2_length"] == i, v["arg1_ndim"] >= i, True)) for i in range(6)]))
)

def rule_17_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)) or (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 17
        rule_17(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
