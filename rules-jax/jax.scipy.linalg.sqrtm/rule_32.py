import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# all first two dimensions of tensor A must be of identical size (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (1 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], 0)) for i in range(6)])) if n else
          And([Implies(i < (1 + 1), Select(v["arg1_shape"], i) == Select(v["arg1_shape"], 0)) for i in range(6)]))
)

def rule_32_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 32
        rule_32(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_shape': arg1['shape']}, neg)
