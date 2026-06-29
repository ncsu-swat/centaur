import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# accuracy tolerance must be a pair of non-negative floats, conditional on input tensor having floating-point or complex type (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If((Or([And(i < (10 + 1), v["arg1_dtype"] == i) for i in range(6)])), And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), False)) if n else
          If((Or([And(i < (10 + 1), v["arg1_dtype"] == i) for i in range(6)])), And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) >= 0), Select(v["arg2_values"], 1) >= 0), False))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 63
        rule_63(solver, {'arg1_dtype': arg1_dtype, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_dtype': arg1['dtype'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
