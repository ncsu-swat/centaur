import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# the input tuple of floats must have at least one element, and the first element must be at least -180 degrees (Rule 137)

rule_137 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] >= 1, Select(v["arg1_values"], 0) >= -180)) if n else
          And(v["arg1_length"] >= 1, Select(v["arg1_values"], 0) >= -180))
)

def rule_137_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 137
        rule_137(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
