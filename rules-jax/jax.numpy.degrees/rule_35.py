import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# for a tuple of floats representing 3D Euler angles, it should contain exactly 3 elements within the -pi to pi range (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == 3, And([Implies(i < (2 + 1), And(Select(v["arg1_values"], i) >= -3.15, Select(v["arg1_values"], i) <= 3.15)) for i in range(6)]))) if n else
          And(v["arg1_length"] == 3, And([Implies(i < (2 + 1), And(Select(v["arg1_values"], i) >= -3.15, Select(v["arg1_values"], i) <= 3.15)) for i in range(6)])))
)

def rule_35_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 35
        rule_35(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
