import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# precision tuple must be of length 2 containing valid precision indicators (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == 2, (And([Implies(i < (1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 37, Select(v["arg1_values"], i) == 38), Select(v["arg1_values"], i) == 39), Select(v["arg1_values"], i) == 5)) for i in range(6)])))) if n else
          And(v["arg1_length"] == 2, (And([Implies(i < (1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 37, Select(v["arg1_values"], i) == 38), Select(v["arg1_values"], i) == 39), Select(v["arg1_values"], i) == 5)) for i in range(6)]))))
)

def rule_93_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 93
        rule_93(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)
