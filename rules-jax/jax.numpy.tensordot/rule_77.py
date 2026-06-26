import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# elements of the axes list must be valid dimensions of the input tensors (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_length"] <= v["arg1_ndim"], v["arg3_length"] <= v["arg2_ndim"]), (And([Implies(i < (v["arg3_length"] - 1 + 1), And(And(Select(v["arg3_values"], i) >= 0, Select(v["arg3_values"], i) < v["arg1_ndim"]), Select(v["arg3_values"], i) < v["arg2_ndim"])) for i in range(6)])))) if n else
          And(And(v["arg3_length"] <= v["arg1_ndim"], v["arg3_length"] <= v["arg2_ndim"]), (And([Implies(i < (v["arg3_length"] - 1 + 1), And(And(Select(v["arg3_values"], i) >= 0, Select(v["arg3_values"], i) < v["arg1_ndim"]), Select(v["arg3_values"], i) < v["arg2_ndim"])) for i in range(6)]))))
)

def rule_77_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 77
        rule_77(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
