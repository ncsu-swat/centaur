import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# if the where tensor has the same rank as the input tensor, its shape along the axes specified in the list must be either 1 or equal to the input tensor's shape along those axes (Rule 131)

rule_131 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_ndim"] == v["arg1_ndim"], (And([Implies(i < (v["arg2_length"] - 1 + 1), If(And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], i) < v["arg1_ndim"]), Or(Select(v["arg3_shape"], Select(v["arg2_values"], i)) == Select(v["arg1_shape"], Select(v["arg2_values"], i)), Select(v["arg3_shape"], Select(v["arg2_values"], i)) == 1), True)) for i in range(6)])), True)) if n else
          If(v["arg3_ndim"] == v["arg1_ndim"], (And([Implies(i < (v["arg2_length"] - 1 + 1), If(And(Select(v["arg2_values"], i) >= 0, Select(v["arg2_values"], i) < v["arg1_ndim"]), Or(Select(v["arg3_shape"], Select(v["arg2_values"], i)) == Select(v["arg1_shape"], Select(v["arg2_values"], i)), Select(v["arg3_shape"], Select(v["arg2_values"], i)) == 1), True)) for i in range(6)])), True))
)

def rule_131_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 131
        rule_131(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_131(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
