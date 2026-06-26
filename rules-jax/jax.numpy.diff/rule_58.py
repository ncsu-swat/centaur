import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# prepend and append tensors must have the same shape along non-axis dimensions if both are provided (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg2_ndim"] > 0, v["arg3_ndim"] > 0), v["arg2_ndim"] == v["arg3_ndim"]), (If(v["arg1_value"] < 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(i == v["arg1_value"] + v["arg2_ndim"], True, Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)])), (And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(i == v["arg1_value"], True, Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)])))), True)) if n else
          If(And(And(v["arg2_ndim"] > 0, v["arg3_ndim"] > 0), v["arg2_ndim"] == v["arg3_ndim"]), (If(v["arg1_value"] < 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(i == v["arg1_value"] + v["arg2_ndim"], True, Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)])), (And([Implies(i < (v["arg2_ndim"] - 1 + 1), If(i == v["arg1_value"], True, Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i))) for i in range(6)])))), True))
)

def rule_58_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)
