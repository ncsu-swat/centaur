import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# all dimensions of all operand tensors must be strictly positive (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(And(And(And((And([Implies(i < (2 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (And([Implies(j < (2 + 1), Select(v["arg2_shape"], j) > 0) for j in range(6)]))), (And([Implies(k < (2 + 1), Select(v["arg3_shape"], k) > 0) for k in range(6)]))), (And([Implies(l < (2 + 1), Select(v["arg4_shape"], l) > 0) for l in range(6)])))) if n else
          And(And(And((And([Implies(i < (2 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (And([Implies(j < (2 + 1), Select(v["arg2_shape"], j) > 0) for j in range(6)]))), (And([Implies(k < (2 + 1), Select(v["arg3_shape"], k) > 0) for k in range(6)]))), (And([Implies(l < (2 + 1), Select(v["arg4_shape"], l) > 0) for l in range(6)]))))
)

def rule_20_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 20
        rule_20(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape']}, neg)
