import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# Input shape matches fft_lengths for FFT/IFFT/RFFT with list-typed fft_lengths (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg2_value"] == 29, v["arg2_value"] == 30), v["arg2_value"] == 31), (And(And(v["arg3_length"] > 0, v["arg3_length"] <= v["arg1_ndim"]), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg1_shape"], v["arg1_ndim"] - v["arg3_length"] + i) == Select(v["arg3_values"], i)) for i in range(6)]))), True)) if n else
          If(Or(Or(v["arg2_value"] == 29, v["arg2_value"] == 30), v["arg2_value"] == 31), (And(And(v["arg3_length"] > 0, v["arg3_length"] <= v["arg1_ndim"]), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg1_shape"], v["arg1_ndim"] - v["arg3_length"] + i) == Select(v["arg3_values"], i)) for i in range(6)]))), True))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_jax.index(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 63
        rule_63(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
