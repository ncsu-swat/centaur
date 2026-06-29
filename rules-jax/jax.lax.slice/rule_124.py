import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# For 4-parameter slice, all sequence lengths must match the operand rank (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_length"] == v["arg1_ndim"], v["arg3_length"] == v["arg1_ndim"]), v["arg4_length"] == v["arg1_ndim"])) if n else
          And(And(v["arg2_length"] == v["arg1_ndim"], v["arg3_length"] == v["arg1_ndim"]), v["arg4_length"] == v["arg1_ndim"]))
)

def rule_124_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)) or (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2))):
            return False
        if not ((isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)) or (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3))):
            return False
        if not ((isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)) or (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 124
        rule_124(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_length': arg3_length, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length'], 'arg4_length': arg4['length']}, neg)
