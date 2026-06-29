import numpy as np
import torch 
import tensorflow as tf
import jax
import jax.numpy as jnp

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_jax, np_dtype
from z3 import *

# argument must be a valid dtype or dtype-representing string (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(Or((Or(v["arg1_value"] == 38, v["arg1_value"] == 39)), (And(v["arg1_value"] >= 0, v["arg1_value"] <= 10)))) if n else
          Or((Or(v["arg1_value"] == 38, v["arg1_value"] == 39)), (And(v["arg1_value"] >= 0, v["arg1_value"] <= 10))))
)

def rule_85_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType) or isinstance(arg1, np.dtype)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 85
        rule_85(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_value': arg1['value']}, neg)
