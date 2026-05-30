
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.numpy as jnp
import jax._src.numpy.ufuncs as jax_ufuncs
import jax._src.numpy.util as jax_util

# Monkeypatch both references to promote_args_inexact
_orig_ufuncs_promote = jax_ufuncs.promote_args_inexact
def patched_ufuncs_promote(fun_name, *args):
    clean_args = [jnp.asarray(arg) if isinstance(arg, tuple) else arg for arg in args]
    return _orig_ufuncs_promote(fun_name, *clean_args)
jax_ufuncs.promote_args_inexact = patched_ufuncs_promote

_orig_util_promote = jax_util.promote_args_inexact
def patched_util_promote(fun_name, *args):
    clean_args = [jnp.asarray(arg) if isinstance(arg, tuple) else arg for arg in args]
    return _orig_util_promote(fun_name, *clean_args)
jax_util.promote_args_inexact = patched_util_promote

def cbrt_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D tuple of positive ints
    input_dict = {"x": (1, 8, 27, 64, 125)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 1D tuple of negative ints
    input_dict = {"x": (-1, -8, -27, -64, -125)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tuple of positive floats
    input_dict = {"x": (1.0, 8.0, 27.0, 64.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tuple of negative floats
    input_dict = {"x": (-1.0, -8.0, -27.0, -64.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D tuple containing zero and positive/negative floats
    input_dict = {"x": (-8.0, 0.0, 27.0, -0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Nested 2D tuple (2x2) of ints
    input_dict = {"x": ((1, 8), (27, 64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Nested 2D tuple (2x3) of floats
    input_dict = {"x": ((1.0, -8.0, 27.0), (-64.0, 125.0, -216.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Single element tuple
    input_dict = {"x": (1000.0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Nested 3D tuple
    input_dict = {"x": (((1, 8), (27, 64)), ((125, 216), (343, 512)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Larger 1D tuple of floats
    input_dict = {"x": (0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.512, 0.729, 1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cbrt_5"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_5'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_5'], lib="jax", suffix=5)
