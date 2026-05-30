
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.util as jax_util
import jax._src.numpy.ufuncs as jax_ufuncs

orig_promote_args_inexact = jax_util.promote_args_inexact

def patched_promote_args_inexact(fun_name, *args):
    new_args = []
    for arg in args:
        if isinstance(arg, list):
            new_args.append(jnp.asarray(arg))
        else:
            new_args.append(arg)
    return orig_promote_args_inexact(fun_name, *new_args)

# Patch promote_args_inexact in all namespaces
jax_util.promote_args_inexact = patched_promote_args_inexact
if hasattr(jax_ufuncs, 'promote_args_inexact'):
    jax_ufuncs.promote_args_inexact = patched_promote_args_inexact

# Patch check_arraylike just in case
jax_util.check_arraylike = lambda *args, **kwargs: None
if hasattr(jax_ufuncs, 'check_arraylike'):
    jax_ufuncs.check_arraylike = lambda *args, **kwargs: None

def radians_inputs():
    list_of_inputs = []

    # Input 1: 1D list of standard positive angles (floats)
    input_dict = {"x": [0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D list of negative angles (floats)
    input_dict = {"x": [-30.0, -45.0, -90.0, -180.0, -360.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D list of integers
    input_dict = {"x": [0, 90, 180, 270, 360]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D list of floats
    input_dict = {"x": [[0.0, 90.0], [180.0, 270.0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D nested list of floats
    input_dict = {"x": [[[0.0, 45.0]], [[90.0, 135.0]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List containing negative and positive integers
    input_dict = {"x": [-180, -90, 0, 90, 180]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with very large and small values
    input_dict = {"x": [1e-6, 1e5, -1e5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty list
    input_dict = {"x": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D list of integers
    input_dict = {"x": [[0, -45], [45, 90]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of numpy float64 types
    input_dict = {"x": [np.float64(30.0), np.float64(60.0), np.float64(90.0)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.radians_2"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_2'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_2'], lib="jax", suffix=2)
