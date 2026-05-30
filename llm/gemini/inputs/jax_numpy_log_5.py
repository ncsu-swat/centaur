
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

try:
    import jax
    import jax.numpy as jnp
    import jax._src.numpy.util as jax_util
    import jax._src.numpy.ufuncs as jax_ufuncs
    
    original_promote = jax_util.promote_args_inexact
    
    def patched_promote(fun_name, *args):
        new_args = [jnp.asarray(arg) if isinstance(arg, tuple) else arg for arg in args]
        return original_promote(fun_name, *new_args)
        
    jax_util.promote_args_inexact = patched_promote
    jax_ufuncs.promote_args_inexact = patched_promote
except Exception:
    pass

def log_inputs():
    list_of_inputs = []

    # Input 1: 1D tuple of positive floats
    input_dict = {"x": (1.0, 2.0, 3.0, 4.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D nested tuple of floats
    input_dict = {"x": ((1.0, 2.0), (3.0, 4.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of integers
    input_dict = {"x": (1, 10, 100, 1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single element tuple
    input_dict = {"x": (0.5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple with very small positive float values
    input_dict = {"x": (1e-5, 1e-10, 1e-15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple with very large float values
    input_dict = {"x": (1e10, 1e20, 1e30)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D nested tuple of floats
    input_dict = {"x": (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple with complex numbers
    input_dict = {"x": (1.0 + 0.0j, -1.0 + 0.0j, -2.5 + 1.5j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple representing a 1x5 row vector
    input_dict = {"x": ((0.1, 0.2, 0.3, 0.4, 0.5),)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple of powers of 2
    input_dict = {"x": (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.log_5"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_5'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_5'], lib="jax", suffix=5)
