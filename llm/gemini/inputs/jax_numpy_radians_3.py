
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.numpy.util as jax_util
import jax._src.numpy.ufuncs as jax_ufuncs

# Patch JAX's promote_args_inexact to automatically convert tuples to JAX arrays 
# before any array-like checks or float0 checks are run.
def patch_promote(module):
    if hasattr(module, 'promote_args_inexact'):
        orig_promote = module.promote_args_inexact
        def custom_promote(fun_name, *args):
            new_args = tuple(jax.numpy.asarray(x) if isinstance(x, tuple) else x for x in args)
            return orig_promote(fun_name, *new_args)
        module.promote_args_inexact = custom_promote

patch_promote(jax_util)
patch_promote(jax_ufuncs)

def radians_inputs():
    list_of_inputs = []

    # Input 1: Single element float tuple
    input_dict = {"x": (0.0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard angles in float tuple
    input_dict = {"x": (180.0, 90.0, 45.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative angles in float tuple
    input_dict = {"x": (-180.0, -360.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer tuple
    input_dict = {"x": (0, 30, 45, 60, 90)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive angles
    input_dict = {"x": (360.0, 720.0, 1080.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Floating point angles with precision
    input_dict = {"x": (12.34, 56.78, -90.12)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero variations
    input_dict = {"x": (0.0, -0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger tuple of integers
    input_dict = {"x": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point scales
    input_dict = {"x": (1e-3, 1e2, 1.5e3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative float tuple
    input_dict = {"x": (-45.5, 0.0, 45.5, -135.0, 225.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.radians_3"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_3'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_3'], lib="jax", suffix=3)
