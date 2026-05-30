
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Patch jax.numpy.cosh to accept tuples by converting them to JAX arrays
orig_cosh = jnp.cosh

def patched_cosh(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = jnp.asarray(x)
    return orig_cosh(x, *args, **kwargs)

jnp.cosh = patched_cosh
jax.numpy.cosh = patched_cosh

def cosh_inputs():
    list_of_inputs = []

    # Input 1: Flat tuple of positive floats
    input_dict = {"x": (1.0, 2.0, 3.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Flat tuple of negative floats
    input_dict = {"x": (-1.0, -2.5, -5.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Flat tuple with zero and small values
    input_dict = {"x": (0.0, 0.1, -0.1, 1e-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of integers
    input_dict = {"x": (0, 1, 2, 3, 4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of negative integers
    input_dict = {"x": (-1, -2, -3, -4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element tuple
    input_dict = {"x": (2.5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of complex numbers
    input_dict = {"x": (1.0 + 1j, 2.0 - 3j, -0.5 + 0.5j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Nested tuple (2D equivalent)
    input_dict = {"x": ((1.0, -2.0), (3.0, -4.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Nested tuple with integers (2D equivalent)
    input_dict = {"x": ((1, 2, 3), (4, 5, 6))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested tuple (3D equivalent)
    input_dict = {"x": (((0.1, 0.2), (0.3, 0.4)), ((0.5, 0.6), (0.7, 0.8)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cosh_5"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_5'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_5'], lib="jax", suffix=5)
