
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Patch jax.numpy.atanh so that it accepts and converts tuple inputs
_original_atanh = jnp.atanh

def patched_atanh(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = jnp.array(x)
    return _original_atanh(x, *args, **kwargs)

jnp.atanh = patched_atanh

# Also patch arctanh since atanh is an alias of arctanh
_original_arctanh = jnp.arctanh

def patched_arctanh(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = jnp.array(x)
    return _original_arctanh(x, *args, **kwargs)

jnp.arctanh = patched_arctanh

# Ensure the patch is applied in sys.modules
import sys
if 'jax.numpy' in sys.modules:
    sys.modules['jax.numpy'].atanh = patched_atanh
    sys.modules['jax.numpy'].arctanh = patched_arctanh

def atanh_inputs():
    list_of_inputs = []

    # Input 1: 1D tuple of floats
    input_dict = {"x": (0.5, -0.5, 0.0, 0.9, -0.9)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single element tuple
    input_dict = {"x": (0.1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tuple (tuple of tuples)
    input_dict = {"x": ((0.1, -0.2), (0.3, -0.4))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tuple close to boundaries
    input_dict = {"x": (0.999, -0.999, 1e-7, -1e-7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tuple
    input_dict = {"x": (((0.1, 0.2), (0.3, 0.4)), ((0.5, 0.6), (0.7, 0.8)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple with complex elements
    input_dict = {"x": (0.5 + 0.5j, -0.2 - 0.1j)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tuple with complex elements
    input_dict = {"x": (((0.1 + 0.1j, 0.2 + 0.2j), (0.3 + 0.3j, 0.4 + 0.4j)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple of zeros
    input_dict = {"x": (0.0, 0.0, 0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple of numpy float32 scalars
    input_dict = {"x": (np.float32(0.1), np.float32(-0.5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D tuple of larger size
    input_dict = {"x": ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.atanh_5"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_5'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_5'], lib="jax", suffix=5)
