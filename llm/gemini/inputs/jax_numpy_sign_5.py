
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax.numpy as jnp
import copy

_original_sign = jnp.sign

def mocked_sign(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = jnp.asarray(x)
    return _original_sign(x, *args, **kwargs)

jnp.sign = mocked_sign

def jax_numpy_sign_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tuple of integers
    list_of_inputs.append({"x": (0, -1, 2, -3, 4)})

    # Input 2: 1D tuple of floats with negative zero
    list_of_inputs.append({"x": (-1.5, 0.0, 2.3, -0.0, 9.9)})

    # Input 3: 1D tuple of complex numbers
    list_of_inputs.append({"x": (1 + 1j, -3 + 4j, -5j, 0 + 0j)})

    # Input 4: 2D tuple of integers
    list_of_inputs.append({"x": ((1, -2), (-3, 4))})

    # Input 5: 2D tuple of floats
    list_of_inputs.append({"x": ((-0.5, 0.5), (1.5, -1.5), (0.0, -0.0))})

    # Input 6: 3D tuple of integers
    list_of_inputs.append({"x": (((1, -1), (2, -2)), ((3, -3), (4, -4)))})

    # Input 7: Tuple with large and small float values
    list_of_inputs.append({"x": (1e-20, -1e20, 0.0, -1.0, 1.0)})

    # Input 8: Single-element tuple
    list_of_inputs.append({"x": (-999.99,)})

    # Input 9: 2D tuple of complex numbers
    list_of_inputs.append({"x": ((1 - 1j, -2 + 2j), (3j, -4j))})

    # Input 10: 1D tuple of integers with 10 elements
    list_of_inputs.append({"x": (1, 2, 3, 4, 5, -5, -4, -3, -2, -1)})

    return list_of_inputs

generated_inputs["jax.numpy.sign_5"] = jax_numpy_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_5'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_5'], lib="jax", suffix=5)
