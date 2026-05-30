
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax._src.numpy.ufuncs as ufuncs

orig_degrees = jnp.degrees

def patched_degrees(x, *args, **kwargs):
    if isinstance(x, list):
        x = jnp.array(x)
    return orig_degrees(x, *args, **kwargs)

jnp.degrees = patched_degrees
ufuncs.degrees = patched_degrees

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Basic positive radian angles
    list_of_inputs.append({"x": [0.0, 0.5, 1.0, 1.5707963267948966, 3.141592653589793]})

    # Input 2: Negative radian angles
    list_of_inputs.append({"x": [-3.141592653589793, -1.5707963267948966, -0.5, 0.0]})

    # Input 3: 2D list of floats
    list_of_inputs.append({"x": [[0.0, 0.7853981633974483], [1.5707963267948966, 3.141592653589793]]})

    # Input 4: 1D list of integers
    list_of_inputs.append({"x": [0, 1, 2, 3, 4]})

    # Input 5: 2D list of integers
    list_of_inputs.append({"x": [[1, 2], [3, 4]]})

    # Input 6: 3D list of floats
    list_of_inputs.append({"x": [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]]]})

    # Input 7: Single element list
    list_of_inputs.append({"x": [3.141592653589793]})

    # Input 8: List with special float values
    list_of_inputs.append({"x": [float('nan'), float('inf'), float('-inf'), 0.0]})

    # Input 9: List with very small float values
    list_of_inputs.append({"x": [1e-10, -1e-10, 5e-8]})

    # Input 10: Standard float angles
    list_of_inputs.append({"x": [1.0, 2.0, 3.0, 4.0]})

    return list_of_inputs

generated_inputs["jax.numpy.degrees_4"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_4'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_4'], lib="jax", suffix=4)
