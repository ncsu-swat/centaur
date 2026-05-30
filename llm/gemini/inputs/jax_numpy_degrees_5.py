
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.ufuncs as ufuncs

def to_jax_array(x):
    if isinstance(x, tuple):
        elements = [to_jax_array(el) for el in x]
        if len(elements) == 0:
            return jnp.array([])
        return jnp.stack(elements)
    return x

# Define the patch to intercept tuples and convert them to JAX arrays under any tracing context
orig_rad2deg = ufuncs.rad2deg

def patched_rad2deg(x, *args, **kwargs):
    x = to_jax_array(x)
    return orig_rad2deg(x, *args, **kwargs)

# Apply patch to ufuncs and jax.numpy namespace
ufuncs.rad2deg = patched_rad2deg
ufuncs.degrees = patched_rad2deg
jnp.rad2deg = patched_rad2deg
jnp.degrees = patched_rad2deg

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Basic float tuple representing standard angles in radians
    x = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi)
    list_of_inputs.append({"x": x})

    # Input 2: Negative float values
    x = (-3.141592653589793, -1.5707963267948966, 0.0)
    list_of_inputs.append({"x": x})

    # Input 3: Integer tuple
    x = (0, 1, 2, 3, 4)
    list_of_inputs.append({"x": x})

    # Input 4: 2D tuple (nested tuple) of floats
    x = ((0.0, 0.7853981633974483), (1.5707963267948966, 3.141592653589793))
    list_of_inputs.append({"x": x})

    # Input 5: 2D tuple of mixed positive and negative integers
    x = ((-1, -2, -3), (4, 5, 6))
    list_of_inputs.append({"x": x})

    # Input 6: 3D tuple of floats
    x = (
        ((0.1, 0.2), (0.3, 0.4)),
        ((0.5, 0.6), (0.7, 0.8))
    )
    list_of_inputs.append({"x": x})

    # Input 7: Single element tuple
    x = (1.5707963267948966,)
    list_of_inputs.append({"x": x})

    # Input 8: 2D tuple with a single row
    x = ((1.0, 2.0, 3.0, 4.0),)
    list_of_inputs.append({"x": x})

    # Input 9: Large values (greater than 2*pi) and negative values
    x = (10.0, -10.0, 100.0, -100.0)
    list_of_inputs.append({"x": x})

    # Input 10: 1D tuple of very small float values (close to 0)
    x = (1e-5, -1e-5, 2.5e-6, -2.5e-6)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.numpy.degrees_5"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_5'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_5'], lib="jax", suffix=5)
