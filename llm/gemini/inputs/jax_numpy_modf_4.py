
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.ufuncs as _ufuncs

# Patch JAX modf to accept lists and ignore 'out' because JAX does not support it
_original_modf = jnp.modf

def _patched_modf(x, *args, **kwargs):
    if isinstance(x, list):
        x = np.array(x)
    return _original_modf(x)

jnp.modf = _patched_modf
_ufuncs.modf = _patched_modf

def modf_inputs():
    list_of_inputs = []

    # Input 1, 1D positive floats
    x = [0.5, 1.2, 3.9, 10.5]
    out = np.zeros((4,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 2, 1D negative floats
    x = [-0.5, -1.2, -3.9, -10.5]
    out = np.zeros((4,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 3, 2D list of mixed floats
    x = [[1.5, -2.3], [3.7, -4.1]]
    out = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 4, 1D list of float64-like precision
    x = [1.23456789, -9.87654321]
    out = np.zeros((2,), dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 5, 3D list
    x = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
    out = np.zeros((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 6, integer values in list
    x = [1, 2, -3, 4, -5]
    out = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 7, large float values
    x = [1e5 + 0.123, -2e6 - 0.456]
    out = np.zeros((2,), dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 8, list with a single element
    x = [3.14159]
    out = np.zeros((1,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 9, list with infinity and NaN values
    x = [float('nan'), float('inf'), float('-inf'), 0.0]
    out = np.zeros((4,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    # Input 10, 4D list
    x = [[[[1.1, -1.1]]]]
    out = np.zeros((1, 1, 1, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "out": copy.deepcopy(out)})

    return list_of_inputs

generated_inputs["jax.numpy.modf_4"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_4'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_4'], lib="jax", suffix=4)
