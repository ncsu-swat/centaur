
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log2_inputs():
    list_of_inputs = []

    # Input 1: 1D positive float32 array
    x = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D positive float64 array
    x = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D array) float32
    x = np.array(64.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int32 array
    x = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array with special values: zero, infinity, nan
    x = np.array([0.0, np.inf, np.nan, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with negative numbers
    x = np.array([-4.0, -2.0, -1.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D int64 array
    x = np.array([[1, 2], [4, 8]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float32 array
    x = np.ones((2, 1, 3, 2), dtype=np.float32) * 4.0
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large values in float64
    x = np.array([1e10, 1e20, 1e30], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Subnormal / very small positive float32 values
    x = np.array([1e-10, 1e-20, 1e-30], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.log2"] = log2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log2'.")


check_valid('jax.numpy.log2', generated_inputs['jax.numpy.log2'], lib="jax", suffix=0)
