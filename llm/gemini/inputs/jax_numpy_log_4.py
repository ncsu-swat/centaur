
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

    # Input 2: 1D float64 array
    list_of_inputs.append({"x": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)})

    # Input 3: 2D float32 array
    list_of_inputs.append({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    # Input 4: 2D float64 array
    list_of_inputs.append({"x": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)})

    # Input 5: 3D float32 array
    list_of_inputs.append({"x": np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)})

    # Input 6: 1D array with very small positive values
    list_of_inputs.append({"x": np.array([1e-15, 1e-10, 1e-5], dtype=np.float32)})

    # Input 7: 1D array with very large positive values
    list_of_inputs.append({"x": np.array([1e5, 1e10, 1e15], dtype=np.float32)})

    # Input 8: Array containing negative numbers (valid for execution, returns NaN)
    list_of_inputs.append({"x": np.array([1.0, -2.0, 3.0], dtype=np.float32)})

    # Input 9: Single-element array
    list_of_inputs.append({"x": np.array([10.0], dtype=np.float32)})

    # Input 10: 1D int32 array (will be promoted to inexact type)
    list_of_inputs.append({"x": np.array([1, 2, 3, 4], dtype=np.int32)})

    return list_of_inputs

generated_inputs["jax.numpy.log_4"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_4'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_4'], lib="jax", suffix=4)
