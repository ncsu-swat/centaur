
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sin_inputs():
    list_of_inputs = []

    # Input 1: Basic python integer (positive)
    list_of_inputs.append({"x": 5})

    # Input 2: Basic python integer (zero)
    list_of_inputs.append({"x": 0})

    # Input 3: Basic python integer (negative)
    list_of_inputs.append({"x": -10})

    # Input 4: NumPy int32 scalar
    list_of_inputs.append({"x": np.int32(45)})

    # Input 5: NumPy int64 scalar (negative)
    list_of_inputs.append({"x": np.int64(-123)})

    # Input 6: 1D NumPy array of int32
    list_of_inputs.append({"x": np.array([0, 1, -1, 2], dtype=np.int32)})

    # Input 7: 2D NumPy array of int64
    list_of_inputs.append({"x": np.array([[1, 2], [-3, -4]], dtype=np.int64)})

    # Input 8: NumPy int32 scalar
    list_of_inputs.append({"x": np.int32(8)})

    # Input 9: 3D NumPy array of int32
    list_of_inputs.append({"x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)})

    # Input 10: NumPy int64 scalar
    list_of_inputs.append({"x": np.int64(100)})

    return list_of_inputs

generated_inputs["jax.numpy.sin_3"] = jax_numpy_sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_3'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_3'], lib="jax", suffix=3)
