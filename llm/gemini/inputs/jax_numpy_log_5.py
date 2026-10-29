
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive float32
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of positive float64
    x = np.array([0.5, 1.5, 2.5, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of floats
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of floats
    x = np.arange(1, 9, dtype=np.float32).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array of positive integers
    x = np.array([1, 2, 10, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array of integers
    x = np.array([[1, 5], [10, 50]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array of complex numbers
    x = np.array([1.0 + 1.0j, 2.0 - 3.0j, -1.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex128 array
    x = np.array([1.0j, -2.0j, 3.0 + 4.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D single-element array
    x = np.array([np.e], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D array of positive floats
    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Array with extreme positive float values
    x = np.array([1e-30, 1e-15, 1.0, 1e15, 1e30], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.log_5"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_5'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_5'], lib="jax", suffix=5)
