
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with common angles
    x = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array with negative angles
    x = np.array([-np.pi, -np.pi / 2, -np.pi / 4, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with single element
    x = np.array([np.pi / 2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array of integers
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array of integers
    x = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D float32 matrix
    x = np.linspace(-np.pi, np.pi, 12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D float32 tensor
    x = np.ones((2, 3, 4), dtype=np.float32) * (np.pi / 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D boolean array
    x = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D complex numbers array
    x = np.array([1.0 + 1.0j, -0.5 + 2.0j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 4D float32 array with extreme values
    x = np.array([[[[1e-7]], [[-1e-7]]], [[[1e5]], [[-1e5]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Special values (inf, -inf, nan)
    x = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sin_5"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_5'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_5'], lib="jax", suffix=5)
