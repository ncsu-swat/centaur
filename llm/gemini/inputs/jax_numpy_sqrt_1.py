
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sqrt_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar tensor), positive float32
    x = np.array(4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array, positive float32
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with negative values (results in NaN for real floats)
    x = np.array([-1.0, -4.0, 2.0, -9.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array, float64
    x = np.random.uniform(0.1, 100.0, size=(4, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array, integers (int32)
    x = np.array([[1, 4, 9], [16, 25, 36]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array, complex64
    x = np.array(
        [[[-8.0 - 6.0j, 1.0j], [4.0 + 0.0j, -4.0 + 0.0j]]], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array, complex128 with various values
    x = np.array([-1.0 + 0.0j, 0.0 + 1.0j, -2.0 - 3.0j, 5.0 + 12.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array, float16
    x = np.random.uniform(1.0, 50.0, size=(2, 3, 2, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing special float values (inf, 0, nan)
    x = np.array([0.0, np.inf, -0.0, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D array, unsigned integer (uint8)
    x = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sqrt_1"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_1'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_1'], lib="jax", suffix=1)
