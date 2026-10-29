
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arctanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values within (-1, 1)
    x = np.array([-0.8, -0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64 values
    x = np.array([[-0.9, -0.1], [0.1, 0.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float32)
    x = np.array(0.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float16 values
    x = np.zeros((2, 3, 4), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: Complex64 values (1D)
    x = np.array([-2.0 + 0.0j, 3.0 + 0.0j, 4.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Complex128 2D array
    x = np.array(
        [[0.5 + 0.5j, -0.5 + 0.5j], [0.2 - 0.3j, -0.2 - 0.3j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Integer array (int32) promoting to inexact dtype
    x = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array with random values between -0.99 and 0.99
    x = np.linspace(-0.99, 0.99, 24, dtype=np.float32).reshape((2, 2, 2, 3))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Boundary and out-of-domain values producing inf and nan
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Non-contiguous 2D slice
    full_arr = np.linspace(-0.9, 0.9, 25, dtype=np.float32).reshape(5, 5)
    x = full_arr[::2, ::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.arctanh_1"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctanh_1'.")


check_valid('jax.numpy.arctanh', generated_inputs['jax.numpy.arctanh_1'], lib="jax", suffix=1)
