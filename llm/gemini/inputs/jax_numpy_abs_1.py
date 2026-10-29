
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive and negative floats
    x = np.array([-2.5, 0.0, 3.14, -100.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of integers
    x = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor (float64)
    x = np.array(-42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array with float16
    x = np.random.randn(2, 3, 4).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D complex64 array
    x = np.array([3.0 + 4.0j, -1.0 - 1.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D complex128 array
    x = np.array([[-5.0 + 12.0j, 1.0 - 2.0j], [0.0 + 3.0j, -4.0 + 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D array of int16
    x = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with signed int8
    x = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array with inf and nan values
    x = np.array([[np.inf, -np.inf], [np.nan, -0.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D array of int64
    x = np.random.randint(-1000, 1000, size=(3, 1, 5), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.abs_1"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_1'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_1'], lib="jax", suffix=1)
