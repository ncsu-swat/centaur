
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divmod_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D positive integers
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = np.array([3, 4, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D with negative numbers
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    x2 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Floating point numbers
    x1 = np.array([6.5, 7.2, 8.9], dtype=np.float32)
    x2 = np.array([1.5, 2.0, 3.1], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D arrays
    x1 = np.array([[12, 15], [22, 35]], dtype=np.int32)
    x2 = np.array([[5, 4], [7, 8]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Broadcasting 2D and 1D
    x1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    x2 = np.array([3, 7, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D float arrays
    x1 = np.arange(1, 25, dtype=np.float64).reshape((2, 3, 4))
    x2 = np.full((2, 3, 4), 2.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Mixed signs in floats
    x1 = np.array([-10.5, 10.5, -20.0, 20.0], dtype=np.float32)
    x2 = np.array([3.0, -3.0, -4.5, 4.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Scalar-like 0D arrays
    x1 = np.array(42, dtype=np.int32)
    x2 = np.array(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Unsigned integer types
    x1 = np.array([100, 200, 250], dtype=np.uint8)
    x2 = np.array([30, 40, 60], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Broadcasting with leading singleton dimension
    x1 = np.array([[50], [100]], dtype=np.int64)
    x2 = np.array([[3, 7, 11]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 4D arrays
    x1 = np.ones((2, 2, 2, 2), dtype=np.float32) * 17.0
    x2 = np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_1"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_1'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_1'], lib="jax", suffix=1)
