
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with same shape
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 arrays with negative values
    x = np.array([[-1, -2], [3, 4]], dtype=np.int64)
    y = np.array([[7, -8], [-5, 6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Boolean arrays (1D)
    x = np.array([True, True, False, False], dtype=np.bool_)
    y = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar arrays (int32)
    x = np.array(12, dtype=np.int32)
    y = np.array(25, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcasting 1D to 2D int32 arrays
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D int64 arrays with broadcasting along middle dimension
    x = np.ones((2, 1, 4), dtype=np.int64) * 15
    y = np.ones((2, 3, 4), dtype=np.int64) * 7
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D boolean arrays with broadcasting
    x = np.array([[True], [False], [True]], dtype=np.bool_)
    y = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 arrays with large values
    x = np.array([1000000000, -2000000000], dtype=np.int64)
    y = np.array([500000000, 3000000000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty int32 arrays with matching shapes
    x = np.empty((0, 5), dtype=np.int32)
    y = np.empty((0, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: int32 arrays with 0D broadcast to 2D
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    y = np.array(7, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_and'.")


check_valid('jax.numpy.bitwise_and', generated_inputs['jax.numpy.bitwise_and'], lib="jax", suffix=0)
