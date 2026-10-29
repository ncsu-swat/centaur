
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of the same shape
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with negative values
    x = np.array([[-1, -2], [3, 4]], dtype=np.int32)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D boolean arrays with broadcasting
    x = np.array([[True], [False], [True]], dtype=np.bool_)
    y = np.array([[False, True, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar arrays (int64)
    x = np.array(42, dtype=np.int64)
    y = np.array(-17, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D int32 arrays
    x = np.random.randint(0, 256, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(0, 256, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D int64 arrays
    x = np.random.randint(-1000, 1000, size=(2, 2, 3, 2), dtype=np.int64)
    y = np.random.randint(-1000, 1000, size=(2, 2, 3, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Broadcasting 1D array to 2D array (int64)
    x = np.array([1, 2, 4, 8], dtype=np.int64)
    y = np.array([[15], [0], [-1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int32 arrays with bitwise patterns
    x = np.array([[0x7FFFFFFF, 0x00000000], [0x12345678, 0x55555555]], dtype=np.int32)
    y = np.array([[0x55555555, 0x7FFFFFFF], [0x7F007F00, 0x007F007F]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Negative int32 values with broadcasting
    x = np.array([-128, -1, 0, 127], dtype=np.int32)
    y = np.array([[-1], [1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D bool arrays with broadcasting
    x = np.ones((2, 1, 4), dtype=np.bool_)
    y = np.zeros((1, 3, 4), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_xor_1"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_1'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_1'], lib="jax", suffix=1)
