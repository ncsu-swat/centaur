
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays, same shape
    x = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 arrays with negative values
    x = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    y = np.array([[5, -15], [-25, 35]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D boolean arrays
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[True, True], [False, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D int32 arrays with broadcasting (1,) and (5,)
    x = np.array([255], dtype=np.int32)
    y = np.array([1, 2, 4, 8, 16], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D int64 arrays with broadcasting (3, 1) and (3, 4)
    x = np.array([[1], [-2], [3]], dtype=np.int64)
    y = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D int32 arrays
    x = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    y = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 0D scalar array and 2D array
    x = np.array(42, dtype=np.int32)
    y = np.array([[10, 20], [30, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D bool arrays with broadcasting (2, 1, 4) and (2, 3, 4)
    x = np.array(
        [[[True, False, True, False]], [[False, True, False, True]]],
        dtype=np.bool_,
    )
    y = np.ones((2, 3, 4), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D int64 arrays
    x = np.array([0, -1, 1024, -2048], dtype=np.int64)
    y = np.array([-1, 0, -2048, 1024], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 0D scalar arrays
    x = np.array(-1, dtype=np.int64)
    y = np.array(1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.bitwise_xor_1"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_1'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_1'], lib="jax", suffix=1)
