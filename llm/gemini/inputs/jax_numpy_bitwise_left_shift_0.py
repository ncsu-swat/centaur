
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_left_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([1, 2, 4, 8], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 arrays with negative values for x
    x = np.array([[-1, -2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    y = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 2D and 1D (int64)
    x = np.random.randint(-1000, 1000, size=(3, 3)).astype(np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar-like 0D arrays (int64)
    x = np.array(5, dtype=np.int64)
    y = np.array(2, dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 arrays with positive values
    x = np.array([1024, 2048, 4096], dtype=np.int32)
    y = np.array([5, 10, 15], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Signed int32 with negative and zero values
    x = np.array([-128, -64, -1, 0, 1, 127], dtype=np.int32)
    y = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger shape 3D (int64)
    x = np.random.randint(-50, 50, size=(5, 5, 5)).astype(np.int64)
    y = np.random.randint(0, 5, size=(5, 5, 5)).astype(np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcastable shapes (1, 5) and (5, 1) in int32
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    y = np.array([[1], [2], [3], [4], [5]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 1D arrays
    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([8, 8, 8], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_left_shift"] = bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_left_shift'.")


check_valid('jax.numpy.bitwise_left_shift', generated_inputs['jax.numpy.bitwise_left_shift'], lib="jax", suffix=0)
