
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def shift_right_logical_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([16, -16, 1024, 0], dtype=np.int32)
    y = np.array([1, 2, 3, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with broadcasting
    x = np.array([[128], [256], [512]], dtype=np.int32)
    y = np.array([[1, 2, 3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D (scalar) int64 arrays
    x = np.array(1048576, dtype=np.int64)
    y = np.array(4, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 arrays
    x = np.random.randint(-1000, 1000, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(0, 16, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D int32 arrays with negative values
    x = np.array([-1, -32768, 100, 0], dtype=np.int32)
    y = np.array([1, 2, 0, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D int64 arrays
    x = np.random.randint(-50000, 50000, size=(1, 2, 2, 3), dtype=np.int64)
    y = np.random.randint(0, 32, size=(1, 2, 2, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int32 arrays with broadcasting
    x = np.array([[-128, -64, 32, 64, 127]], dtype=np.int32)
    y = np.array([[1], [2], [3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 arrays with large values
    x = np.array([9223372036854775807, -9223372036854775808, 256, 1], dtype=np.int64)
    y = np.array([62, 60, 2, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 3D int32 arrays with broadcast compatibility
    x = np.random.randint(-1000, 1000, size=(2, 1, 4), dtype=np.int32)
    y = np.random.randint(0, 31, size=(1, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D int64 arrays
    x = np.array([[-1, -2], [3, 4]], dtype=np.int64)
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D int32 zero shifts
    x = np.array([-128, -1, 0, 1, 127], dtype=np.int32)
    y = np.zeros(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.shift_right_logical"] = shift_right_logical_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.shift_right_logical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.shift_right_logical'.")


check_valid('jax.lax.shift_right_logical', generated_inputs['jax.lax.shift_right_logical'], lib="jax", suffix=0)
