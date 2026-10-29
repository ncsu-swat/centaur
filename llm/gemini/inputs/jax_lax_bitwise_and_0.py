
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with int32
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D arrays with int64, including negative values
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int64)
    y = np.array([[7, -8], [9, -10]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D arrays with broadcasting
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Boolean arrays 1D
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Boolean arrays 2D
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[True, True], [False, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalar arrays with int32
    x = np.array(12, dtype=np.int32)
    y = np.array(10, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 3D arrays with int32
    x = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    y = np.full((2, 2, 2), 15, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D arrays with int64, negative values
    x = np.array([-128, -1, 0, 127], dtype=np.int64)
    y = np.array([127, -1, 0, -128], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D arrays with int32 and broadcasting
    x = np.ones((2, 1, 3, 1), dtype=np.int32) * 15
    y = np.ones((2, 4, 1, 5), dtype=np.int32) * 7
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 0D scalar array broadcast with 2D array
    x = np.array(15, dtype=np.int32)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D arrays with boolean broadcasting
    x = np.array([[[True], [False]], [[False], [True]]], dtype=np.bool_)
    y = np.array([[[True, False]], [[False, True]]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_and'.")


check_valid('jax.lax.bitwise_and', generated_inputs['jax.lax.bitwise_and'], lib="jax", suffix=0)
