
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eq_inputs():
    list_of_inputs = []

    # Input 1: Scalar integers (int32)
    x = np.int32(5)
    y = np.int32(5)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of int32, matching shapes
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 0, 3, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays of int64, matching shapes, negative values included
    x = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    y = np.array([[1, 2], [-3, -4]], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array and a scalar (broadcasting), int32
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.int32(20)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays with broadcasting (1, 3) and (4, 1), int32
    x = np.array([[1, 2, 3]], dtype=np.int32)
    y = np.array([[1], [2], [3], [4]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays of int16, matching shapes
    x = np.ones((2, 2, 2), dtype=np.int16)
    y = np.zeros((2, 2, 2), dtype=np.int16)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D arrays of int8, matching shapes, negative values
    x = np.array([-128, 0, 127], dtype=np.int8)
    y = np.array([-128, 1, 127], dtype=np.int8)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array and scalar, int64
    x = np.array([[100, 200], [300, 400]], dtype=np.int64)
    y = np.int64(200)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays with broadcasting (2, 1, 3) and (1, 4, 3), int32
    x = np.ones((2, 1, 3), dtype=np.int32)
    y = np.zeros((1, 4, 3), dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D arrays of int64
    x = np.arange(100, dtype=np.int64)
    y = np.arange(100, dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.eq_2"] = eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_2'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_2'], lib="jax", suffix=2)
