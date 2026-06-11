
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clamp_inputs():
    list_of_inputs = []

    # Input 1: float32, all 1D arrays of the same size
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min_val = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    max_val = np.array([4.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, x is 2D, min and max are 0D (scalars as tensors)
    x = np.random.randn(3, 3).astype(np.float32)
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(0.5, dtype=np.float32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, negative values, 1D arrays
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    min_val = np.array([-2, -2, -2, -2, -2], dtype=np.int32)
    max_val = np.array([3, 3, 3, 3, 3], dtype=np.int32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, 3D arrays, same shapes
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float64)
    min_val = np.full((2, 2, 2), -1.0, dtype=np.float64)
    max_val = np.full((2, 2, 2), 1.0, dtype=np.float64)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, same shape 2D arrays
    x = np.random.randn(2, 4).astype(np.float32)
    min_val = np.full((2, 4), -1.0, dtype=np.float32)
    max_val = np.full((2, 4), 1.0, dtype=np.float32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32, 1D array x, scalar min and max
    x = np.arange(10, dtype=np.int32)
    min_val = np.array(3, dtype=np.int32)
    max_val = np.array(7, dtype=np.int32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 4D arrays, scalar min/max
    x = np.random.randn(1, 2, 2, 1).astype(np.float32)
    min_val = np.array(-0.1, dtype=np.float32)
    max_val = np.array(0.1, dtype=np.float32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32, 2D arrays, same shape
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    min_val = np.array([[15, 15], [25, 25]], dtype=np.int32)
    max_val = np.array([[35, 35], [35, 35]], dtype=np.int32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 5D arrays, scalar min/max
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    min_val = np.array(-2.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32, 1D arrays, same shape
    x = np.array([0, 10, 20, 30], dtype=np.int32)
    min_val = np.array([5, 5, 5, 5], dtype=np.int32)
    max_val = np.array([25, 25, 25, 25], dtype=np.int32)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.clamp_1"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_1'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_1'], lib="jax", suffix=1)
